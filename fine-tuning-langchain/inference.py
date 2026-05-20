"""
inference.py
------------
Carrega o modelo base + adaptadores LoRA e responde perguntas médicas.

Uso:
    python inference.py
"""

import json
from typing import Dict, Optional, TypedDict

import torch
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline as transformers_pipeline
from prontuario import buscar_prontuario_por_cpf, montar_contexto_prontuario, cpf_existe_no_prontuario
from config import EVALUATION_MODEL_NAME, MODEL_NAME, OLLAMA_BASE_URL, OUTPUT_DIR
from logs import (
    add_explainability_to_answer,
    elapsed_seconds,
    log_audit_event,
    log_explainability,
    log_service_call,
    log_service_error,
    new_trace_id,
    setup_logging,
    start_timer,
)


LANGCHAIN_PROMPT = PromptTemplate.from_template("{prompt}")


class MedicalAssistantState(TypedDict):
    model: object
    tokenizer: object
    evaluation_llm: object
    question: str
    context: str
    source: str
    trace_id: str
    answer: str
    evaluation: str
    verdict: str
    attempt: int
    final_answer: str
    cpf: str


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model():
    setup_logging()
    log_audit_event(
        event="model_load_started",
        metadata={"model_name": MODEL_NAME, "output_dir": OUTPUT_DIR},
    )

    print("[inference] Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("[inference] Carregando modelo base (4-bit)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=build_bnb_config(),
        device_map="auto",
    )

    print(f"[inference] Aplicando adaptadores LoRA de '{OUTPUT_DIR}'...")
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()

    log_audit_event(
        event="model_load_finished",
        metadata={"model_name": MODEL_NAME, "output_dir": OUTPUT_DIR},
    )

    return model, tokenizer


def load_evaluation_model():
    setup_logging()
    log_audit_event(
        event="evaluation_model_load_started",
        metadata={
            "provider": "ollama",
            "model_name": EVALUATION_MODEL_NAME,
            "base_url": OLLAMA_BASE_URL,
        },
    )

    print(f"[inference] Carregando avaliador Ollama '{EVALUATION_MODEL_NAME}'...")
    evaluation_llm = ChatOllama(
        model=EVALUATION_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_predict=160,
        reasoning=False,
        format="json",
        validate_model_on_init=True,
    )

    log_audit_event(
        event="evaluation_model_load_finished",
        metadata={
            "provider": "ollama",
            "model_name": EVALUATION_MODEL_NAME,
            "base_url": OLLAMA_BASE_URL,
        },
    )

    return evaluation_llm


def count_tokens(tokenizer, text: str, add_special_tokens: bool = True) -> int:
    return len(tokenizer(text or "", add_special_tokens=add_special_tokens)["input_ids"])


def generate_with_langchain(model, tokenizer, prompt: str, generation_params: Dict[str, object]) -> str:
    text_generation_pipeline = transformers_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    chain = LANGCHAIN_PROMPT | llm.bind(
        pipeline_kwargs={**generation_params, "return_full_text": False},
    )
    return chain.invoke({"prompt": prompt}).strip()

def call_initial_prompt(
    model,
    tokenizer,
    question: str,
    context: str = "",
    trace_id: Optional[str] = None,
    source: str = "",
) -> str:
    context_block = f"Contexto:\n{context}\n\n" if context.strip() else ""

    prompt = (
        "<|system|>\n"
        "Você é um assistente médico especializado. Responda com base no contexto fornecido "
        "de forma clara, objetiva e baseada em evidências. Não prescreva medicamentos, doses, "
        "diagnósticos fechados ou mudanças de tratamento sem validação humana de um profissional "
        "de saúde. Quando houver risco, incerteza ou limitação no contexto, deixe isso explícito "
        "e recomende avaliação profissional. Responda com a resposta principal e, ao final, um "
        "resumo em bullet list com os principais tópicos.\n"
        "</s>\n"
        "<|user|>\n"
        f"{context_block}"
        f"Pergunta: {question}\n"
        "</s>\n"
        "<|assistant|>\n"
    )

    generation_params = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    input_tokens = count_tokens(tokenizer, prompt)
    started_at = start_timer()
    try:
        answer = generate_with_langchain(model, tokenizer, prompt, generation_params)
    except Exception as error:
        log_service_error(
            service_name="initial_prompt",
            question=question,
            error=error,
            context=context,
            trace_id=trace_id,
            source=source,
        )
        raise
    duration = elapsed_seconds(started_at)
    output_tokens = count_tokens(tokenizer, answer, add_special_tokens=False)
    log_service_call(
        service_name="initial_prompt",
        question=question,
        prompt=prompt,
        response=answer,
        context=context,
        trace_id=trace_id,
        generation_params=generation_params,
        duration_seconds=duration,
        source=source,
        metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    )
    log_explainability(
        service_name="initial_prompt",
        question=question,
        answer=answer,
        context=context,
        trace_id=trace_id,
        source=source,
    )
    return answer

def call_evaluation_prompt(
    evaluation_llm,
    question: str,
    answer: str,
    context: str = "",
    trace_id: Optional[str] = None,
    attempt: Optional[int] = None,
    source: str = "",
) -> str:
    context_block = f"Contexto:\n{context}\n\n" if context.strip() else ""

    system_prompt = (
        "Você é um avaliador de segurança e qualidade para respostas médicas. Avalie se a resposta "
        "está clara, objetiva, baseada no contexto/evidências e dentro dos limites de atuação de um "
        "assistente. Reprove respostas que prescrevam diretamente medicamentos, doses ou tratamentos, "
        "façam diagnóstico definitivo, ignorem sinais de urgência, extrapolem o contexto, omitam "
        "incertezas relevantes, deixem de recomendar validação humana quando necessário ou não "
        "contenham um resumo final em bullet list. Retorne obrigatoriamente um único parecer: "
        "APROVADO ou REPROVADO. Não use o parecer oposto na justificativa. "
        "Responda somente JSON válido, sem markdown, no formato "
        '{"parecer":"APROVADO ou REPROVADO","justificativa":"máximo 20 palavras"}.'
    )
    user_prompt = (
        f"{context_block}"
        f"Pergunta: {question}\n\n"
        f"Resposta avaliada:\n{answer}\n\n"
        "Retorne somente o JSON solicitado."
    )
    prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"

    generation_params = {
        "format": "json",
        "num_predict": 160,
        "temperature": 0,
        "reasoning": False,
    }

    started_at = start_timer()
    try:
        response = evaluation_llm.invoke(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )
        evaluation = normalize_evaluation_response(response.content)
    except Exception as error:
        log_service_error(
            service_name="evaluation_prompt",
            question=question,
            error=error,
            context=context,
            trace_id=trace_id,
            metadata={
                "attempt": attempt,
                "provider": "ollama",
                "model_name": EVALUATION_MODEL_NAME,
            },
            source=source,
        )
        raise
    duration = elapsed_seconds(started_at)
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    log_service_call(
        service_name="evaluation_prompt",
        question=question,
        prompt=prompt,
        response=evaluation,
        context=context,
        trace_id=trace_id,
        generation_params=generation_params,
        duration_seconds=duration,
        source=source,
        metadata={
            "attempt": attempt,
            "provider": "ollama",
            "model_name": EVALUATION_MODEL_NAME,
            "input_tokens": usage_metadata.get("input_tokens"),
            "output_tokens": usage_metadata.get("output_tokens"),
            "total_tokens": usage_metadata.get("total_tokens"),
        },
    )
    log_explainability(
        service_name="evaluation_prompt",
        question=question,
        answer=evaluation,
        context=context,
        trace_id=trace_id,
        source=source,
        metadata={
            "attempt": attempt,
            "provider": "ollama",
            "model_name": EVALUATION_MODEL_NAME,
        },
    )
    return evaluation

def call_revision_prompt(
    model,
    tokenizer,
    question: str,
    answer: str,
    evaluation: str,
    context: str = "",
    trace_id: Optional[str] = None,
    attempt: Optional[int] = None,
    source: str = "",
) -> str:
    context_block = f"Contexto:\n{context}\n\n" if context.strip() else ""

    prompt = (
        "<|system|>\n"
        "Você é um assistente médico especializado revisando uma resposta que foi reprovada. Gere "
        "uma nova resposta clara, objetiva, baseada em evidências e alinhada aos limites de segurança: "
        "não prescreva medicamentos, doses, diagnósticos fechados ou mudanças de tratamento sem "
        "validação humana de um profissional de saúde; não extrapole o contexto; explicite incertezas "
        "e recomende avaliação profissional quando necessário. Responda com a resposta principal e, "
        "ao final, um resumo em bullet list com os principais tópicos.\n"
        "</s>\n"
        "<|user|>\n"
        f"{context_block}"
        f"Pergunta: {question}\n\n"
        f"Resposta anterior:\n{answer}\n\n"
        f"Avaliação recebida:\n{evaluation}\n\n"
        "Gere uma nova resposta corrigida.\n"
        "</s>\n"
        "<|assistant|>\n"
    )

    generation_params = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    input_tokens = count_tokens(tokenizer, prompt)
    started_at = start_timer()
    try:
        revised_answer = generate_with_langchain(model, tokenizer, prompt, generation_params)
    except Exception as error:
        log_service_error(
            service_name="revision_prompt",
            question=question,
            error=error,
            context=context,
            trace_id=trace_id,
            metadata={"attempt": attempt},
            source=source,
        )
        raise
    duration = elapsed_seconds(started_at)
    output_tokens = count_tokens(tokenizer, revised_answer, add_special_tokens=False)
    log_service_call(
        service_name="revision_prompt",
        question=question,
        prompt=prompt,
        response=revised_answer,
        context=context,
        trace_id=trace_id,
        generation_params=generation_params,
        duration_seconds=duration,
        source=source,
        metadata={
            "attempt": attempt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    )
    log_explainability(
        service_name="revision_prompt",
        question=question,
        answer=revised_answer,
        context=context,
        trace_id=trace_id,
        source=source,
        metadata={"attempt": attempt},
    )
    return revised_answer


def normalize_evaluation_response(raw_response: str) -> str:
    raw_response = (raw_response or "").strip()
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        return raw_response

    parecer = str(payload.get("parecer", "")).strip().upper().replace("*", "")
    if "REPROVADO" in parecer:
        parecer = "REPROVADO"
    elif "APROVADO" in parecer:
        parecer = "APROVADO"

    justificativa = str(payload.get("justificativa", "")).strip()
    return (
        f"Parecer: {parecer or 'INDEFINIDO'}\n"
        f"Justificativa: {justificativa or 'Sem justificativa retornada.'}"
    )


def extract_evaluation_verdict(evaluation: str) -> str:
    evaluation_lines = [line.strip().upper() for line in evaluation.splitlines() if line.strip()]
    verdict_line = next((line for line in evaluation_lines if line.startswith("PARECER:")), "")
    verdict = verdict_line.split(":", 1)[-1].strip().strip(".") if verdict_line else ""
    verdict = verdict.replace("*", "").strip()

    if "REPROVADO" in verdict:
        return "REPROVADO"
    if "APROVADO" in verdict:
        return "APROVADO"
    return verdict


def ask_without_langgraph(
    model,
    tokenizer,
    evaluation_llm,
    question: str,
    context: str = "",
    source: str = "",
) -> str:
    trace_id = new_trace_id()
    log_audit_event(
        event="ask_started",
        trace_id=trace_id,
        metadata={"has_context": bool(context.strip())},
    )

    answer = call_initial_prompt(model, tokenizer, question, context, trace_id=trace_id, source=source)
    last_evaluation = ""
    verdict = ""

    for attempt in range(3):
        attempt_number = attempt + 1
        last_evaluation = call_evaluation_prompt(
            evaluation_llm,
            question,
            answer,
            context,
            trace_id=trace_id,
            attempt=attempt_number,
            source=source,
        )
        verdict = extract_evaluation_verdict(last_evaluation)
        log_audit_event(
            event="evaluation_verdict",
            trace_id=trace_id,
            metadata={"attempt": attempt_number, "verdict": verdict or "INDEFINIDO"},
        )

        if verdict == "APROVADO":
            final_answer = add_explainability_to_answer(answer, context=context, source=source)
            log_explainability(
                service_name="final_answer",
                question=question,
                answer=final_answer,
                context=context,
                trace_id=trace_id,
                source=source,
                metadata={"status": "approved", "attempt": attempt_number},
            )
            log_audit_event(
                event="ask_finished",
                trace_id=trace_id,
                metadata={"status": "approved", "attempt": attempt_number},
            )
            return final_answer

        if attempt < 2:
            answer = call_revision_prompt(
                model,
                tokenizer,
                question,
                answer,
                last_evaluation,
                context,
                trace_id=trace_id,
                attempt=attempt_number,
                source=source,
            )
        else:
            final_answer = answer
    final_answer = add_explainability_to_answer(final_answer, context=context, source=source)
    log_explainability(
        service_name="final_answer",
        question=question,
        answer=final_answer,
        context=context,
        trace_id=trace_id,
        source=source,
        metadata={
            "status": "best_effort_unapproved",
            "attempt": attempt_number,
            "last_verdict": verdict or "INDEFINIDO",
        },
    )
    log_audit_event(
        event="ask_finished",
        trace_id=trace_id,
        metadata={
            "status": "best_effort_unapproved",
            "attempt": attempt_number,
            "last_verdict": verdict or "INDEFINIDO",
        },
    )
    return final_answer


def generate_initial_answer_node(state: MedicalAssistantState) -> MedicalAssistantState:
    answer = call_initial_prompt(
        state["model"],
        state["tokenizer"],
        state["question"],
        state["context"],
        trace_id=state["trace_id"],
        source=state["source"],
    )
    return {**state, "answer": answer, "attempt": 1}


def evaluate_answer_node(state: MedicalAssistantState) -> MedicalAssistantState:
    evaluation = call_evaluation_prompt(
        state["evaluation_llm"],
        state["question"],
        state["answer"],
        state["context"],
        trace_id=state["trace_id"],
        attempt=state["attempt"],
        source=state["source"],
    )
    verdict = extract_evaluation_verdict(evaluation)

    log_audit_event(
        event="evaluation_verdict",
        trace_id=state["trace_id"],
        metadata={"attempt": state["attempt"], "verdict": verdict or "INDEFINIDO"},
    )
    return {**state, "evaluation": evaluation, "verdict": verdict}


def revise_answer_node(state: MedicalAssistantState) -> MedicalAssistantState:
    revised_answer = call_revision_prompt(
        state["model"],
        state["tokenizer"],
        state["question"],
        state["answer"],
        state["evaluation"],
        state["context"],
        trace_id=state["trace_id"],
        attempt=state["attempt"],
        source=state["source"],
    )
    return {**state, "answer": revised_answer, "attempt": state["attempt"] + 1}


def approve_answer_node(state: MedicalAssistantState) -> MedicalAssistantState:
    final_answer = add_explainability_to_answer(
        state["answer"],
        context=state["context"],
        source=state["source"],
    )
    log_explainability(
        service_name="final_answer",
        question=state["question"],
        answer=final_answer,
        context=state["context"],
        trace_id=state["trace_id"],
        source=state["source"],
        metadata={"status": "approved", "attempt": state["attempt"]},
    )
    log_audit_event(
        event="ask_finished",
        trace_id=state["trace_id"],
        metadata={"status": "approved", "attempt": state["attempt"]},
    )
    return {**state, "final_answer": final_answer}


def return_best_answer_node(state: MedicalAssistantState) -> MedicalAssistantState:
    final_answer = state["answer"]
    final_answer = add_explainability_to_answer(
        final_answer,
        context=state["context"],
        source=state["source"],
    )
    log_explainability(
        service_name="final_answer",
        question=state["question"],
        answer=final_answer,
        context=state["context"],
        trace_id=state["trace_id"],
        source=state["source"],
        metadata={
            "status": "best_effort_unapproved",
            "attempt": state["attempt"],
            "last_verdict": state["verdict"] or "INDEFINIDO",
        },
    )
    log_audit_event(
        event="ask_finished",
        trace_id=state["trace_id"],
        metadata={
            "status": "best_effort_unapproved",
            "attempt": state["attempt"],
            "last_verdict": state["verdict"] or "INDEFINIDO",
        },
    )
    return {**state, "final_answer": final_answer}


def route_after_evaluation(state: MedicalAssistantState) -> str:
    if state["verdict"] == "APROVADO":
        return "approve"
    if state["attempt"] >= 3:
        return "best_effort"
    return "revise"

def route_prontuario(state: MedicalAssistantState) -> str:
    if state["cpf"] == "":
        return "question"
    return "prontuario"


def build_prontuario(state: MedicalAssistantState) -> MedicalAssistantState:
    prontuario = buscar_prontuario_por_cpf(state["cpf"])
    contexto_prontuario = montar_contexto_prontuario(prontuario)
    contextos = [contexto_prontuario, state["context"]]
    context = "\n\n".join(contexto.strip() for contexto in contextos if contexto and contexto.strip())

    return {
        **state,
        "context": context,
        "source": "base_mock_prontuarios.xlsx",
    }


def build_medical_assistant_graph():
    graph = StateGraph(MedicalAssistantState)
    
    graph.add_node("generate_initial_answer", generate_initial_answer_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)
    graph.add_node("revise_answer", revise_answer_node)
    graph.add_node("approve_answer", approve_answer_node)
    graph.add_node("return_best_answer", return_best_answer_node)
    graph.add_node("build_prontuario", build_prontuario)

    graph.set_conditional_entry_point(route_prontuario, {
        "question" : "generate_initial_answer",
        "prontuario" : "build_prontuario"
    })
    graph.add_edge("build_prontuario", "generate_initial_answer")
    graph.add_edge("generate_initial_answer", "evaluate_answer")
    graph.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "approve": "approve_answer",
            "revise": "revise_answer",
            "best_effort": "return_best_answer",
        },
    )
    graph.add_edge("revise_answer", "evaluate_answer")
    graph.add_edge("approve_answer", END)
    graph.add_edge("return_best_answer", END)

    return graph.compile()


def ask(
    model,
    tokenizer,
    evaluation_llm,
    question: str,
    context: str = "",
    source: str = "",
    cpf: str = "",
) -> str:
    trace_id = new_trace_id()
    log_audit_event(
        event="ask_started",
        trace_id=trace_id,
        metadata={"has_context": bool(context.strip())},
    )

    app = build_medical_assistant_graph()

    print(app.get_graph().draw_ascii())  

    initial_state: MedicalAssistantState = {
        "model": model,
        "tokenizer": tokenizer,
        "evaluation_llm": evaluation_llm,
        "question": question,
        "context": context,
        "source": source,
        "trace_id": trace_id,
        "answer": "",
        "evaluation": "",
        "verdict": "",
        "attempt": 0,
        "final_answer": "",
        "cpf" : cpf
    }
    final_state = app.invoke(initial_state)
    return final_state["final_answer"]


def main():
    model, tokenizer = load_model()
    evaluation_llm = load_evaluation_model()

    print("\n=== Assistente Médico (digite 'sair' para encerrar) ===\n")
    while True:
        cpf = input("Informar o CPF para busca de prontuario (opcional): ").strip()
        question = ""
        context = ""

        if cpf:
            existe_cpf = cpf_existe_no_prontuario(cpf)
            if (not existe_cpf):
                print("Paciente não encontrado")
                continue
            question = (
                "Analise o prontuario do paciente com base nos atendimentos, internacoes e alergias "
                "disponiveis. Aponte pontos de atencao, possiveis riscos e informacoes que devem ser "
                "validadas por um profissional de saude. Nao prescreva medicamentos, doses ou condutas "
                "definitivas."
            )

        if not cpf:
            question = input("Pergunta: ").strip()
            if question.lower() in {"sair", "exit", "quit"}:
                break
            if not question:
                continue

            context = input("Contexto (opcional, Enter para pular): ").strip()

        resposta = ask(model, tokenizer, evaluation_llm, question, context, "", cpf)
        print(f"\nResposta:\n{resposta}\n")


if __name__ == "__main__":
    main()
