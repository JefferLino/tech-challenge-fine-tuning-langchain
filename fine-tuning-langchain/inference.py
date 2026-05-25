"""
inference.py
------------
Carrega o modelo base + adaptadores LoRA e responde perguntas médicas.

Uso:
    python inference.py
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, TypedDict

import torch
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline as transformers_pipeline
from transformers.utils import logging as transformers_logging
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
transformers_logging.set_verbosity_error()
BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vector"
RAG_TOP_K = 3


class MedicalAssistantState(TypedDict):
    model: object
    tokenizer: object
    evaluation_llm: object
    original: str
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
    prontuario: str


@lru_cache(maxsize=1)
def load_vector_index():
    chunks_path = VECTOR_DIR / "chunks.json"
    embeddings_path = VECTOR_DIR / "embeddings.pt"
    config_path = VECTOR_DIR / "config.json"

    if not chunks_path.exists() or not embeddings_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Indice RAG nao encontrado. Rode 'python process_rag.py' antes de usar o RAG."
        )

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    embeddings = torch.load(embeddings_path, map_location="cpu")
    embedding_model = SentenceTransformer(config["embedding_model"])
    return chunks, embeddings, embedding_model


def retrieve_context(question: str, top_k: int = RAG_TOP_K) -> str:
    chunks, embeddings, embedding_model = load_vector_index()
    if not chunks:
        return ""

    query_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).cpu()
    scores = embeddings @ query_embedding
    selected = torch.topk(scores, k=min(top_k, len(chunks)))

    contexts = []
    for score, index in zip(selected.values.tolist(), selected.indices.tolist()):
        chunk = chunks[index]
        contexts.append(
            f"Fonte: {chunk.get('source', 'desconhecida')} | Score: {score:.4f}\n"
            f"{chunk.get('text', '')}"
        )

    return "\n\n".join(contexts)


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
        num_predict=80,
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
        "Classify the question using only the context. "
        "Do not use outside knowledge. "
        "The evidence must be copied or directly paraphrased from the context. "
        "If the context does not answer the question, use MAYBE. "
        "Return exactly this format:\n"
        "Answer: YES/NO/MAYBE\n"
        "Evidence: one short sentence from the context that supports the answer and adds a detail not already present in the question.\n"
        "</s>\n"
        "<|user|>\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "</s>\n"
        "<|assistant|>\n"
        "Answer:" 
    )

    generation_params = {
        "max_new_tokens": 256,
        "do_sample": False,
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
    system_prompt = f"""You are a safety evaluator for medical assistant answers.
Evaluate only whether the answer violates basic medical safety rules.
Do not evaluate whether the answer is factually correct, complete, well written, or grounded in the context.

Reject only if the answer:
- prescribes, recommends a dose, starts, stops, switches, increases, or decreases medication;
- presents medication guidance as a treatment plan for the user;
- claims to replace a qualified healthcare professional.

The "verdict" field must be exactly "APPROVED" or exactly "REJECTED".
Return only this JSON, with no extra text:
{{"verdict":"APPROVED","reason":"short reason"}}

Question:
{question}

Answer:
{answer}
"""
    
    prompt = f"{system_prompt}"

    generation_params = {
        "format": "json",
        "num_predict": 80,
        "temperature": 0,
        "reasoning": False,
    }

    started_at = start_timer()
    try:
        response = evaluation_llm.invoke(system_prompt)

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


    prompt = f"""You are a medical education assistant trained with documents.
        Use only the provided context .
        Write a descriptive answer in English.
        Do not recommend, prescribe, dose, start, stop, or adjust any medication.
        If the context is insufficient, say that the dataset does not contain enough information.

        Question:
        {question}

        Context:
        {context_block}

        Answer:"""
    


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

    verdict = str(payload.get("verdict", "")).strip().upper().replace("*", "")
    if "REJECTED" in verdict:
        verdict = "REJECTED"
    elif "APPROVED" in verdict:
        verdict = "APPROVED"

    reason = str(payload.get("reason", "")).strip()
    return (
        f"Verdict: {verdict or 'UNDEFINED'}\n"
        f"Reason: {reason or 'No reason returned.'}"
    )


def extract_evaluation_verdict(evaluation: str) -> str:
    evaluation_lines = [line.strip().upper() for line in evaluation.splitlines() if line.strip()]
    verdict_line = next((line for line in evaluation_lines if line.startswith("VERDICT:")), "")
    verdict = verdict_line.split(":", 1)[-1].strip().strip(".") if verdict_line else ""
    verdict = verdict.replace("*", "").strip()

    if "REJECTED" in verdict:
        return "REJECTED"
    if "APPROVED" in verdict:
        return "APPROVED"
    return verdict


def generate_initial_answer_node(state: MedicalAssistantState) -> MedicalAssistantState:
    answer = call_initial_prompt(
        state["model"],
        state["tokenizer"],
        state["question"],
        state["context"],
        trace_id=state["trace_id"],
        source=state["source"],
    )
    return {**state, "answer": answer, "attempt": 1, "original" : answer}


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
        metadata={"attempt": state["attempt"], "verdict": verdict or "UNDEFINED"},
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
    if state["verdict"] == "APPROVED":
        return "approve"
    if state["attempt"] >= 3:
        return "best_effort"
    return "revise"

def route_prontuario(state: MedicalAssistantState) -> str:
    if state["cpf"] == "":
        return "rag"
    return "prontuario"


def retrieve_context_node(state: MedicalAssistantState) -> MedicalAssistantState:
    if state["context"].strip():
        return state

    context = retrieve_context(state["question"])
    if not context:
        return state

    return {
        **state,
        "context": context,
        "source": str(VECTOR_DIR),
    }


def build_prontuario(state: MedicalAssistantState) -> MedicalAssistantState:
    prontuario = buscar_prontuario_por_cpf(state["cpf"])
    contexto_prontuario = montar_contexto_prontuario(prontuario)

    return {
        **state,
        "prontuario": contexto_prontuario,
    }


def build_medical_assistant_graph():
    graph = StateGraph(MedicalAssistantState)
    
    graph.add_node("generate_initial_answer", generate_initial_answer_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)
    graph.add_node("revise_answer", revise_answer_node)
    graph.add_node("approve_answer", approve_answer_node)
    graph.add_node("return_best_answer", return_best_answer_node)
    graph.add_node("build_prontuario", build_prontuario)
    graph.add_node("retrieve_context", retrieve_context_node)

    graph.set_conditional_entry_point(route_prontuario, {
        "rag" : "retrieve_context",
        "prontuario" : "build_prontuario"
    })
    graph.add_edge("build_prontuario", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_initial_answer")
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
        "cpf" : cpf,
        "prontuario": "",
    }
    final_state = app.invoke(initial_state)

    final_answer = final_state["final_answer"]
   

    if "prontuario" in final_state:
        prontuario = final_state["prontuario"]
        final_answer = final_answer + "\n\nProntuário Médico:\n" + prontuario
 
    

    return final_answer


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

        question = input("Pergunta: ").strip()
        if question.lower() in {"sair", "exit", "quit"}:
            break
        if not question:
            continue
        
        context = input("Contexto (opcional, Enter para pular): ").strip()

        resposta = ask(model, tokenizer, evaluation_llm, question, context, "", cpf)
        print(f"\n\n{resposta}\n")


if __name__ == "__main__":
    main()
