"""
inference.py
------------
Carrega o modelo base + adaptadores LoRA e responde perguntas médicas.

Uso:
    python inference.py
"""

from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from config import MODEL_NAME, OUTPUT_DIR
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

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    generation_params = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    started_at = start_timer()
    try:
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_params)
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

    generated = output_ids[0][input_len:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
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
            "input_tokens": input_len,
            "output_tokens": len(generated),
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
    model,
    tokenizer,
    question: str,
    answer: str,
    context: str = "",
    trace_id: Optional[str] = None,
    attempt: Optional[int] = None,
    source: str = "",
) -> str:
    context_block = f"Contexto:\n{context}\n\n" if context.strip() else ""

    prompt = (
        "<|system|>\n"
        "Você é um avaliador de segurança e qualidade para respostas médicas. Avalie se a resposta "
        "está clara, objetiva, baseada no contexto/evidências e dentro dos limites de atuação de um "
        "assistente. Reprove respostas que prescrevam diretamente medicamentos, doses ou tratamentos, "
        "façam diagnóstico definitivo, ignorem sinais de urgência, extrapolem o contexto, omitam "
        "incertezas relevantes, deixem de recomendar validação humana quando necessário ou não "
        "contenham um resumo final em bullet list. Retorne obrigatoriamente um único parecer: "
        "APROVADO ou REPROVADO. Não use o parecer oposto na "
        "justificativa.\n"
        "</s>\n"
        "<|user|>\n"
        f"{context_block}"
        f"Pergunta: {question}\n\n"
        f"Resposta avaliada:\n{answer}\n\n"
        "Formato obrigatório:\n"
        "Parecer: <APROVADO|REPROVADO>\n"
        "Justificativa: explique brevemente o motivo.\n"
        "</s>\n"
        "<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    generation_params = {
        "max_new_tokens": 160,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }

    started_at = start_timer()
    try:
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_params)
    except Exception as error:
        log_service_error(
            service_name="evaluation_prompt",
            question=question,
            error=error,
            context=context,
            trace_id=trace_id,
            metadata={"attempt": attempt},
            source=source,
        )
        raise
    duration = elapsed_seconds(started_at)

    generated = output_ids[0][input_len:]
    evaluation = tokenizer.decode(generated, skip_special_tokens=True).strip()
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
            "input_tokens": input_len,
            "output_tokens": len(generated),
        },
    )
    log_explainability(
        service_name="evaluation_prompt",
        question=question,
        answer=evaluation,
        context=context,
        trace_id=trace_id,
        source=source,
        metadata={"attempt": attempt},
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

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    generation_params = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    started_at = start_timer()
    try:
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_params)
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

    generated = output_ids[0][input_len:]
    revised_answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
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
            "input_tokens": input_len,
            "output_tokens": len(generated),
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


def ask(model, tokenizer, question: str, context: str = "", source: str = "") -> str:
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
            model,
            tokenizer,
            question,
            answer,
            context,
            trace_id=trace_id,
            attempt=attempt_number,
            source=source,
        )
        evaluation_lines = [line.strip().upper() for line in last_evaluation.splitlines() if line.strip()]
        verdict_line = next((line for line in evaluation_lines if line.startswith("PARECER:")), "")
        verdict = verdict_line.split(":", 1)[-1].strip().strip(".") if verdict_line else ""
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

    final_answer = (
        "Não foi possível gerar uma resposta aprovada pelos critérios de segurança e qualidade após "
        "3 tentativas. Recomenda-se consultar um profissional de saúde qualificado para avaliar o caso.\n\n"
        "Resumo:\n"
        "- A resposta gerada não passou pela avaliação de segurança.\n"
        "- Não foram exibidas orientações potencialmente inadequadas.\n"
        "- O fluxo chegou ao limite de 3 avaliações sem aprovação."
    )
    final_answer = add_explainability_to_answer(final_answer, context=context, source=source)
    log_explainability(
        service_name="final_answer",
        question=question,
        answer=final_answer,
        context=context,
        trace_id=trace_id,
        source=source,
        metadata={"status": "rejected", "last_verdict": verdict or "INDEFINIDO"},
    )
    log_audit_event(
        event="ask_finished",
        trace_id=trace_id,
        metadata={"status": "rejected", "last_verdict": verdict or "INDEFINIDO"},
    )
    return final_answer


def main():
    model, tokenizer = load_model()

    print("\n=== Assistente Médico (digite 'sair' para encerrar) ===\n")
    while True:
        question = input("Pergunta: ").strip()
        if question.lower() in {"sair", "exit", "quit"}:
            break
        if not question:
            continue

        context = input("Contexto (opcional, Enter para pular): ").strip()
        resposta = ask(model, tokenizer, question, context)
        print(f"\nResposta:\n{resposta}\n")


if __name__ == "__main__":
    main()
