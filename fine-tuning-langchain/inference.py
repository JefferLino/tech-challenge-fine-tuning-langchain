"""
inference.py
------------
Carrega o modelo base + adaptadores LoRA e responde perguntas medicas.

O fluxo de inferencia usa LangChain em tres etapas:
1. geracao inicial da resposta;
2. avaliacao critica da resposta;
3. formatacao final em bullets com fontes e ressalvas.

Uso:
    python inference.py
"""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from uuid import uuid4

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODEL_NAME, OUTPUT_DIR

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda
except ImportError:
    PromptTemplate = None
    RunnableLambda = None


LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "inference_audit.log"
AUDIT_FULL_TEXT = os.getenv("AUDIT_FULL_TEXT", "0").lower() in {"1", "true", "yes"}
SENSITIVE_LOG_FIELDS = {
    "question",
    "context",
    "prompt",
    "answer",
    "evaluation",
    "final_answer",
    "initial_answer",
}

SAFETY_BOUNDARIES = """
- Nao prescrever medicamentos, doses, ajustes de dose ou tratamentos de forma direta.
- Nao substituir avaliacao, diagnostico, prescricao ou acompanhamento de profissional de saude.
- Nao afirmar diagnostico definitivo; use linguagem probabilistica quando houver incerteza.
- Em sinais de emergencia, orientar busca imediata por atendimento medico.
- Usar a fonte informada no prompt; se nao houver contexto, declarar essa limitacao.
- Nao inventar fontes, estudos, resultados ou dados ausentes do contexto.
""".strip()

INITIAL_RESPONSE_PROMPT = """
<|system|>
Voce e um assistente medico especializado. Responda de forma clara, objetiva e baseada em evidencias.

Limites obrigatorios de atuacao:
{safety_boundaries}

Explique a base da resposta e indique explicitamente a fonte utilizada.
</s>
<|user|>
Fonte disponivel: {source}
{context_block}
Pergunta: {question}

Responda no formato:
Resposta:
Justificativa:
Fonte:
</s>
<|assistant|>
""".strip()

EVALUATION_PROMPT = """
<|system|>
Voce e um avaliador medico e de seguranca. Sua tarefa e verificar se a resposta inicial faz sentido,
se esta coerente com a pergunta, se respeita os limites de atuacao e se declara a fonte corretamente.

Limites obrigatorios:
{safety_boundaries}

Retorne uma avaliacao objetiva no formato:
Status: APROVADA ou REVISAR
Coerencia:
Riscos:
Fonte:
Acao recomendada:
</s>
<|user|>
Pergunta: {question}
Fonte disponivel: {source}
{context_block}
Resposta inicial:
{initial_answer}

Sinais deterministicos de possivel violacao:
{guardrail_findings}
</s>
<|assistant|>
""".strip()

FINAL_FORMAT_PROMPT = """
<|system|>
Voce e um editor clinico. Formate a resposta final como um resumo em bullets, usando apenas linhas
iniciadas por "-". Preserve prudencia medica, fonte utilizada e necessidade de validacao humana.

Se a avaliacao indicar REVISAR ou riscos relevantes, corrija a resposta para uma versao segura,
sem prescrever diretamente e sem extrapolar a fonte.
</s>
<|user|>
Pergunta: {question}
Fonte disponivel: {source}
Resposta inicial:
{initial_answer}

Avaliacao:
{evaluation}

Produza bullets contendo:
- resposta central
- principais pontos
- ressalvas ou incertezas
- fonte utilizada
- orientacao de validacao humana
</s>
<|assistant|>
""".strip()

UNSAFE_DIRECTIVE_RE = re.compile(
    r"\b(tome|use|inicie|comece|aumente|reduza|suspenda|pare|interrompa|prescreva)\b",
    flags=re.IGNORECASE,
)
UNSAFE_DOSAGE_RE = re.compile(
    r"\b\d+([,.]\d+)?\s?(mg|mcg|g|ml|ui|iu|comprimidos?|capsulas?|gotas?)\b",
    flags=re.IGNORECASE,
)
DEFINITIVE_DIAGNOSIS_RE = re.compile(
    r"\b(voce tem|o paciente tem|diagnostico definitivo|com certeza e)\b",
    flags=re.IGNORECASE,
)


def configure_logging() -> logging.Logger:
    """Configura logging de auditoria sem poluir a interface interativa."""
    logger = logging.getLogger("inference.audit")
    if logger.handlers:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


LOGGER = configure_logging()


def _redact_or_hash(text: str) -> dict:
    text = text or ""
    payload = {
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    if AUDIT_FULL_TEXT:
        payload["text"] = text
    return payload


def audit_event(event: str, request_id=None, **fields) -> None:
    payload = {
        "event": event,
        "request_id": request_id,
        "timestamp_unix": round(time.time(), 3),
    }

    for key, value in fields.items():
        if isinstance(value, str) and key in SENSITIVE_LOG_FIELDS:
            payload[key] = _redact_or_hash(value)
        else:
            payload[key] = value

    LOGGER.info(json.dumps(payload, ensure_ascii=False, default=str))


def require_langchain() -> None:
    if PromptTemplate is None or RunnableLambda is None:
        raise RuntimeError(
            "LangChain nao esta instalado. Execute `pip install -r requirements.txt` "
            "dentro da pasta fine-tuning-langchain antes de rodar a inferencia."
        )


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model():
    audit_event("model_load_started", model_name=MODEL_NAME, output_dir=OUTPUT_DIR)

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

    audit_event("model_load_finished", model_name=MODEL_NAME, output_dir=OUTPUT_DIR)
    return model, tokenizer


def _model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _generate_text(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(_model_device(model))
    input_len = inputs["input_ids"].shape[1]

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_args.update({"temperature": temperature, "top_p": 0.9})

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_args)

    generated = output_ids[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _context_block(context: str) -> str:
    if not context.strip():
        return ""
    return f"Contexto:\n{context.strip()}\n"


def _source_for_context(context: str) -> str:
    if context.strip():
        return "Contexto fornecido pelo usuario nesta sessao."
    return (
        "Sem contexto externo fornecido; resposta baseada apenas no conhecimento parametrico "
        f"do modelo local {MODEL_NAME} ajustado com adaptadores LoRA em {OUTPUT_DIR}."
    )


def _detect_guardrail_findings(answer: str) -> str:
    findings = []
    has_directive = bool(UNSAFE_DIRECTIVE_RE.search(answer or ""))
    has_dosage = bool(UNSAFE_DOSAGE_RE.search(answer or ""))

    if has_directive and has_dosage:
        findings.append("Possivel prescricao direta com dose ou quantidade.")
    if DEFINITIVE_DIAGNOSIS_RE.search(answer or ""):
        findings.append("Possivel diagnostico definitivo sem validacao humana.")
    if "Fonte:" not in (answer or "") and "fonte" not in (answer or "").lower():
        findings.append("Fonte nao declarada explicitamente.")

    return "\n".join(f"- {item}" for item in findings) if findings else "- Nenhum sinal deterministico relevante."


def _prompt(template: str, **kwargs) -> str:
    require_langchain()
    return PromptTemplate.from_template(template).format(**kwargs)


def build_langchain_flow(model, tokenizer):
    require_langchain()

    def initial_step(payload: dict) -> dict:
        request_id = payload["request_id"]
        prompt = _prompt(
            INITIAL_RESPONSE_PROMPT,
            safety_boundaries=SAFETY_BOUNDARIES,
            source=payload["source"],
            context_block=_context_block(payload["context"]),
            question=payload["question"],
        )

        audit_event("initial_prompt_built", request_id, prompt=prompt)
        answer = _generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=256,
            temperature=0.7,
        )
        audit_event("initial_answer_generated", request_id, answer=answer)
        return {**payload, "initial_answer": answer}

    def evaluation_step(payload: dict) -> dict:
        request_id = payload["request_id"]
        guardrail_findings = _detect_guardrail_findings(payload["initial_answer"])
        prompt = _prompt(
            EVALUATION_PROMPT,
            safety_boundaries=SAFETY_BOUNDARIES,
            source=payload["source"],
            context_block=_context_block(payload["context"]),
            question=payload["question"],
            initial_answer=payload["initial_answer"],
            guardrail_findings=guardrail_findings,
        )

        audit_event(
            "evaluation_prompt_built",
            request_id,
            prompt=prompt,
            guardrail_findings=guardrail_findings,
        )
        evaluation = _generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=180,
            temperature=0.0,
        )
        audit_event("answer_evaluated", request_id, evaluation=evaluation)
        return {
            **payload,
            "evaluation": evaluation,
            "guardrail_findings": guardrail_findings,
        }

    def final_format_step(payload: dict) -> str:
        request_id = payload["request_id"]
        prompt = _prompt(
            FINAL_FORMAT_PROMPT,
            source=payload["source"],
            question=payload["question"],
            initial_answer=payload["initial_answer"],
            evaluation=payload["evaluation"],
        )

        audit_event("final_prompt_built", request_id, prompt=prompt)
        final_answer = _generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=220,
            temperature=0.2,
        )
        final_answer = _enforce_final_contract(final_answer, payload["source"])
        audit_event("final_answer_generated", request_id, final_answer=final_answer)
        return final_answer

    initial_model = RunnableLambda(initial_step).with_config(
        run_name="modelo_resposta_inicial"
    )
    evaluation_model = RunnableLambda(evaluation_step).with_config(
        run_name="modelo_avaliador"
    )
    formatter_model = RunnableLambda(final_format_step).with_config(
        run_name="modelo_formatador"
    )

    return initial_model | evaluation_model | formatter_model


def _enforce_final_contract(answer: str, source: str) -> str:
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]

    if not lines:
        lines = [
            "Nao foi possivel gerar uma resposta confiavel com as informacoes disponiveis.",
        ]

    normalized = []
    for line in lines:
        if line.startswith(("-", "*")):
            normalized.append("- " + line.lstrip("-* ").strip())
        else:
            normalized.append("- " + line)

    joined = "\n".join(normalized)
    if "fonte" not in joined.lower():
        normalized.append(f"- Fonte utilizada: {source}")
    if "validacao" not in joined.lower() and "profissional" not in joined.lower():
        normalized.append(
            "- Validacao humana: use esta resposta apenas como apoio informativo; "
            "decisoes clinicas devem ser confirmadas por um profissional de saude."
        )

    return "\n".join(normalized)


def ask(model, tokenizer, question: str, context: str = "") -> str:
    request_id = str(uuid4())
    source = _source_for_context(context)
    audit_event(
        "request_received",
        request_id,
        question=question,
        context=context,
        source=source,
        context_present=bool(context.strip()),
    )

    chain = build_langchain_flow(model, tokenizer)
    response = chain.invoke(
        {
            "request_id": request_id,
            "question": question.strip(),
            "context": context.strip(),
            "source": source,
        }
    )
    audit_event("request_finished", request_id, final_answer=response)
    return response


def main():
    model, tokenizer = load_model()

    print("\n=== Assistente Medico (digite 'sair' para encerrar) ===\n")
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
