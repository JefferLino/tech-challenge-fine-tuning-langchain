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
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest
from uuid import uuid4

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    MODEL_BACKEND,
    MODEL_NAME,
    OLLAMA_FALLBACK_MODELS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OUTPUT_DIR,
)

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda
except ImportError:
    PromptTemplate = None
    RunnableLambda = None


LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "inference_audit.log"
SESSION_LOG_DIR = LOG_DIR / "sessions"
ADAPTER_CONFIG_NAME = "adapter_config.json"
VALID_BACKENDS = {"auto", "ollama", "hf"}
MIN_CUDA_4BIT_VRAM_GB = float(os.getenv("HF_4BIT_MIN_VRAM_GB", "4"))
DEFAULT_CPU_MAX_MEMORY = os.getenv("HF_CPU_MAX_MEMORY", "12GiB")
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
- Usar apenas os dados clinicos fornecidos pelo usuario.
- Quando faltarem dados importantes, indicar que eles precisam ser avaliados.
- Nao mencionar tecnologia, backend, modelo, Ollama, Hugging Face, LangChain, BigAI,
  inteligencia artificial, conhecimento parametrico ou fontes internas do sistema.
""".strip()

INITIAL_RESPONSE_PROMPT = """
<|system|>
Voce e um assistente medico especializado. Responda de forma clara, objetiva e baseada em evidencias.

Limites obrigatorios de atuacao:
{safety_boundaries}

Gere orientacoes clinicas gerais e prudentes para o problema descrito.
Use frases simples. Nao use Markdown alem de texto comum. Nao use numeracao.
</s>
<|user|>
{context_block}
Pergunta: {question}

Responda no formato:
Resumo clinico:
Recomendacoes gerais:
Sinais de alerta:
Proximos passos:
</s>
<|assistant|>
""".strip()

EVALUATION_PROMPT = """
<|system|>
Voce e um avaliador medico e de seguranca. Sua tarefa e verificar se a resposta inicial faz sentido,
se esta coerente com a pergunta e se respeita os limites de atuacao.

Limites obrigatorios:
{safety_boundaries}

Retorne uma avaliacao objetiva no formato:
Status: APROVADA ou REVISAR
Coerencia:
Riscos:
Acao recomendada:
</s>
<|user|>
Pergunta: {question}
{context_block}
Resposta inicial:
{initial_answer}

Sinais deterministicos de possivel violacao:
{guardrail_findings}
</s>
<|assistant|>
""".strip()

REVISION_PROMPT = """
<|system|>
Voce e um assistente medico especializado em gerar respostas clinicas seguras e aderentes.
Use a resposta inicial e a avaliacao anterior para corrigir ou melhorar a resposta.

Limites obrigatorios:
{safety_boundaries}

Pergunta: {question}
{context_block}
Resposta inicial:
{initial_answer}

Avaliacao do avaliador:
{evaluation}

Gere uma nova resposta no formato:
Resumo clinico:
Recomendacoes gerais:
Sinais de alerta:
Proximos passos:
</s>
<|assistant|>
""".strip()

FINAL_FORMAT_PROMPT = """
<|system|>
Voce e um assistente clinico. Produza somente recomendacoes gerais ao paciente,
em bullets iniciados por "-".

Limites obrigatorios:
{safety_boundaries}

Nao mencione tecnologia, backend, modelo, Ollama, Hugging Face, LangChain, BigAI,
inteligencia artificial, conhecimento parametrico, fonte utilizada ou fonte interna.
Nao inclua status de avaliacao, "APROVADA", "REVISAR", "OK", titulos, numeracao,
cabecalhos Markdown, negrito ou secoes de fonte. Cada bullet deve ser uma
recomendacao clinica direta.
</s>
<|user|>
Pergunta: {question}
{context_block}

Retorne de 4 a 7 bullets contendo:
- cuidados gerais seguros
- medidas de acompanhamento
- o que evitar
- quando procurar atendimento medico
- quando procurar atendimento urgente
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
OLLAMA_ROLE_TAG_RE = re.compile(r"<\|(system|user|assistant)\|>")
FORBIDDEN_PUBLIC_TERMS = (
    "ollama",
    "hugging face",
    "langchain",
    "bigai",
    "backend",
    "algoritmo",
    "inteligencia artificial",
    "conhecimento parametrico",
    "modelo local",
    "modelo de linguagem",
    "fonte",
)
FINAL_LABEL_PREFIXES = (
    "status",
    "coerencia",
    "riscos",
    "acao recomendada",
    "avaliacao",
    "resposta",
    "principais pontos",
    "ressalvas ou incertezas",
    "avaliacao inicial",
    "resumo clinico",
    "recomendacoes gerais",
    "sinais de alerta",
    "proximos passos",
    "orientacao de validacao humana",
)
FINAL_SKIP_LINES = {"aprovada", "revisar", "ok", "nenhuma", "nenhum"}
MAX_EVALUATION_ATTEMPTS = 3
EVALUATION_STATUS_RE = re.compile(
    r"^Status\s*:\s*(APROVADA|REVISAR)\b",
    flags=re.IGNORECASE | re.MULTILINE,
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


def _create_session_log_file() -> Path:
    SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    session_file = SESSION_LOG_DIR / (
        f"session_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.txt"
    )
    with session_file.open("w", encoding="utf-8") as handle:
        handle.write(
            "Sessao iniciada: "
            f"{datetime.now():%Y-%m-%d %H:%M:%S}\n"
        )
        handle.write("=== Registro de sessao ===\n\n")
    return session_file


def _append_session_log(session_log_path: Path, label: str, content: str) -> None:
    with session_log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"--- {datetime.now():%Y-%m-%d %H:%M:%S} - {label} ---\n"
        )
        handle.write(content.rstrip() + "\n\n")


def _close_session_log(session_log_path: Path) -> None:
    with session_log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "Sessao finalizada: "
            f"{datetime.now():%Y-%m-%d %H:%M:%S}\n"
        )


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


class OllamaUnavailable(RuntimeError):
    """Falha esperada ao tentar usar um servidor Ollama local."""


class OllamaModel:
    backend = "ollama"

    def __init__(self, model_name: str, host: str, preferred_model: str):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.preferred_model = preferred_model

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        options = {
            "num_predict": max_new_tokens,
            "temperature": max(temperature, 0.0),
        }
        if temperature > 0:
            options["top_p"] = 0.9

        messages = _ollama_messages_from_prompt(prompt)
        if messages:
            response = _ollama_request(
                self.host,
                "/api/chat",
                payload={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": options,
                },
                timeout=300,
            )
            message = response.get("message") or {}
            return (message.get("content") or "").strip()

        response = _ollama_request(
            self.host,
            "/api/generate",
            payload={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
            timeout=300,
        )
        return (response.get("response") or "").strip()


def _ollama_messages_from_prompt(prompt: str) -> list[dict] | None:
    matches = list(OLLAMA_ROLE_TAG_RE.finditer(prompt))
    if not matches:
        return None

    messages = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        content = prompt[start:end].replace("</s>", "").strip()
        if content:
            messages.append({"role": match.group(1), "content": content})

    return messages or None


def _ollama_url(host: str, path: str) -> str:
    return f"{host.rstrip('/')}/{path.lstrip('/')}"


def _ollama_request(
    host: str,
    path: str,
    *,
    payload: dict | None = None,
    timeout: int = 10,
) -> dict:
    data = None
    headers = {}
    method = "GET"

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"

    request = urlrequest.Request(
        _ollama_url(host, path),
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaUnavailable(f"Ollama retornou HTTP {exc.code}: {detail}") from exc
    except (OSError, TimeoutError, urlerror.URLError) as exc:
        raise OllamaUnavailable(
            f"Ollama indisponivel em {host}. Inicie com `ollama serve`."
        ) from exc

    if not body:
        return {}

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise OllamaUnavailable("Ollama retornou uma resposta invalida.") from exc


def _list_ollama_models(host: str) -> list[str]:
    payload = _ollama_request(host, "/api/tags", timeout=5)
    names = []
    for item in payload.get("models", []):
        for key in ("name", "model"):
            value = item.get(key)
            if value and value not in names:
                names.append(value)
    return names


def _match_ollama_model(installed_models: list[str], candidate: str) -> str | None:
    if candidate in installed_models:
        return candidate
    if ":" not in candidate and f"{candidate}:latest" in installed_models:
        return f"{candidate}:latest"
    if candidate.endswith(":latest"):
        short_name = candidate.rsplit(":", 1)[0]
        if short_name in installed_models:
            return short_name
    return None


def _select_ollama_model(installed_models: list[str]) -> str:
    candidates = [OLLAMA_MODEL]
    candidates.extend(
        model for model in OLLAMA_FALLBACK_MODELS if model not in candidates
    )

    for candidate in candidates:
        match = _match_ollama_model(installed_models, candidate)
        if match:
            return match

    installed = ", ".join(installed_models) if installed_models else "nenhum"
    raise OllamaUnavailable(
        f"O modelo recomendado '{OLLAMA_MODEL}' nao esta baixado no Ollama. "
        f"Modelos locais encontrados: {installed}. "
        f"Execute `ollama pull {OLLAMA_MODEL}` para usar o backend Ollama recomendado."
    )


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )


def resolve_output_dir() -> Path:
    output_dir = Path(OUTPUT_DIR)
    if output_dir.is_absolute():
        return output_dir
    return Path(__file__).resolve().parent / output_dir


def has_lora_adapter(output_dir: Path) -> bool:
    return (output_dir / ADAPTER_CONFIG_NAME).is_file()


def _fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return without_accents.lower()


def _mentions_internal_system(text: str) -> bool:
    folded = _fold_text(text)
    return any(term in folded for term in FORBIDDEN_PUBLIC_TERMS)


def _cuda_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / 1024**3


def _cuda_name() -> str:
    if not torch.cuda.is_available():
        return "CUDA indisponivel"
    return torch.cuda.get_device_name(0)


def _hf_max_memory() -> dict:
    vram_gb = _cuda_vram_gb()
    if vram_gb <= 0:
        return {"cpu": DEFAULT_CPU_MAX_MEMORY}

    usable_mib = max(512, int(vram_gb * 1024 * 0.85))
    return {0: f"{usable_mib}MiB", "cpu": DEFAULT_CPU_MAX_MEMORY}


def _should_use_hf_4bit() -> bool:
    return torch.cuda.is_available() and _cuda_vram_gb() >= MIN_CUDA_4BIT_VRAM_GB


def _load_hf_base_model_cpu():
    print("[inference] Carregando modelo Hugging Face em CPU sem BitsAndBytes...")
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": "cpu"},
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )


def load_ollama_model():
    host = OLLAMA_HOST.rstrip("/")
    audit_event(
        "model_load_started",
        backend="ollama",
        model_name=OLLAMA_MODEL,
        host=host,
    )

    installed_models = _list_ollama_models(host)
    selected_model = _select_ollama_model(installed_models)

    if selected_model != OLLAMA_MODEL:
        print(
            "[inference] AVISO: modelo Ollama recomendado "
            f"'{OLLAMA_MODEL}' nao encontrado; usando '{selected_model}'."
        )
        print(f"[inference] Para usar o recomendado: ollama pull {OLLAMA_MODEL}")

    print(f"[inference] Usando Ollama local: {selected_model}")
    model = OllamaModel(selected_model, host, OLLAMA_MODEL)

    audit_event(
        "model_load_finished",
        backend="ollama",
        model_name=selected_model,
        preferred_model=OLLAMA_MODEL,
        adapter_loaded=False,
    )
    return model, None


def load_hf_model():
    output_dir = resolve_output_dir()
    adapter_config = output_dir / ADAPTER_CONFIG_NAME
    audit_event(
        "model_load_started",
        backend="hf",
        model_name=MODEL_NAME,
        output_dir=str(output_dir),
    )

    print("[inference] Carregando tokenizer Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    use_4bit = _should_use_hf_4bit()
    if use_4bit:
        print("[inference] Carregando modelo base Hugging Face (4-bit com offload)...")
        model_kwargs = {
            "quantization_config": build_bnb_config(),
            "device_map": "auto",
            "max_memory": _hf_max_memory(),
            "dtype": torch.float16,
        }
    else:
        if torch.cuda.is_available():
            print(
                "[inference] GPU detectada, mas com pouca VRAM para 4-bit "
                f"({_cuda_name()}, {_cuda_vram_gb():.1f} GB)."
            )
        else:
            print("[inference] PyTorch CUDA indisponivel nesta venv.")
        model_kwargs = {
            "device_map": {"": "cpu"},
            "dtype": torch.float32,
            "low_cpu_mem_usage": True,
        }

    try:
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    except ValueError as exc:
        if use_4bit and "Some modules are dispatched" in str(exc):
            print(
                "[inference] 4-bit nao coube na GPU mesmo com offload; "
                "tentando fallback em CPU."
            )
            base_model = _load_hf_base_model_cpu()
            use_4bit = False
        else:
            raise

    if has_lora_adapter(output_dir):
        print(f"[inference] Aplicando adaptadores LoRA de '{output_dir}'...")
        model = PeftModel.from_pretrained(base_model, str(output_dir))
        adapter_loaded = True
    else:
        print(
            "[inference] AVISO: adaptadores LoRA nao encontrados em "
            f"'{adapter_config}'. Rodando apenas com o modelo base."
        )
        print(
            "[inference] Para usar o fine-tuning, execute `python train.py` "
            "a partir da pasta fine-tuning-langchain."
        )
        audit_event(
            "lora_adapter_missing",
            model_name=MODEL_NAME,
            output_dir=str(output_dir),
            expected_file=str(adapter_config),
            fallback="base_model",
        )
        model = base_model
        adapter_loaded = False

    model.eval()

    audit_event(
        "model_load_finished",
        backend="hf",
        model_name=MODEL_NAME,
        output_dir=str(output_dir),
        adapter_loaded=adapter_loaded,
        quantized_4bit=use_4bit,
    )
    return model, tokenizer


def load_model():
    backend = MODEL_BACKEND
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"MODEL_BACKEND invalido: '{backend}'. Use auto, ollama ou hf."
        )

    if backend in {"auto", "ollama"}:
        try:
            return load_ollama_model()
        except OllamaUnavailable as exc:
            if backend == "ollama":
                raise RuntimeError(str(exc)) from exc
            print(f"[inference] Ollama nao sera usado: {exc}")
            audit_event("ollama_backend_skipped", reason=str(exc))

    return load_hf_model()


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
    if isinstance(model, OllamaModel):
        return model.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    if tokenizer is None:
        raise RuntimeError("Tokenizer Hugging Face ausente para gerar texto.")

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


def _source_for_context(context: str, model=None) -> str:
    if context.strip():
        return "dados_clinicos_fornecidos_pelo_usuario"
    return "sem_dados_clinicos_fornecidos_pelo_usuario"


def _detect_guardrail_findings(answer: str) -> str:
    findings = []
    has_directive = bool(UNSAFE_DIRECTIVE_RE.search(answer or ""))
    has_dosage = bool(UNSAFE_DOSAGE_RE.search(answer or ""))

    if has_directive and has_dosage:
        findings.append("Possivel prescricao direta com dose ou quantidade.")
    if DEFINITIVE_DIAGNOSIS_RE.search(answer or ""):
        findings.append("Possivel diagnostico definitivo sem validacao humana.")
    if _mentions_internal_system(answer or ""):
        findings.append("Resposta menciona tecnologia, modelo ou fonte interna.")

    return "\n".join(f"- {item}" for item in findings) if findings else "- Nenhum sinal deterministico relevante."


def _prompt(template: str, **kwargs) -> str:
    require_langchain()
    return PromptTemplate.from_template(template).format(**kwargs)


def _is_evaluation_approved(evaluation: str) -> bool:
    match = EVALUATION_STATUS_RE.search(evaluation or "")
    return bool(match and match.group(1).strip().upper() == "APROVADA")


def _evaluation_status(evaluation: str) -> str:
    match = EVALUATION_STATUS_RE.search(evaluation or "")
    if match:
        return match.group(1).strip().upper()
    return "REVISAR"


def _build_initial_answer(model, tokenizer, question: str, context: str) -> tuple[str, str]:
    prompt = _prompt(
        INITIAL_RESPONSE_PROMPT,
        safety_boundaries=SAFETY_BOUNDARIES,
        context_block=_context_block(context),
        question=question,
    )
    answer = _generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=256,
        temperature=0.2,
    )
    return prompt, answer


def _build_revision_answer(
    model,
    tokenizer,
    question: str,
    context: str,
    initial_answer: str,
    evaluation: str,
) -> tuple[str, str]:
    prompt = _prompt(
        REVISION_PROMPT,
        safety_boundaries=SAFETY_BOUNDARIES,
        context_block=_context_block(context),
        question=question,
        initial_answer=initial_answer,
        evaluation=evaluation,
    )
    answer = _generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=256,
        temperature=0.2,
    )
    return prompt, answer


def _build_evaluation(
    model,
    tokenizer,
    question: str,
    context: str,
    initial_answer: str,
) -> tuple[str, str]:
    guardrail_findings = _detect_guardrail_findings(initial_answer)
    prompt = _prompt(
        EVALUATION_PROMPT,
        safety_boundaries=SAFETY_BOUNDARIES,
        context_block=_context_block(context),
        question=question,
        initial_answer=initial_answer,
        guardrail_findings=guardrail_findings,
    )
    evaluation = _generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=180,
        temperature=0.0,
    )
    return prompt, evaluation


def _build_final_summary(model, tokenizer, question: str, context: str) -> tuple[str, str]:
    prompt = _prompt(
        FINAL_FORMAT_PROMPT,
        safety_boundaries=SAFETY_BOUNDARIES,
        context_block=_context_block(context),
        question=question,
    )
    final_answer = _generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=220,
        temperature=0.0,
    )
    final_answer = _enforce_final_contract(final_answer)
    return prompt, final_answer


def build_langchain_flow(model, tokenizer):
    require_langchain()

    def initial_step(payload: dict) -> dict:
        request_id = payload["request_id"]
        prompt = _prompt(
            INITIAL_RESPONSE_PROMPT,
            safety_boundaries=SAFETY_BOUNDARIES,
            context_block=_context_block(payload["context"]),
            question=payload["question"],
        )

        audit_event("initial_prompt_built", request_id, prompt=prompt)
        answer = _generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=256,
            temperature=0.2,
        )
        audit_event("initial_answer_generated", request_id, answer=answer)
        return {**payload, "initial_answer": answer}

    def evaluation_step(payload: dict) -> dict:
        request_id = payload["request_id"]
        guardrail_findings = _detect_guardrail_findings(payload["initial_answer"])
        prompt = _prompt(
            EVALUATION_PROMPT,
            safety_boundaries=SAFETY_BOUNDARIES,
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

    def final_format_step(payload: dict) -> dict:
        request_id = payload["request_id"]
        prompt = _prompt(
            FINAL_FORMAT_PROMPT,
            safety_boundaries=SAFETY_BOUNDARIES,
            context_block=_context_block(payload["context"]),
            question=payload["question"],
        )

        audit_event("final_prompt_built", request_id, prompt=prompt)
        final_answer = _generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=220,
            temperature=0.0,
        )
        final_answer = _enforce_final_contract(final_answer)
        audit_event("final_answer_generated", request_id, final_answer=final_answer)
        return {
            "evaluation": payload["evaluation"],
            "final_answer": final_answer,
        }

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


def _strip_final_label(line: str) -> str:
    cleaned = line.lstrip("-* ").strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
    cleaned = cleaned.replace("**", "").strip()
    folded = _fold_text(cleaned)

    for prefix in FINAL_LABEL_PREFIXES:
        if folded == prefix or folded == f"{prefix}:":
            return ""
        if folded.startswith(f"{prefix}:"):
            return cleaned.split(":", 1)[1].strip()

    if ":" in cleaned:
        label, content = cleaned.split(":", 1)
        if 0 < len(label.split()) <= 4 and content.strip():
            return content.strip()

    return cleaned


def _enforce_final_contract(answer: str) -> str:
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]

    if not lines:
        lines = [
            "Nao foi possivel gerar uma resposta confiavel com as informacoes disponiveis.",
        ]

    normalized = []
    for line in lines:
        cleaned = _strip_final_label(line)
        folded_cleaned = _fold_text(cleaned)
        if (
            not cleaned
            or folded_cleaned in FINAL_SKIP_LINES
            or _mentions_internal_system(cleaned)
        ):
            continue
        normalized.append("- " + cleaned)

    if not normalized:
        normalized = [
            "- Nao foi possivel gerar uma recomendacao confiavel com os dados informados.",
        ]

    joined = _fold_text("\n".join(normalized))
    has_professional_reference = re.search(
        r"\b(profissional|medico|medica|medicos|medicas|enfermeiro|enfermeira)\b",
        joined,
    )
    if not has_professional_reference:
        normalized.append(
            "- Procure um profissional de saude para avaliar o quadro e definir a conduta adequada."
        )
    if "urgenc" not in joined and "emergenc" not in joined:
        normalized.append(
            "- Procure atendimento urgente se houver falta de ar, dor intensa, confusao, "
            "desmaio, desidratacao, piora rapida ou sinais de gravidade."
        )

    return "\n".join(normalized)


def ask(
    model,
    tokenizer,
    question: str,
    context: str = "",
    session_log_path: Path | None = None,
) -> str:
    request_id = str(uuid4())
    source = _source_for_context(context, model)
    audit_event(
        "request_received",
        request_id,
        question=question,
        context=context,
        source=source,
        context_present=bool(context.strip()),
    )

    question = question.strip()
    context = context.strip()

    if session_log_path is not None:
        _append_session_log(session_log_path, "Pergunta", question)
        if context:
            _append_session_log(session_log_path, "Contexto", context)

    prompt, initial_answer = _build_initial_answer(model, tokenizer, question, context)
    audit_event(
        "initial_prompt_built",
        request_id,
        prompt=prompt,
    )
    audit_event(
        "initial_answer_generated",
        request_id,
        answer=initial_answer,
    )
    if session_log_path is not None:
        _append_session_log(session_log_path, "Prompt inicial", prompt)
        _append_session_log(session_log_path, "Resposta inicial", initial_answer)

    evaluation_prompt, evaluation = _build_evaluation(
        model,
        tokenizer,
        question,
        context,
        initial_answer,
    )
    audit_event(
        "evaluation_prompt_built",
        request_id,
        prompt=evaluation_prompt,
    )
    audit_event(
        "answer_evaluated",
        request_id,
        evaluation=evaluation,
    )
    if session_log_path is not None:
        _append_session_log(session_log_path, "Prompt de avaliacao", evaluation_prompt)
        _append_session_log(session_log_path, "Avaliacao", evaluation)

    attempt = 1
    while not _is_evaluation_approved(evaluation) and attempt < MAX_EVALUATION_ATTEMPTS:
        attempt += 1
        revision_prompt, revised_answer = _build_revision_answer(
            model,
            tokenizer,
            question,
            context,
            initial_answer,
            evaluation,
        )
        audit_event(
            "revision_prompt_built",
            request_id,
            prompt=revision_prompt,
        )
        audit_event(
            "revision_answer_generated",
            request_id,
            revised_answer=revised_answer,
            attempt=attempt,
        )
        if session_log_path is not None:
            _append_session_log(session_log_path, f"Prompt de revisao (tentativa {attempt})", revision_prompt)
            _append_session_log(session_log_path, f"Resposta revisada (tentativa {attempt})", revised_answer)

        initial_answer = revised_answer
        evaluation_prompt, evaluation = _build_evaluation(
            model,
            tokenizer,
            question,
            context,
            initial_answer,
        )
        audit_event(
            "re_evaluation_prompt_built",
            request_id,
            prompt=evaluation_prompt,
            attempt=attempt,
        )
        audit_event(
            "re_evaluated_answer",
            request_id,
            evaluation=evaluation,
            attempt=attempt,
        )
        if session_log_path is not None:
            _append_session_log(session_log_path, f"Prompt de reavaliacao (tentativa {attempt})", evaluation_prompt)
            _append_session_log(session_log_path, f"Reavaliacao (tentativa {attempt})", evaluation)

    final_prompt, final_answer = _build_final_summary(model, tokenizer, question, context)
    audit_event(
        "final_prompt_built",
        request_id,
        final_answer=final_answer,
    )
    if session_log_path is not None:
        _append_session_log(session_log_path, "Prompt final de resumo", final_prompt)
        _append_session_log(session_log_path, "Resposta final", final_answer)

    audit_event(
        "request_finished",
        request_id,
        evaluation=evaluation,
        final_answer=final_answer,
        accepted=_is_evaluation_approved(evaluation),
        attempts=attempt,
    )

    return f"{evaluation}\n\n===Resumo===\n\n{final_answer}"


def main():
    model, tokenizer = load_model()
    session_log_path = _create_session_log_file()

    print("\n=== Assistente Medico (digite 'sair' para encerrar) ===\n")
    while True:
        question = input("Pergunta: ").strip()
        if question.lower() in {"sair", "exit", "quit"}:
            break
        if not question:
            continue

        context = input("Contexto (opcional, Enter para pular): ").strip()
        resposta = ask(model, tokenizer, question, context, session_log_path=session_log_path)
        print(f"\nResposta:\n{resposta}\n")

    _close_session_log(session_log_path)


if __name__ == "__main__":
    main()
