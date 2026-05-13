"""
logs.py
-------
Centraliza auditoria e explainability das chamadas da LLM.

Arquivos gerados em runtime:
    logs/audit.log
    logs/explainability.jsonl
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
AUDIT_LOG_FILE = LOG_DIR / "audit.log"
EXPLAINABILITY_LOG_FILE = LOG_DIR / "explainability.jsonl"
MAX_TEXT_CHARS = 4000
MAX_CONTEXT_PREVIEW_CHARS = 700

_LOGGER: Optional[logging.Logger] = None


def setup_logging() -> logging.Logger:
    """Configura o logger de auditoria e garante a existencia dos arquivos."""
    global _LOGGER

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAINABILITY_LOG_FILE.touch(exist_ok=True)

    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("llm_audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    audit_path = str(AUDIT_LOG_FILE)
    has_audit_handler = any(getattr(handler, "baseFilename", None) == audit_path for handler in logger.handlers)
    if not has_audit_handler:
        handler = logging.FileHandler(AUDIT_LOG_FILE, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    _LOGGER = logger
    return logger


def new_trace_id() -> str:
    return str(uuid.uuid4())


def start_timer() -> float:
    return time.perf_counter()


def elapsed_seconds(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 6)


def log_audit_event(
    event: str,
    trace_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    logger = setup_logging()
    logger.info(
        _to_json(
            {
                "timestamp": _utc_now(),
                "trace_id": trace_id,
                "event": event,
                "metadata": metadata or {},
            }
        )
    )


def log_service_call(
    service_name: str,
    question: str,
    prompt: str,
    response: str,
    context: str = "",
    trace_id: Optional[str] = None,
    generation_params: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    duration_seconds: Optional[float] = None,
    source: Optional[str] = None,
) -> None:
    """Registra uma chamada completa da LLM para rastreamento e auditoria."""
    source_info = build_source_info(context=context, source=source)
    log_audit_event(
        event="llm_service_call",
        trace_id=trace_id,
        metadata={
            "service_name": service_name,
            "question": _truncate(question),
            "prompt": _truncate(prompt),
            "response": _truncate(response),
            "generation_params": generation_params or {},
            "duration_seconds": duration_seconds,
            "source": source_info,
            **(metadata or {}),
        },
    )


def log_service_error(
    service_name: str,
    question: str,
    error: Exception,
    context: str = "",
    trace_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> None:
    """Registra falhas sem engolir a excecao original."""
    log_audit_event(
        event="llm_service_error",
        trace_id=trace_id,
        metadata={
            "service_name": service_name,
            "question": _truncate(question),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "source": build_source_info(context=context, source=source),
            **(metadata or {}),
        },
    )


def log_explainability(
    service_name: str,
    question: str,
    answer: str,
    context: str = "",
    trace_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> None:
    """Grava a fonte/contexto usados para explicar uma resposta da LLM."""
    setup_logging()
    payload = {
        "timestamp": _utc_now(),
        "trace_id": trace_id,
        "service_name": service_name,
        "question": _truncate(question),
        "answer": _truncate(answer),
        "source": build_source_info(context=context, source=source),
        "metadata": metadata or {},
    }

    with EXPLAINABILITY_LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(_to_json(payload) + "\n")


def build_source_info(context: str = "", source: Optional[str] = None) -> Dict[str, Any]:
    context = context or ""
    source = (source or "").strip()

    if source:
        source_label = source
    elif context.strip():
        source_label = "Contexto fornecido na chamada de inferencia"
    else:
        source_label = "Sem fonte externa/contexto informado"

    return {
        "label": source_label,
        "has_context": bool(context.strip()),
        "context_sha256": _hash_text(context),
        "context_preview": _truncate(context, MAX_CONTEXT_PREVIEW_CHARS),
    }


def add_explainability_to_answer(answer: str, context: str = "", source: Optional[str] = None) -> str:
    """Garante que a resposta entregue informe a fonte usada."""
    if _has_explainability_section(answer):
        return answer

    source_info = build_source_info(context=context, source=source)
    if source_info["has_context"]:
        evidence = source_info["context_preview"] or "contexto informado"
    else:
        evidence = "nenhum contexto externo foi informado para esta pergunta"

    return (
        f"{answer.strip()}\n\n"
        "Explainability:\n"
        f"- Fonte rastreada pela aplicacao: {source_info['label']}.\n"
        f"- Evidencia considerada: {evidence}"
    ).strip()


def _has_explainability_section(answer: str) -> bool:
    normalized = (answer or "").lower()
    return "fonte rastreada pela aplicacao:" in normalized


def _truncate(text: Any, max_chars: int = MAX_TEXT_CHARS) -> str:
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated {len(text) - max_chars} chars]"


def _hash_text(text: str) -> str:
    text = text or ""
    if not text.strip():
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True)
