import uuid
from datetime import datetime
from pathlib import Path

SESSION_LOG_DIR = Path(__file__).resolve().parent / "logs" / "sessions"


def create_session_log_file() -> Path:
    SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    session_file = SESSION_LOG_DIR / (
        f"session_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.txt"
    )
    with session_file.open("w", encoding="utf-8") as handle:
        handle.write(
            "Sessao iniciada: "
            f"{datetime.now():%Y-%m-%d %H:%M:%S}\n"
        )
        handle.write("=== Registro de sessao ===\n\n")
    return session_file


def append_session_log(session_log_path: Path, label: str, content: str) -> None:
    with session_log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"--- {datetime.now():%Y-%m-%d %H:%M:%S} - {label} ---\n"
        )
        handle.write(content.rstrip() + "\n\n")


def close_session_log(session_log_path: Path) -> None:
    with session_log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "Sessao finalizada: "
            f"{datetime.now():%Y-%m-%d %H:%M:%S}\n"
        )
