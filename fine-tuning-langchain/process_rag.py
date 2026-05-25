"""
process_rag.py
--------------
Processa arquivos de contexto (.json, .txt, .pdf) e gera um indice vetorial
local para uso posterior no inference.py.

Entrada padrao:
    data/contexts/

Saida padrao:
    vector/

Uso:
    python process_rag.py
"""

import json
import re
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


BASE_DIR = Path(__file__).resolve().parent
CONTEXT_DIR = BASE_DIR / "data" / "contexts"
TRAIN_DATASET_PATH = BASE_DIR / "data" / "train_dataset.json"
VECTOR_DIR = BASE_DIR / "vector"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
MIN_CHARS = 40


def extract_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        return ""
    return normalize_text(text[start:end])


def load_train_questions() -> list[str]:
    if not TRAIN_DATASET_PATH.exists():
        return []

    with TRAIN_DATASET_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    questions = []
    for record in records:
        text = str(record.get("text", ""))
        question = extract_between(text, "Question:", "\n</s>")
        if not question:
            question = extract_between(text, "Pergunta:", "\n</s>")
        questions.append(question)

    return questions


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def extract_strings_from_json(value: Any) -> list[str]:
    if isinstance(value, str):
        text = normalize_text(value)
        return [text] if len(text) >= MIN_CHARS else []

    if isinstance(value, list):
        texts = []
        for item in value:
            texts.extend(extract_strings_from_json(item))
        return texts

    if isinstance(value, dict):
        preferred_keys = ("context", "Contexto", "content", "text", "answer", "Answer")
        texts = []

        for key in preferred_keys:
            if key in value:
                texts.extend(extract_strings_from_json(value[key]))

        if texts:
            return texts

        for item in value.values():
            texts.extend(extract_strings_from_json(item))
        return texts

    return []


def read_json(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if path.name.lower() == "pubmedqa.json" and isinstance(payload, list):
        questions = load_train_questions()
        enriched_texts = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            context = normalize_text(str(item.get("Contexto") or item.get("context") or ""))
            question = questions[index] if index < len(questions) else ""
            if context and question:
                enriched_texts.append(f"Question: {question}\nContext: {context}")
            elif context:
                enriched_texts.append(context)

        if enriched_texts:
            return enriched_texts

    return extract_strings_from_json(payload)


def read_txt(path: Path) -> list[str]:
    return [path.read_text(encoding="utf-8", errors="ignore")]


def read_pdf(path: Path) -> list[str]:
    if PdfReader is None:
        print(f"[process_rag] PDF ignorado, instale pypdf para ler: {path}")
        return []

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return ["\n".join(pages)]


def read_document(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return read_json(path)
    if suffix == ".txt":
        return read_txt(path)
    if suffix == ".pdf":
        return read_pdf(path)
    return []


def collect_chunks(context_dir: Path) -> list[dict[str, object]]:
    records = []
    supported_files = sorted(
        path
        for path in context_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".txt", ".pdf"}
    )

    for path in supported_files:
        try:
            texts = read_document(path)
        except Exception as error:
            print(f"[process_rag] Arquivo ignorado por erro: {path} ({error})")
            continue

        for document_index, text in enumerate(texts):
            chunks = [normalize_text(text)] if path.suffix.lower() == ".json" else chunk_text(text)

            for chunk_index, chunk in enumerate(chunks):
                if len(chunk) < MIN_CHARS:
                    continue
                records.append(
                    {
                        "id": len(records),
                        "source": str(path.relative_to(BASE_DIR)),
                        "document_index": document_index,
                        "chunk_index": chunk_index,
                        "text": chunk,
                    }
                )

    return records


def save_index(records: list[dict[str, object]], embeddings: torch.Tensor) -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    with (VECTOR_DIR / "chunks.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    torch.save(embeddings.cpu(), VECTOR_DIR / "embeddings.pt")

    config = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_chunks": len(records),
    }
    with (VECTOR_DIR / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def main() -> None:
    if not CONTEXT_DIR.exists():
        raise FileNotFoundError(f"Pasta de contextos nao encontrada: {CONTEXT_DIR}")

    print(f"[process_rag] Lendo arquivos em: {CONTEXT_DIR}")
    records = collect_chunks(CONTEXT_DIR)
    if not records:
        raise RuntimeError("Nenhum chunk gerado. Verifique se ha arquivos .json, .txt ou .pdf com texto.")

    print(f"[process_rag] Chunks gerados: {len(records)}")
    print(f"[process_rag] Carregando modelo de embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [str(record["text"]) for record in records]
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    save_index(records, embeddings)
    print(f"[process_rag] Indice salvo em: {VECTOR_DIR}")
    print("[process_rag] Arquivos gerados: chunks.json, embeddings.pt, config.json")


if __name__ == "__main__":
    main()
