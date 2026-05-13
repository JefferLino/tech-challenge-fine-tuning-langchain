# ─── Modelo ──────────────────────────────────────────────────────────────────
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Backends de inferencia:
# - auto: usa Ollama se estiver rodando; caso contrario tenta Hugging Face local
# - ollama: exige Ollama local
# - hf: exige Transformers/Hugging Face local
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "auto").lower()

# Modelo Ollama recomendado para maquinas com pouca VRAM/RAM.
# Para trocar sem editar codigo: set OLLAMA_MODEL=llama3.2:1b
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_FALLBACK_MODELS = tuple(
    item.strip()
    for item in os.getenv(
        "OLLAMA_FALLBACK_MODELS",
        "qwen2.5:0.5b,qwen3:0.6b,tinyllama:latest,tinyllama",
    ).split(",")
    if item.strip()
)

# ─── Diretórios ───────────────────────────────────────────────────────────────
OUTPUT_DIR     = str(BASE_DIR / "output")
PROCESSED_DATA = str(BASE_DIR / "data" / "train_dataset.json")

# ─── Sequência ────────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 1024

# ─── Treinamento ──────────────────────────────────────────────────────────────
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS     = 3
LR         = 2e-4

# ─── LoRA ─────────────────────────────────────────────────────────────────────
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "v_proj", "k_proj", "o_proj"]
