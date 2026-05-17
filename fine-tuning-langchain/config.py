# ─── Modelo ──────────────────────────────────────────────────────────────────
#MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ─── Diretórios ───────────────────────────────────────────────────────────────
OUTPUT_DIR     = "./output"
PROCESSED_DATA = "./data/train_dataset.json"

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
