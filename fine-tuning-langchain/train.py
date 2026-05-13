"""
train.py
--------
Fine-tuning QLoRA do TinyLlama com dados PubMedQA pré-processados.

Uso:
    python prepare_data.py
    python train.py
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from config import (
    MODEL_NAME, OUTPUT_DIR, PROCESSED_DATA,
    BATCH_SIZE, GRAD_ACCUM, EPOCHS, LR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
)


MIN_TRAIN_VRAM_GB = 6.0


def load_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return Dataset.from_list(records)


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def validate_training_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Treino QLoRA requer PyTorch com CUDA ativo. "
            f"Ambiente atual: torch {torch.__version__}, "
            "torch.cuda.is_available()=False. "
            "Para esta maquina, use a inferencia com Ollama (`python inference.py`) "
            "ou treine em uma GPU NVIDIA com pelo menos 6 GB de VRAM."
        )

    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / 1024**3
    if total_vram_gb < MIN_TRAIN_VRAM_GB:
        raise RuntimeError(
            f"GPU detectada: {torch.cuda.get_device_name(0)} "
            f"com {total_vram_gb:.1f} GB de VRAM. "
            f"O treino QLoRA deste projeto precisa de pelo menos "
            f"{MIN_TRAIN_VRAM_GB:.0f} GB. "
            "Use Colab/Kaggle/VM com GPU maior para treinar, e Ollama "
            "para inferencia local nesta maquina."
        )


def train():
    validate_training_environment()
    from trl import SFTConfig, SFTTrainer

    print("[train] Carregando dataset...")
    dataset = load_dataset(PROCESSED_DATA)
    print(f"[train] {len(dataset)} exemplos carregados.")

    print("[train] Carregando tokenizer e modelo base (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=build_bnb_config(),
        device_map="auto",
        dtype=torch.float16,  # evita o aviso sobre torch_dtype
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="none",
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("[train] Iniciando fine-tuning...")
    trainer.train()

    print(f"[train] Salvando adaptadores LoRA em '{OUTPUT_DIR}'...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[train] Concluído.")


if __name__ == "__main__":
    train()
