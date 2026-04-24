"""
inference.py
------------
Carrega o modelo base + adaptadores LoRA e responde perguntas médicas.

Uso:
    python inference.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from config import MODEL_NAME, OUTPUT_DIR


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model():
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

    return model, tokenizer


def ask(model, tokenizer, question: str, context: str = "") -> str:
    context_block = f"Contexto:\n{context}\n\n" if context.strip() else ""

    prompt = (
        "<|system|>\n"
        "Você é um assistente médico especializado. Responda com base no contexto fornecido "
        "de forma clara, objetiva e baseada em evidências.\n"
        "</s>\n"
        "<|user|>\n"
        f"{context_block}"
        f"Pergunta: {question}\n"
        "</s>\n"
        "<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decodifica apenas os tokens gerados (descarta o prompt de entrada)
    generated = output_ids[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


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
