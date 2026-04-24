"""
prepare_data.py
---------------
Lê ori_pqal.json + test_ground_truth.json, mescla pelo ID da questão,
constrói prompts no formato TinyLlama-Chat e salva em data/train_dataset.json.

Uso:
    python prepare_data.py
"""

import json
import os
import re


ORI_PATH   = "./data/ori_pqal.json"
TRUTH_PATH = "./data/test_ground_truth.json"
OUT_PATH   = "./data/train_dataset.json"

# Número máximo de caracteres do contexto (evita ultrapassar MAX_SEQ_LENGTH)
MAX_CONTEXT_CHARS = 1800


def anonymize(text: str) -> str:
    """Remove padrões simples que possam identificar pacientes."""
    # Datas no formato dd/mm/aaaa ou mm/dd/aaaa
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATA]", text)
    # Nomes próprios antecedidos de "Dr.", "Dr", "Patient", "Mr.", "Mrs."
    text = re.sub(r"\b(Dr\.?|Patient|Mr\.?|Mrs\.?|Ms\.?)\s+[A-Z][a-z]+", r"\1 [NOME]", text)
    # Números que parecem IDs / telefones (sequências de 7+ dígitos)
    text = re.sub(r"\b\d{7,}\b", "[ID]", text)
    return text


def build_prompt(question: str, context: str, answer: str, long_answer: str) -> str:
    context = anonymize(context[:MAX_CONTEXT_CHARS])
    question = anonymize(question)
    long_answer = anonymize(long_answer)
    answer = answer.strip().upper() if answer else "INDEFINIDO"

    return (
        "<|system|>\n"
        "Você é um assistente médico especializado. Responda com base no contexto fornecido "
        "de forma clara, objetiva e baseada em evidências.\n"
        "</s>\n"
        "<|user|>\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta: {question}\n"
        "</s>\n"
        "<|assistant|>\n"
        f"Resposta: {answer}\n\n"
        f"Explicação: {long_answer}\n"
        "</s>"
    )


def prepare(ori_path: str = ORI_PATH, truth_path: str = TRUTH_PATH, out_path: str = OUT_PATH):
    with open(ori_path, "r", encoding="utf-8") as f:
        ori_data: dict = json.load(f)

    with open(truth_path, "r", encoding="utf-8") as f:
        truth_data: dict = json.load(f)

    records = []
    skipped = 0

    for qid, entry in ori_data.items():
        question    = entry.get("QUESTION", "").strip()
        long_answer = entry.get("LONG_ANSWER", "").strip()
        contexts    = entry.get("CONTEXTS", [])

        if not question or not long_answer:
            skipped += 1
            continue

        # Resposta: prefere test_ground_truth, cai para final_decision
        answer = truth_data.get(qid) or entry.get("final_decision", "")

        # Junta os parágrafos de contexto
        context = " ".join(contexts) if isinstance(contexts, list) else str(contexts)

        prompt = build_prompt(question, context, answer, long_answer)
        records.append({"text": prompt})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[prepare_data] Total gerado : {len(records)}")
    print(f"[prepare_data] Ignorados    : {skipped}")
    print(f"[prepare_data] Salvo em     : {out_path}")


if __name__ == "__main__":
    prepare()
