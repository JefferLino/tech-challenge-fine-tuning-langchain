import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict

import torch
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline as transformers_pipeline

from config import MODEL_NAME, OUTPUT_DIR


LANGCHAIN_PROMPT = PromptTemplate.from_template("{prompt}")
BASE_DIR = Path(__file__).resolve().parent
CONTEXT_FILE = BASE_DIR / "data" / "context.json"
ADAPTER_DIR = (BASE_DIR / OUTPUT_DIR).resolve() if not Path(OUTPUT_DIR).is_absolute() else Path(OUTPUT_DIR)
RAG_TOP_K = 1


@lru_cache(maxsize=1)
def load_context_documents() -> tuple[str, ...]:
    with CONTEXT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    documents = []
    for record in records:
        if isinstance(record, dict):
            context = str(record.get("Contexto") or record.get("context") or "").strip()
        else:
            context = str(record).strip()

        if context:
            documents.append(context)

    return tuple(documents)


def tokenize_for_retrieval(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", (text or "").lower())


def retrieve_context(question: str, top_k: int = RAG_TOP_K) -> str:
    documents = load_context_documents()
    query_terms = tokenize_for_retrieval(question)
    if not documents or not query_terms:
        return ""

    query_term_set = set(query_terms)
    document_term_sets = [set(tokenize_for_retrieval(document)) for document in documents]
    document_frequency = {
        term: sum(1 for document_terms in document_term_sets if term in document_terms)
        for term in query_term_set
    }

    scored_documents = []
    total_documents = len(documents)
    for document, document_terms in zip(documents, document_term_sets):
        if not document_terms:
            continue

        score = 0.0
        for term in query_terms:
            if term in document_terms:
                score += math.log((total_documents + 1) / (document_frequency.get(term, 0) + 1)) + 1

        if score > 0:
            scored_documents.append((score, document))

    if not scored_documents:
        return ""

    scored_documents.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(document for _, document in scored_documents[:top_k])


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model():
    print("[test_model] Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("[test_model] Carregando modelo base (4-bit)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=build_bnb_config(),
        device_map="auto",
    )

    print(f"[test_model] Aplicando adaptadores LoRA de '{ADAPTER_DIR}'...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    return model, tokenizer


def generate_with_langchain(model, tokenizer, prompt: str, generation_params: Dict[str, object]) -> str:
    text_generation_pipeline = transformers_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    chain = LANGCHAIN_PROMPT | llm.bind(
        pipeline_kwargs={**generation_params, "return_full_text": False},
    )


    print("----------------")
    return chain.invoke({"prompt": prompt}).strip()


def build_prompt(question: str, context: str) -> str:
    return (
        "<|system|>\n"
        "Classify the question using only the context. "
        "Do not use outside knowledge. "
        "The evidence must be copied or directly paraphrased from the context. "
        "If the context does not answer the question, use MAYBE. "
        "Return exactly this format:\n"
        "Answer: YES/NO/MAYBE\n"
        "Evidence: one short sentence from the context.\n"
        "</s>\n"
        "<|user|>\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "</s>\n"
        "<|assistant|>\n"
        "Answer:"
    )


def main():
    model, tokenizer = load_model()

    question = "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"
    context = retrieve_context(question)
    context= "You are a medical evidence classification assistant. Answer only from the provided context. Classify the main question as YES, NO, or MAYBE. Base the label on the primary comparison or main outcome asked in the question. Do not treat subgroup findings as the answer to the main question unless the context explicitly says so. If a comparison is reported as not statistically significant, do not describe one group as better, worse, or more harmful. The explanation must directly justify the label and must not add unrelated background findings. Return exactly this format:\nAnswer: YES, NO, or MAYBE\nExplanation: one short sentence that cites the key evidence for the label.\n</s>\n<|user|>\nContext:\nProgrammed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longitudinal and transverse veins enclosing areoles. PCD occurs in the cells at the center of these areoles and progresses outwards, stopping approximately five cells from the vasculature. The role of mitochondria during PCD has been recognized in animals; however, it has been less studied during PCD in plants. The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. A single areole within a window stage leaf (PCD is occurring) was divided into three areas based on the progression of PCD; cells that will not undergo PCD (NPCD), cells in early stages of PCD (EPCD), and cells in late stages of PCD (LPCD). Window stage leaves were stained with the mitochondrial dye MitoTracker Red CMXRos and examined. Mitochondrial dynamics were delineated into four categories (M1-M4) based on characteristics including distribution, motility, and membrane potential (ΔΨm). A TUNEL assay showed fragmented nDNA in a gradient over these mitochondrial stages. Chloroplasts and transvacuolar strands were also examined using live cell imaging. The possible importance of mitochondrial permeability transition pore (PTP) formation during PCD was indirectly examined via in vivo cyclosporine A (CsA) treatment. This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells.\n\nQuestion: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"
    prompt = build_prompt(question, context)

    

    print("Prompt ---------------------------------- ")
    print(prompt)
    print("---------------------------------- Prompt")

    generation_params = {
        "max_new_tokens": 70,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    print(f"[rag] Fonte: {CONTEXT_FILE}")
    print(f"[rag] Contexto recuperado: {context[:300]}...\n")
    result = (generate_with_langchain(model, tokenizer, prompt, generation_params))

    print("---------------------")
    print(result)


if __name__ == "__main__":
    main()
