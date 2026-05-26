# 🏥 Assistente Médico com Fine-Tuning QLoRA + LangChain

Este projeto implementa um **Assistente Médico Virtual** baseado em fine-tuning supervisionado (**QLoRA**) do modelo **Qwen/Qwen2.5-3B-Instruct** sobre dados biomédicos reais do **PubMedQA**, com fluxo de avaliação de segurança (via Ollama), RAG opcional, integração com prontuários e rastreabilidade completa. Roda 100% local em GPU NVIDIA, sem APIs pagas.

---

# 📂 Estrutura do Projeto

## config.py
Hiperparâmetros centralizados:
- Nome do modelo base
- Parâmetros de treino (épocas, batch, learning rate)
- Configuração dos adaptadores LoRA
- Caminhos de entrada e saída

## prepare_data.py
Pipeline de preparação dos dados:
- Leitura do dataset PubMedQA
- Anonimização de datas, nomes e IDs
- Truncagem de contextos longos
- Formatação dos prompts de instrução
- Geração do dataset de treino

## train.py
Fine-tuning com QLoRA:
- Carregamento do modelo base em 4-bit (BitsAndBytes / NF4)
- Injeção de adaptadores LoRA nas camadas de atenção
- Treinamento supervisionado com SFTTrainer (TRL)
- Salvamento dos adaptadores em `./output/`

## inference.py
Assistente médico com fluxo LangGraph:
- Carregamento do modelo + adaptadores LoRA
- Geração de resposta inicial
- Avaliação automática de segurança via Ollama (`qwen3:4b`)
- Revisão iterativa (até 3 tentativas) se rejeitado
- RAG opcional por similaridade semântica (índice em `vector/`)
- Integração com prontuário do paciente via CPF

## prontuario.py
Integração com prontuários médicos:
- Leitura de dados simulados em Excel
- Busca por CPF
- Formatação do histórico clínico como contexto para o modelo

## logs.py
Sistema de auditoria e explicabilidade:
- Log de auditoria em JSON com trace_id
- Log de explicabilidade com prompts, respostas, tokens e hashes
- Rastreabilidade completa de cada interação

## process_rag.py
Indexação vetorial para RAG:
- Leitura de arquivos `.json`, `.txt`, `.pdf` em `data/contexts/`
- Geração de embeddings com `sentence-transformers/all-MiniLM-L6-v2`
- Salvamento do índice em `vector/` (chunks.json, embeddings.pt, config.json)

## test_model.py
Teste rápido de verificação:
- Carrega modelo + adaptadores LoRA
- Testa geração e RAG básico (TF-IDF)
- Valida integração ponta a ponta

---

# ⚙️ Dependências

Instale com:

```bash
pip install -r requirements.txt
```

Principais bibliotecas:

```
torch>=2.1.0
transformers>=4.36.0
langchain>=1.0.0
langchain-huggingface>=1.0.0
langchain-ollama>=1.1.0
peft>=0.7.0
trl>=0.7.4
bitsandbytes>=0.41.3
accelerate>=0.25.0
datasets>=2.16.0
sentence-transformers>=2.6.0
pypdf>=4.0.0
grandalf>=0.8
pandas
openpyxl
```

> **CUDA**: instale o PyTorch com suporte a CUDA compatível com seu driver em https://pytorch.org/get-started/locally/

---

# 🚀 Como Executar

**1. Preparar os dados:**
```bash
python prepare_data.py
```

**2. Treinar o modelo:**
```bash
python train.py
```

**3. (Opcional) Indexar documentos para RAG:**
```bash
python process_rag.py
```

**4. (Opcional) Testar carregamento do modelo:**
```bash
python test_model.py
```

**5. Executar o assistente:**
```bash
python inference.py
```

---

# 📊 Funcionalidades

- Fine-tuning eficiente com QLoRA (6 GB de VRAM)
- Respostas baseadas em literatura biomédica real (PubMedQA)
- RAG opcional por similaridade semântica (índice vetorial local)
- Avaliação automática de segurança via Ollama com até 3 revisões
- Integração com prontuário médico do paciente via CPF
- Fluxo de estados com LangGraph
- Logs de auditoria e explicabilidade por interação
- Rastreabilidade completa via trace_id

---

# 🧠 Conceitos Principais

**QLoRA** combina duas técnicas para viabilizar fine-tuning em hardware de consumidor:

- **Quantização 4-bit (BitsAndBytes / NF4):** reduz o modelo de ~6 GB para ~1.5 GB na GPU, com o modelo base completamente congelado.
- **LoRA (Low-Rank Adaptation):** injeta matrizes de baixo rank (`r=16`) nas camadas de atenção (QKV + gate/up/down). Menos de ~0,3% dos parâmetros são treinados.

O resultado: fine-tuning que cabe em 6 GB de VRAM e treina em horas, não dias.

---

# 📌 Resumo

O projeto entrega um assistente médico inteligente treinado localmente, combinando fine-tuning eficiente com QLoRA, orquestração de fluxo com LangGraph e salvaguardas de segurança automáticas para garantir respostas clínicas responsáveis e rastreáveis.
