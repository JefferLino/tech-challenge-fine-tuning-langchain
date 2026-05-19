# 🏥 Assistente Médico com Fine-Tuning QLoRA + LangChain

Este projeto implementa um **Assistente Médico Virtual** baseado em fine-tuning supervisionado (**QLoRA**) do modelo **Qwen2.5-0.5B-Instruct** sobre dados biomédicos reais do **PubMedQA**, com fluxo de avaliação de segurança, integração com prontuários e rastreabilidade completa. Roda 100% local em GPU NVIDIA, sem APIs pagas.

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
- Avaliação automática de segurança
- Revisão iterativa (até 3 tentativas)
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

## test.py
Arquivo de testes:
- Testa carregamento do modelo
- Testa geração e avaliação de respostas
- Testa integração com prontuários

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
peft>=0.7.0
trl>=0.7.4
bitsandbytes>=0.41.3
accelerate>=0.25.0
langchain>=1.0.0
langchain-huggingface>=1.0.0
datasets>=2.16.0
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

**3. Executar o assistente:**
```bash
python inference.py
```

---

# 📊 Funcionalidades

- Fine-tuning eficiente com QLoRA (6 GB de VRAM)
- Respostas baseadas em literatura biomédica real (PubMedQA)
- Avaliação automática de segurança com até 3 revisões
- Integração com prontuário médico do paciente via CPF
- Fluxo de estados com LangGraph
- Logs de auditoria e explicabilidade por interação
- Rastreabilidade completa via trace_id

---

# 🧠 Conceitos Principais

**QLoRA** combina duas técnicas para viabilizar fine-tuning em hardware de consumidor:

- **Quantização 4-bit (BitsAndBytes / NF4):** reduz o modelo de ~2 GB para ~700 MB na GPU, com o modelo base completamente congelado.
- **LoRA (Low-Rank Adaptation):** injeta matrizes de baixo rank (`r=16`) nas camadas de atenção. Apenas ~0,5% dos parâmetros são treinados.

O resultado: fine-tuning que cabe em 6 GB de VRAM e treina em horas, não dias.

---

# 📌 Resumo

O projeto entrega um assistente médico inteligente treinado localmente, combinando fine-tuning eficiente com QLoRA, orquestração de fluxo com LangGraph e salvaguardas de segurança automáticas para garantir respostas clínicas responsáveis e rastreáveis.
