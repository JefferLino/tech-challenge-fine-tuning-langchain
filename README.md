# Fine-tuning LLaMA — Assistente Médico com PubMedQA

Fine-tuning supervisionado (QLoRA) do **TinyLlama-1.1B** usando dados do PubMedQA para criar um assistente médico virtual. Roda 100% local em GPU NVIDIA, sem APIs pagas.

---

## Estrutura do projeto

```
fine-tuning-langchain/
├── data/
│   ├── ori_pqal.json          # perguntas + contextos + respostas longas
│   └── test_ground_truth.json # respostas yes/no/maybe por ID
├── output/                    # adaptadores LoRA (gerado após treino)
├── config.py                  # hiperparâmetros centralizados
├── prepare_data.py            # pré-processamento e anonimização
├── train.py                   # fine-tuning QLoRA
├── inference.py               # chatbot interativo
└── requirements.txt
```

---

## Requisitos de hardware

| Componente | Mínimo recomendado |
|------------|-------------------|
| GPU NVIDIA | 6 GB VRAM (RTX 3060 / RTX 2070 ou superior) |
| RAM        | 16 GB |
| Armazenamento | ~10 GB livres |

---

## Instalação

```bash
pip install -r requirements.txt
```

> **CUDA**: instale o PyTorch com suporte a CUDA compatível com seu driver.  
> Consulte https://pytorch.org/get-started/locally/

---

## Passo a passo

### 1. Preparar os dados

Coloque `ori_pqal.json` e `test_ground_truth.json` dentro da pasta `data/`.

```bash
python prepare_data.py
```

Gera `data/train_dataset.json` com prompts no formato TinyLlama-Chat.  
O script também aplica anonimização básica (datas, nomes, IDs numéricos).

---

### 2. Treinar o modelo

```bash
python train.py
```

O que acontece internamente:
- Carrega TinyLlama-1.1B em **4-bit** (QLoRA via BitsAndBytes)
- Injeta adaptadores **LoRA** apenas nas camadas de atenção (`q_proj`, `v_proj`, `k_proj`, `o_proj`)
- Treina por 3 épocas com batch efetivo de 8 (`BATCH_SIZE=2` × `GRAD_ACCUM=4`)
- Salva os adaptadores em `./output/`

**Somente ~0,5% dos parâmetros são treinados** — o restante permanece congelado.

---

### 3. Usar o assistente

```bash
python inference.py
```

Exemplo de sessão:

```
=== Assistente Médico (digite 'sair' para encerrar) ===

Pergunta: Does metformin reduce cardiovascular risk in type 2 diabetes?
Contexto (opcional, Enter para pular): Several RCTs showed...

Resposta:
YES

Metformin has been shown to reduce cardiovascular events in overweight patients
with type 2 diabetes according to the UKPDS trial...
```

---

## Configuração (`config.py`)

| Parâmetro | Valor padrão | Descrição |
|-----------|-------------|-----------|
| `MODEL_NAME` | TinyLlama-1.1B-Chat-v1.0 | Modelo base (arquitetura LLaMA) |
| `MAX_SEQ_LENGTH` | 1024 | Comprimento máximo da sequência |
| `BATCH_SIZE` | 2 | Exemplos por passo de GPU |
| `GRAD_ACCUM` | 4 | Acumulação de gradiente |
| `EPOCHS` | 3 | Épocas de treino |
| `LR` | 2e-4 | Taxa de aprendizado |
| `LORA_R` | 16 | Rank das matrizes LoRA |
| `LORA_ALPHA` | 32 | Escala LoRA |

---

## Conceitos principais

**QLoRA** combina duas técnicas:

1. **Quantização 4-bit (BitsAndBytes)**: reduz o modelo de ~2 GB (fp16) para ~700 MB na GPU, tornando possível rodar em GPUs de consumidor.

2. **LoRA (Low-Rank Adaptation)**: ao invés de atualizar todos os pesos, injeta matrizes de baixo rank (`r=16`) nas camadas de atenção. Apenas ~8 milhões de parâmetros são treinados de um total de ~1,1 bilhão.

O resultado: fine-tuning que cabe em 6 GB de VRAM e treina em horas, não dias.

---

## Modelo base

`TinyLlama/TinyLlama-1.1B-Chat-v1.0` — arquitetura LLaMA, completamente gratuito, sem necessidade de login no Hugging Face.

Para usar `meta-llama/Llama-3.2-1B-Instruct` (opcional), aceite a licença em huggingface.co e faça `huggingface-cli login`.
