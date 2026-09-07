# ================================================================
#  IA UFOP — ETAPA 3: Fine-tuning com QLoRA
# ================================================================
# Esta etapa treina o modelo para ter melhor qualidade de resposta.
# É OPCIONAL: o RAG da Etapa 1/2 já funciona sem treino.
#
# ⚠️ REQUISITOS:
#   - Execute DEPOIS da Etapa 2 (precisa do arquivo chunks.jsonl)
#   - No Colab: Menu "Ambiente de execução" → "Alterar tipo de
#     ambiente de execução" → GPU: T4
#   - Tempo estimado: 3-5 horas (pode desconectar; tem checkpoint!)
#
# Como usar no Google Colab:
#   Copie cada bloco "# %% CÉLULA X" como uma célula separada
#
# NOTA HISTÓRICA: nesta execução, o modelo efetivamente ajustado foi o
# Qwen2.5-1.5B-Instruct (MODEL_ID na CÉLULA 4). O adaptador para o
# Qwen2.5-7B-Instruct ainda não foi treinado — ver README, seção
# "Limitações e lições aprendidas".
# ================================================================


# %%  CÉLULA 1 — Verificar GPU e instalar dependências de treino

import subprocess, sys

# Verificar GPU disponível
import os
gpu_info = os.popen('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader').read()
if gpu_info:
    print(f"✅ GPU encontrada: {gpu_info.strip()}")
else:
    print("❌ ERRO: Nenhuma GPU detectada!")
    print("   Vá em: Ambiente de execução → Alterar tipo → GPU: T4")
    raise SystemExit("GPU necessária para treino")

packages = [
    "transformers>=4.45.0",
    "peft>=0.12.0",
    "trl>=0.11.0",
    "bitsandbytes>=0.43.0",
    "accelerate>=0.34.0",
    "datasets>=2.21.0",
    "sentence-transformers",
    "chromadb",
]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✅ Dependências de treino instaladas!")


# %%  CÉLULA 2 — Montar Drive e configurar caminhos

from google.colab import drive
drive.mount('/content/drive')

import os

PASTA_SAIDA     = "/content/drive/MyDrive/ufop_ia_dados"
ARQUIVO_CHUNKS  = os.path.join(PASTA_SAIDA, "chunks.jsonl")
PASTA_ADAPTER   = os.path.join(PASTA_SAIDA, "modelo_ufop_adapter_v2")
PASTA_CHECKPT   = "/content/ufop_checkpoints"  # Temporário no Colab

os.makedirs(PASTA_CHECKPT, exist_ok=True)

if not os.path.exists(ARQUIVO_CHUNKS):
    print(f"❌ ERRO: Arquivo de chunks não encontrado em '{ARQUIVO_CHUNKS}'")
    print("   Execute a Etapa 2 primeiro (rag/construir_base_conhecimento.py)!")
    raise SystemExit("Execute a Etapa 2 primeiro")

# Contar chunks disponíveis
with open(ARQUIVO_CHUNKS, 'r', encoding='utf-8') as f:
    n_chunks = sum(1 for _ in f)

print(f"✅ Drive montado. Chunks disponíveis para treino: {n_chunks}")


# %%  CÉLULA 3 — Preparar dataset de treinamento

import json
import random
import re
from datasets import Dataset
import os

PASTA_PDFS     = "/content/drive/MyDrive/IC_UFOP_Arquivos"
PASTA_SAIDA    = "/content/drive/MyDrive/ufop_ia_dados"
ARQUIVO_CHUNKS = os.path.join(PASTA_SAIDA, "chunks.jsonl")

MAX_EXEMPLOS   = 8000
SEED_ALEATORIO = 42

SISTEMA_PROMPT = (
    "Você é um assistente especializado em documentos da UFOP, "
    "incluindo resoluções, editais do PIBIC, regulamentos e normas. "
    "Responda com base apenas nos documentos fornecidos, "
    "de forma clara, objetiva e em português."
)

ARQUIVO_JSONL_ORIGINAL = os.path.join(PASTA_PDFS, "dataset_ufop.jsonl")


def limpar_texto_jsonl(texto: str) -> str:
    """Remove ruído do campo 'input' do JSONL (URLs, timestamps, rodapés)."""
    texto = re.sub(r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}', '', texto)
    texto = re.sub(r'https?://\S+', '', texto)
    texto = re.sub(r'^\s*\d+/\d+\s*$', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'SEI/UFOP\s*-\s*\d+\s*-.*', '', texto)
    texto = re.sub(r'R\.\s*Diogo de Vasconcelos.*', '', texto, flags=re.DOTALL)
    texto = re.sub(r'Documento assinado eletronicamente.*?\d{4}\.', '', texto, flags=re.DOTALL)
    texto = re.sub(r'[ \t]{2,}', ' ', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto.strip()


def gerar_resposta_do_documento(texto_doc: str, fonte: str) -> str:
    """
    Gera uma resposta de treinamento a partir do texto do documento.
    Estratégia: extração + resumo estruturado do conteúdo real.
    """
    # Extrair informações-chave com regex
    numero_res   = re.search(r'RESOLUÇÃO[^\n]*N[ºo°]?\s*[\d.]+', texto_doc, re.IGNORECASE)
    data_texto   = re.search(r'(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})', texto_doc, re.IGNORECASE)
    artigos      = re.findall(r'Art\.?\s*\d+[º°]?\.?\s*[^\n]{20,150}', texto_doc)

    partes = [f"Com base no documento da UFOP (fonte: {fonte}):\n"]

    if numero_res:
        partes.append(f"**Documento**: {numero_res.group(0).strip()}")
    if data_texto:
        partes.append(f"**Data**: {data_texto.group(1).strip()}")

    if artigos:
        partes.append("\n**Principais disposições:**")
        for art in artigos[:5]:  # Máx 5 artigos por exemplo
            partes.append(f"• {art.strip()}")

    # Se não conseguiu extrair estrutura, usa o trecho inicial do texto
    if len(partes) <= 2:
        trecho = texto_doc[:600].strip()
        partes.append(f"\n{trecho}")

    return "\n".join(partes)


def formatar_do_jsonl_original(registro: dict) -> dict:
    """
    Formata um registro do dataset_ufop.jsonl ORIGINAL.
    Gera a resposta correta a partir do campo 'input' (já que 'output' está vazio).
    """
    instrucao  = registro.get('instruction', 'Extraia as principais informações desta resolução da UFOP.')
    texto_bruto = registro.get('input', '')
    output_orig = registro.get('output', '').strip()

    texto_limpo = limpar_texto_jsonl(texto_bruto)
    if len(texto_limpo) < 80:
        return None  # Documento vazio, ignorar

    # Detectar nome da resolução/documento
    fonte = "Documento UFOP"
    match_sei = re.search(r'RESOLUÇÃO[^\n]*N[ºo°]?\s*([\d.]+)', texto_bruto, re.IGNORECASE)
    if match_sei:
        fonte = f"Resolução UFOP Nº {match_sei.group(1)}"

    # Usar output original se existir; caso contrário, gerar a partir do texto
    if output_orig:
        resposta = output_orig
    else:
        resposta = gerar_resposta_do_documento(texto_limpo, fonte)

    return {
        "text": (
            f"<|im_start|>system\n{SISTEMA_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{instrucao}\n\n"
            f"DOCUMENTO:\n{texto_limpo[:1200]}<|im_end|>\n"
            f"<|im_start|>assistant\n{resposta}<|im_end|>\n"
        )
    }


def formatar_do_chunk(chunk: dict) -> dict:
    """Formata um chunk de texto como exemplo de treino (fonte: chunks.jsonl)."""
    texto  = chunk['texto']
    fonte  = chunk['fonte']
    return {
        "text": (
            f"<|im_start|>system\n{SISTEMA_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Explique o que descreve este trecho do documento '{fonte}':\n\n{texto}"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"Com base no documento da UFOP '{fonte}':\n\n{texto}"
            f"<|im_end|>\n"
        )
    }


# ─── Escolher fonte de dados ──────────────────────────────────
exemplos = []

if os.path.exists(ARQUIVO_JSONL_ORIGINAL):
    print(f"⚡ Usando dataset_ufop.jsonl como fonte principal de treino...")
    registros = []
    with open(ARQUIVO_JSONL_ORIGINAL, 'r', encoding='utf-8') as f:
        for linha in f:
            try:
                registros.append(json.loads(linha))
            except Exception:
                pass

    com_output = sum(1 for r in registros if r.get('output', '').strip())
    sem_output = len(registros) - com_output
    print(f"📊 Registros encontrados : {len(registros)}")
    print(f"   Com output original   : {com_output}")
    print(f"   Com output VAZIO      : {sem_output} → gerando respostas automaticamente")

    for reg in registros:
        ex = formatar_do_jsonl_original(reg)
        if ex:
            exemplos.append(ex)

    print(f"✅ {len(exemplos)} exemplos gerados do JSONL original")

else:
    print("📦 dataset_ufop.jsonl não encontrado. Usando chunks.jsonl...")
    todos_chunks = []
    with open(ARQUIVO_CHUNKS, 'r', encoding='utf-8') as f:
        for linha in f:
            try:
                todos_chunks.append(json.loads(linha))
            except Exception:
                pass
    exemplos = [formatar_do_chunk(c) for c in todos_chunks]
    print(f"✅ {len(exemplos)} exemplos gerados dos chunks")

# ─── Amostrar e criar dataset ─────────────────────────────────
random.seed(SEED_ALEATORIO)
if len(exemplos) > MAX_EXEMPLOS:
    exemplos = random.sample(exemplos, MAX_EXEMPLOS)
    print(f"⚠️  Amostrando {MAX_EXEMPLOS} exemplos (de {len(exemplos)} disponíveis)")

dataset = Dataset.from_list(exemplos)
split = dataset.train_test_split(test_size=0.05, seed=SEED_ALEATORIO)
dataset_treino = split['train']
dataset_val    = split['test']

print(f"\n✅ Dataset pronto:")
print(f"   Treino    : {len(dataset_treino)} exemplos")
print(f"   Validação : {len(dataset_val)} exemplos")
print(f"\nExemplo (primeiros 600 chars):")
print(dataset_treino[0]['text'][:600])


# %%  CÉLULA 4 — Carregar modelo Qwen2.5-1.5B com QLoRA (4-bit)

import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print("⏳ Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✅ Tokenizer carregado!")

print("⏳ Carregando modelo 1.5B em 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.enable_input_require_grads()   # sem gradient checkpointing — 1.5B cabe fácil

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

alocado = torch.cuda.memory_allocated() / 1e9
livre   = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
print(f"✅ Modelo pronto! GPU: {alocado:.1f}GB | Livre: {livre:.1f}GB")
model.print_trainable_parameters()


# %%  CÉLULA 5 — Treinamento com Trainer (Transformers)

import os, shutil, torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import Dataset as TorchDataset

PASTA_CHECKPT = "/content/ufop_checkpoints"
PASTA_ADAPTER = "/content/drive/MyDrive/ufop_ia_dados/modelo_ufop_adapter_v1_5b"
PASTA_BACKUP  = "/content/drive/MyDrive/ufop_ia_dados/checkpoints_backup"
os.makedirs(PASTA_CHECKPT, exist_ok=True)
os.makedirs(PASTA_BACKUP,  exist_ok=True)

class UFOPDataset(TorchDataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        print(f"⏳ Tokenizando {len(hf_dataset)} exemplos...")
        self.items = []
        for ex in hf_dataset:
            enc = tokenizer(ex["text"], max_length=max_length,
                            truncation=True, padding="max_length", return_tensors="pt")
            ids  = enc["input_ids"][0]
            mask = enc["attention_mask"][0]
            lbl  = ids.clone()
            lbl[mask == 0] = -100
            self.items.append({"input_ids": ids, "attention_mask": mask, "labels": lbl})
        print(f"✅ {len(self.items)} exemplos prontos!")
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

train_ds = UFOPDataset(dataset_treino, tokenizer)
eval_ds  = UFOPDataset(dataset_val, tokenizer)

class DriveBackupCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        src  = os.path.join(args.output_dir, f"checkpoint-{step}")
        dst  = os.path.join(PASTA_BACKUP, f"checkpoint-{step}")
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"\n💾 Checkpoint-{step} salvo no Drive!")
        return control

training_args = TrainingArguments(
    output_dir=PASTA_CHECKPT,
    num_train_epochs=3,
    per_device_train_batch_size=1,      # ← Reduzido para 1 (máxima segurança)
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,      # ← Ajustado para manter batch efetivo
    gradient_checkpointing=False,       # ← Já ativado no modelo (Passo 1)
    optim="paged_adamw_8bit",
    fp16=True,
    bf16=False,
    learning_rate=2e-4,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_steps=10,
    eval_steps=100,
    save_steps=100,
    save_total_limit=3,
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    callbacks=[DriveBackupCallback()],
)

print("🚀 Iniciando treinamento 1.5B (Batch 1 + GC)...")
print(f"   Treino: {len(train_ds)} | Validação: {len(eval_ds)} | Épocas: 3")
trainer.train()
print("\n✅ Treinamento concluído!")


# %%  CÉLULA 6 — Salvar adapter no Google Drive

PASTA_ADAPTER = "/content/drive/MyDrive/ufop_ia_dados/modelo_ufop_adapter_v2"
print("⏳ Salvando adapter final consolidado no Drive...")

os.makedirs(PASTA_ADAPTER, exist_ok=True)
# Salvando diretamente do objeto 'model' que carregamos no Passo anterior
model.save_pretrained(PASTA_ADAPTER)
tokenizer.save_pretrained(PASTA_ADAPTER)

print(f"✅ Adapter consolidado e salvo com sucesso em: {PASTA_ADAPTER}")


# %%  CÉLULA 7 — Teste rápido do modelo treinado

import torch

# ─── Recriando a variável que foi apagada no reinício ───
SISTEMA_PROMPT = (
    "Você é um assistente especializado em documentos da UFOP, "
    "incluindo resoluções, editais do PIBIC, regulamentos e normas. "
    "Responda com base apenas nos documentos fornecidos, "
    "de forma clara, objetiva e em português."
)
# ────────────────────────────────────────────────────────

model.eval()

def testar_modelo(pergunta: str) -> str:
    """Testa o modelo fine-tuned diretamente (sem RAG)."""
    prompt = (
        f"<|im_start|>system\n{SISTEMA_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{pergunta}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    gerado = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gerado, skip_special_tokens=True).strip()


perguntas = [
    "O que é o PIBIC e quem pode participar?",
    "Quais são os prazos para entrega de relatórios de IC?",
]

for pergunta in perguntas:
    print(f"🔍 Pergunta: {pergunta}")
    resposta = testar_modelo(pergunta)
    print(f"🤖 Resposta: {resposta}")
    print("─" * 60)

print("\n✅ ETAPA 3 CONCLUÍDA!")
print("Baixe o adapter e o banco chromadb do Drive para usar localmente.")
