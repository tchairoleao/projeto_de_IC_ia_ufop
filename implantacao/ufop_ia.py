import os
import sys
import subprocess
import warnings

# Silencia avisos desnecessários do HuggingFace e PyTorch para um terminal mais limpo
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ==========================================
# 0. VERIFICAÇÃO E INSTALAÇÃO AUTOMÁTICA
# ==========================================
def verificar_e_instalar():
    print("="*60)
    print("🔍 Verificando integridade do sistema...")
    print("="*60)

    precisa_instalar = False
    try:
        import torch
        import chromadb
        import sentence_transformers
        import transformers
        import bitsandbytes
        import accelerate
        import peft
    except ImportError:
        precisa_instalar = True

    if precisa_instalar:
        print("⚠️ Primeira execução detectada no PC.")
        print("⏳ Baixando e instalando os pré-requisitos automáticos...")
        print("☕ Isso pode demorar alguns minutos. Aguarde...\n")

        # Instala o PyTorch para Placa de Vídeo (CUDA 12.1)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])

        # Instala as bibliotecas de IA (versões fixadas para estabilidade)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "chromadb",
            "sentence-transformers",
            "transformers>=4.45.0",
            "bitsandbytes>=0.43.0",
            "accelerate>=0.34.0",
            "peft>=0.12.0",
        ])

        print("\n✅ Instalação concluída com sucesso!\n")
    else:
        print("✅ Bibliotecas já instaladas! Inicializando...\n")

verificar_e_instalar()

# ==========================================
# IMPORTAÇÕES DA IA
# ==========================================
import torch
import chromadb
import gc
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# 1. CONFIGURAÇÕES LOCAIS
# ==========================================
PASTA_CHROMADB = "./chromadb"
PASTA_ADAPTER  = "./adapter"   # ← Copie aqui o conteúdo da pasta salva na Etapa 3
                                #   (no Drive: modelo_ufop_adapter_v2)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

print("="*60)
print("🚀 INICIANDO O ASSISTENTE DA UFOP (MOTOR 7B LOCAL)")
print("="*60)

# ==========================================
# 2. VERIFICAÇÃO DE GPU
# ==========================================
# O carregamento em 4-bit (bitsandbytes) exige uma GPU NVIDIA com CUDA.
# Checar isso ANTES de tentar carregar o modelo evita um traceback confuso
# para quem for rodar o programa sem entender de Python.
if not torch.cuda.is_available():
    print("\n❌ ERRO: Nenhuma GPU NVIDIA com suporte a CUDA foi detectada.")
    print("   Este assistente precisa de uma placa de vídeo NVIDIA para funcionar")
    print("   (o modelo é carregado em 4-bit, o que exige aceleração por GPU).")
    print("\n   Possíveis causas:")
    print("   • O computador não tem placa de vídeo NVIDIA dedicada")
    print("   • O driver da NVIDIA não está instalado/atualizado")
    print("   • O PyTorch foi instalado na versão CPU-only")
    input("\nPressione ENTER para sair...")
    sys.exit(1)

nome_gpu = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"✅ GPU detectada: {nome_gpu} ({vram_gb:.1f} GB VRAM)")

VRAM_MINIMA_RECOMENDADA = 6.0  # GB — o modelo 7B em 4-bit precisa de ~5.5-6.5GB
if vram_gb < VRAM_MINIMA_RECOMENDADA:
    print(f"⚠️  Aviso: sua GPU tem {vram_gb:.1f} GB de VRAM. O recomendado é "
          f"{VRAM_MINIMA_RECOMENDADA:.0f}GB ou mais para carregar o modelo 7B com folga.")
    print("   Vamos tentar carregar mesmo assim — se faltar memória, o programa avisa")
    print("   claramente em vez de travar sem explicação.\n")
else:
    print()

# ==========================================
# 3. CARREGANDO O MOTOR DE BUSCA (RAG)
# ==========================================
print("⏳ 1/4 Carregando banco de dados das Resoluções (ChromaDB)...")

if not os.path.exists(PASTA_CHROMADB):
    print(f"\n❌ ERRO: A pasta '{PASTA_CHROMADB}' não foi encontrada.")
    print("   Verifique se a pasta 'chromadb' está ao lado deste programa")
    print("   (dentro da mesma pasta 'Assistente_UFOP').")
    input("\nPressione ENTER para sair...")
    sys.exit(1)

try:
    modelo_emb = SentenceTransformer('intfloat/multilingual-e5-base')
    client = chromadb.PersistentClient(path=PASTA_CHROMADB)
    colecao = client.get_collection(name="ufop_documentos")
    print(f"✅ Banco carregado! Documentos indexados: {colecao.count()}\n")
except Exception as e:
    print(f"\n❌ ERRO ao carregar o banco de dados ChromaDB: {e}")
    print("   A pasta 'chromadb' pode estar incompleta ou corrompida.")
    input("\nPressione ENTER para sair...")
    sys.exit(1)


def buscar_contexto(pergunta: str, n_resultados: int = 5) -> str:
    emb = modelo_emb.encode([f"query: {pergunta}"])
    res = colecao.query(
        query_embeddings=emb.tolist(),
        n_results=n_resultados,
        include=['documents', 'metadatas']
    )

    textos_encontrados = []
    for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
        textos_encontrados.append(f"[Fonte: {meta['fonte']}]\n{doc}")

    return "\n\n".join(textos_encontrados)

# ==========================================
# 4. CARREGANDO O CÉREBRO DA IA (7B em 4-bit + Adapter LoRA)
# ==========================================
print("⏳ 2/4 Carregando o vocabulário (Tokenizer)...")
if os.path.exists(PASTA_ADAPTER):
    # Usa o tokenizer salvo junto do adapter (garante compatibilidade exata com o treino,
    # sem depender de internet ou de o repositório no Hugging Face não ter mudado)
    tokenizer = AutoTokenizer.from_pretrained(PASTA_ADAPTER, trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("⏳ 3/4 Carregando o Modelo Base de 7 Bilhões na Placa de Vídeo...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

try:
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
except torch.cuda.OutOfMemoryError:
    print("\n❌ ERRO: Memória da GPU (VRAM) insuficiente para carregar o modelo.")
    print(f"   Sua GPU tem {vram_gb:.1f} GB — este modelo precisa de ~6GB ou mais.")
    print("   Feche outros programas que usem a GPU (jogos, outro navegador com muitas abas,")
    print("   outros processos de IA) e tente novamente. Se persistir, esta máquina pode não")
    print("   ter VRAM suficiente para rodar o modelo de 7B nesta configuração.")
    input("\nPressione ENTER para sair...")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERRO ao carregar o modelo base: {e}")
    input("\nPressione ENTER para sair...")
    sys.exit(1)

print("⏳ 4/4 Aplicando o ajuste fino (adapter LoRA) treinado para a UFOP...")

if not os.path.exists(PASTA_ADAPTER):
    print(f"\n⚠️  AVISO: A pasta do adapter '{PASTA_ADAPTER}' não foi encontrada.")
    print("   O assistente vai rodar com o modelo Qwen2.5-7B-Instruct BASE,")
    print("   ou seja, SEM o ajuste fino treinado nos documentos da UFOP.")
    print("   Para usar a versão treinada, copie a pasta do adapter (salva no")
    print("   Drive como 'modelo_ufop_adapter_v2') para './adapter' aqui do lado.")
    resposta = input("\n   Continuar mesmo assim com o modelo base? (s/n): ").strip().lower()
    if resposta != "s":
        sys.exit(0)
    model = model_base
else:
    try:
        model = PeftModel.from_pretrained(model_base, PASTA_ADAPTER)
        print("✅ Adapter aplicado com sucesso!")
    except Exception as e:
        print(f"\n❌ ERRO ao aplicar o adapter: {e}")
        print("   Verifique se a pasta './adapter' contém os arquivos corretos")
        print("   (adapter_config.json, adapter_model.safetensors, etc.)")
        input("\nPressione ENTER para sair...")
        sys.exit(1)

model.eval()
print("✅ IA GIGANTE CARREGADA COM SUCESSO!\n")

# ==========================================
# 5. A LÓGICA DE GERAÇÃO INTELIGENTE
# ==========================================
SISTEMA_PROMPT = (
    "Você é o Assistente Virtual Oficial da Universidade Federal de Ouro Preto (UFOP). "
    "Sua função é ler os documentos oficiais fornecidos e responder à pergunta do usuário de forma clara, "
    "educada e direta. Não use jargões excessivos e não copie cabeçalhos de ofícios."
)

def perguntar_a_ia(pergunta_usuario: str) -> str:
    print("\n[Buscando dados nos documentos da UFOP...]")
    contexto_verdadeiro = buscar_contexto(pergunta_usuario)

    prompt = (
        f"<|im_start|>system\n{SISTEMA_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"DOCUMENTOS ENCONTRADOS:\n{contexto_verdadeiro}\n\n"
        f"Pergunta: {pergunta_usuario}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    gerado = outputs[0][inputs["input_ids"].shape[-1]:]
    resposta = tokenizer.decode(gerado, skip_special_tokens=True).strip()
    return resposta

# ==========================================
# 6. O CHATBOT (Loop interativo)
# ==========================================
print("💬 Digite sua pergunta sobre as normas da UFOP (ou 'sair' para encerrar):")
while True:
    pergunta = input("\n👤 Você: ")
    if pergunta.lower() in ['sair', 'exit', 'quit']:
        print("Encerrando o sistema. Até logo!")
        break

    try:
        resposta = perguntar_a_ia(pergunta)
        print(f"\n🤖 Assistente UFOP:\n{resposta}")
    except Exception as e:
        print(f"\n⚠️  Ocorreu um erro ao gerar a resposta: {e}")
        print("   Tente reformular a pergunta ou reinicie o programa.")
    print("─" * 70)

    # LIMPEZA DE MEMÓRIA (Anti-Crash)
    # Garante que a Placa de Vídeo não vai encher após várias perguntas
    gc.collect()
    torch.cuda.empty_cache()
