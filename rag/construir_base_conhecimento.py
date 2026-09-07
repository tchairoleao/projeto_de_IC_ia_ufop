# ================================================================
#  IA UFOP — ETAPA 2: Extração de PDFs e Criação do Banco RAG
# ================================================================
# Como usar no Google Colab:
#   1. Abra o Colab: colab.research.google.com
#   2. Crie um novo notebook
#   3. Copie cada bloco marcado com "# %%  CÉLULA X" como uma
#      célula separada no notebook
#   4. Execute célula por célula, de cima para baixo
#
# ANTES DE COMEÇAR:
#   No Google Drive web (drive.google.com):
#   1. Vá em "Compartilhados comigo"
#   2. Encontre a pasta "IC_UFOP_Arquivos" (dentro de PROPPI, PIBIC-TI, Tchairo)
#   3. Clique com botão direito → "Adicionar atalho ao Google Drive"
#   4. Escolha "Meu Drive" → "Adicionar"
#   Depois disso o Colab vai conseguir acessar os arquivos.
#
# ⚡ ATALHO DISPONÍVEL:
#   Se você já tem o arquivo "dataset_ufop.jsonl" na pasta IC_UFOP_Arquivos,
#   a CÉLULA 3A usa ele diretamente (muito mais rápido que re-extrair PDFs)!
#   A CÉLULA 3B extrai os PDFs que não estão no JSONL (complementar).
# ================================================================


# %%  CÉLULA 1 — Instalação de dependências
# ⏱️ Tempo estimado: 2-3 minutos

import subprocess, sys

packages = [
    "pymupdf",          # Extração de texto de PDF
    "pdfplumber",       # Fallback para PDFs difíceis
    "sentence-transformers",  # Embeddings multilíngues
    "chromadb",         # Banco vetorial
    "tqdm",             # Barra de progresso
]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✅ Dependências instaladas com sucesso!")


# %%  CÉLULA 2 — Montar Google Drive e configurar caminhos

from google.colab import drive
drive.mount('/content/drive')

import os

# ─── AJUSTE AQUI SE NECESSÁRIO ────────────────────────────────
#  Se a pasta foi adicionada com outro nome ou em subpasta, altere:
PASTA_PDFS    = "/content/drive/MyDrive/IC_UFOP_Arquivos"
PASTA_SAIDA   = "/content/drive/MyDrive/ufop_ia_dados"
PASTA_CHROMADB = os.path.join(PASTA_SAIDA, "chromadb")
ARQUIVO_CHUNKS = os.path.join(PASTA_SAIDA, "chunks.jsonl")
# ──────────────────────────────────────────────────────────────

os.makedirs(PASTA_SAIDA, exist_ok=True)

# Verificar acesso à pasta de PDFs
if not os.path.exists(PASTA_PDFS):
    print(f"❌ ERRO: Pasta não encontrada em '{PASTA_PDFS}'")
    print()
    print("Soluções:")
    print("  1. Verifique se adicionou o atalho ao 'Meu Drive' (veja instruções acima)")
    print("  2. Se a pasta está em outro local, altere a variável PASTA_PDFS acima")
    print()
    # Tentar listar Meu Drive para ajudar na localização
    print("Conteúdo de 'Meu Drive':")
    for item in sorted(os.listdir("/content/drive/MyDrive"))[:20]:
        print(f"  📁 {item}" if os.path.isdir(f"/content/drive/MyDrive/{item}") else f"  📄 {item}")
else:
    # Contar PDFs (incluindo subpastas)
    total_pdfs = sum(
        1 for _, _, files in os.walk(PASTA_PDFS)
        for f in files if f.lower().endswith('.pdf')
    )
    print(f"✅ Pasta encontrada! Total de PDFs (incluindo subpastas): {total_pdfs}")
    print(f"📂 Saída dos dados: {PASTA_SAIDA}")


# %%  CÉLULA 3A — ⚡ CAMINHO RÁPIDO: Usar dataset_ufop.jsonl existente
# ⏱️ Tempo estimado: 2-5 minutos (vs 30-90 min re-extraindo PDFs)
#
# IMPORTANTE: O campo "output" do seu JSONL estava VAZIO em todos os exemplos.
# Isso explica por que o modelo anterior não respondia nada — ele foi treinado
# para produzir respostas vazias. Este script usa apenas o campo "input"
# (texto do documento) para criar o banco de busca RAG. O problema de treino
# será corrigido na Etapa 2 (ver finetuning/treino_qlora.py).
#
# Se você NÃO tem o dataset_ufop.jsonl, pule para a CÉLULA 3B.

import json
import os
import re
from tqdm import tqdm

ARQUIVO_JSONL_ORIGINAL = os.path.join(PASTA_PDFS, "dataset_ufop.jsonl")


def limpar_texto_jsonl(texto: str) -> str:
    """
    Limpa o texto extraído do campo 'input' do JSONL.
    Remove ruído típico: URLs do SEI, timestamps, cabeçalhos, rodapés.
    """
    # Remover timestamps (ex: "03/12/2021 19:37")
    texto = re.sub(r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}', '', texto)
    # Remover URLs do SEI/UFOP
    texto = re.sub(r'https?://\S+', '', texto)
    # Remover linhas com apenas números de página (ex: "1/1", "2/3")
    texto = re.sub(r'^\s*\d+/\d+\s*$', '', texto, flags=re.MULTILINE)
    # Remover linhas de cabeçalho repetitivas do SEI
    texto = re.sub(r'SEI/UFOP\s*-\s*\d+\s*-.*', '', texto)
    # Remover rodapé padrão da UFOP
    texto = re.sub(r'R\.\s*Diogo de Vasconcelos.*', '', texto, flags=re.DOTALL)
    # Remover "Documento assinado eletronicamente..." (rodapé)
    texto = re.sub(r'Documento assinado eletronicamente.*?\d{4}\.', '', texto, flags=re.DOTALL)
    # Normalizar espaçamento
    texto = re.sub(r'[ \t]{2,}', ' ', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto.strip()


def limpar_texto(texto: str) -> str:
    """Versão geral para texto de PDFs."""
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r'[ \t]{2,}', ' ', texto)
    return texto.strip()


def chunkar_texto(texto: str, nome_arquivo: str,
                  tamanho_chunk: int = 400,
                  sobreposicao: int = 60) -> list:
    """
    Divide o texto em chunks com sobreposição para preservar contexto.
    Tenta respeitar quebras de parágrafo.
    """
    paragrafos = [p.strip() for p in texto.split('\n\n') if p.strip()]
    chunks = []
    buffer_palavras = []

    for paragrafo in paragrafos:
        palavras_par = paragrafo.split()
        buffer_palavras.extend(palavras_par)

        while len(buffer_palavras) >= tamanho_chunk:
            chunk_texto = " ".join(buffer_palavras[:tamanho_chunk])
            if len(chunk_texto) > 80:
                chunks.append({
                    "texto": chunk_texto,
                    "fonte": nome_arquivo,
                })
            buffer_palavras = buffer_palavras[tamanho_chunk - sobreposicao:]

    if len(buffer_palavras) > 30:
        chunk_texto = " ".join(buffer_palavras)
        chunks.append({"texto": chunk_texto, "fonte": nome_arquivo})

    return chunks


if not os.path.exists(ARQUIVO_JSONL_ORIGINAL):
    print(f"⚠️  dataset_ufop.jsonl não encontrado em '{ARQUIVO_JSONL_ORIGINAL}'")
    print("   Pule para a CÉLULA 3B para extrair os PDFs diretamente.")
else:
    print(f"✅ dataset_ufop.jsonl encontrado! Carregando...")

    # Verificar checkpoint
    ja_processados_jsonl = set()
    if os.path.exists(ARQUIVO_CHUNKS):
        with open(ARQUIVO_CHUNKS, 'r', encoding='utf-8') as f:
            for linha in f:
                try:
                    dado = json.loads(linha)
                    ja_processados_jsonl.add(dado.get('fonte', ''))
                except Exception:
                    pass

    # Carregar registros do JSONL original
    registros = []
    with open(ARQUIVO_JSONL_ORIGINAL, 'r', encoding='utf-8') as f:
        for linha in f:
            try:
                registros.append(json.loads(linha))
            except Exception:
                pass

    print(f"📚 Registros no dataset_ufop.jsonl: {len(registros)}")

    # Estatísticas dos outputs
    com_output    = sum(1 for r in registros if r.get('output', '').strip())
    sem_output    = len(registros) - com_output
    print(f"   Com output preenchido : {com_output}")
    print(f"   Com output VAZIO      : {sem_output}  ← será corrigido na Etapa 2")

    # ─── Converter registros em chunks para RAG ────────────────
    total_chunks_jsonl = 0
    fonte_jsonl = "dataset_ufop.jsonl"

    if fonte_jsonl not in ja_processados_jsonl:
        with open(ARQUIVO_CHUNKS, 'a', encoding='utf-8') as f_saida:
            for reg in tqdm(registros, desc="Convertendo JSONL → chunks"):
                texto_bruto  = reg.get('input', '')
                texto_limpo  = limpar_texto_jsonl(texto_bruto)

                if len(texto_limpo) < 100:
                    continue  # Documento vazio ou só ruído

                # Tentar recuperar nome do documento do texto
                # (ex: "SEI/UFOP - 0242202 - Resolução")
                nome_doc = fonte_jsonl
                match_sei = re.search(r'SEI/UFOP\s*-\s*(\d+)\s*-\s*(.+?)\n', texto_bruto)
                if match_sei:
                    nome_doc = f"SEI_{match_sei.group(1)}_{match_sei.group(2).strip()}"

                chunks = chunkar_texto(texto_limpo, nome_doc)
                for chunk in chunks:
                    f_saida.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks_jsonl += len(chunks)

        print(f"\n✅ JSONL processado com sucesso!")
        print(f"   Chunks gerados: {total_chunks_jsonl}")
        print(f"   Salvo em: {ARQUIVO_CHUNKS}")
        print()
        print("📌 Continue para a CÉLULA 3B para adicionar PDFs não incluídos no JSONL,")
        print("   ou pule direto para a CÉLULA 4 se o JSONL já contém todos os documentos.")
    else:
        print(f"📌 JSONL já foi processado anteriormente (checkpoint). Pulando.")
        print("   Continue para a CÉLULA 4.")


# %%  CÉLULA 3B — Extração de PDFs complementares (execute após 3A)
# ⏱️ Tempo estimado: 20-60 min para PDFs não incluídos no JSONL
# ✅ Com checkpoint: retoma de onde parou se o Colab desconectar!
#
# Se já executou a CÉLULA 3A e o JSONL contém todos os documentos,
# você pode PULAR esta célula e ir direto para a CÉLULA 4.

import fitz  # PyMuPDF


def extrair_texto_pdf(caminho_pdf: str) -> str:
    """Extrai texto de um PDF. Tenta PyMuPDF primeiro, depois pdfplumber."""
    try:
        doc = fitz.open(caminho_pdf)
        partes = []
        for num_pag, pagina in enumerate(doc):
            texto = pagina.get_text("text")
            if texto.strip():
                partes.append(f"[Pág. {num_pag + 1}] {texto}")
        doc.close()
        texto_final = limpar_texto("\n".join(partes))
        if len(texto_final) > 200:
            return texto_final
    except Exception:
        pass

    try:
        import pdfplumber
        partes = []
        with pdfplumber.open(caminho_pdf) as pdf:
            for num_pag, pagina in enumerate(pdf.pages):
                texto = pagina.extract_text() or ""
                if texto.strip():
                    partes.append(f"[Pág. {num_pag + 1}] {texto}")
        texto_final = limpar_texto("\n".join(partes))
        if len(texto_final) > 200:
            return texto_final
    except Exception:
        pass

    return ""


# ─── Checkpoint ───────────────────────────────────────────────
arquivos_ja_processados = set()
if os.path.exists(ARQUIVO_CHUNKS):
    with open(ARQUIVO_CHUNKS, 'r', encoding='utf-8') as f:
        for linha in f:
            try:
                dado = json.loads(linha)
                arquivos_ja_processados.add(dado['fonte'])
            except Exception:
                pass
    print(f"📌 Checkpoint: {len(arquivos_ja_processados)} fonte(s) já processada(s).")

# ─── Listar PDFs da pasta ─────────────────────────────────────
todos_pdfs = []
for raiz, _, arquivos in os.walk(PASTA_PDFS):
    for arq in arquivos:
        if arq.lower().endswith('.pdf'):
            todos_pdfs.append(os.path.join(raiz, arq))

pdfs_novos = [
    p for p in todos_pdfs
    if os.path.basename(p) not in arquivos_ja_processados
]

print(f"📚 PDFs encontrados na pasta : {len(todos_pdfs)}")
print(f"📥 PDFs novos para processar : {len(pdfs_novos)}")

if not pdfs_novos:
    print("\n✅ Nenhum PDF novo para processar. Vá para a CÉLULA 4.")
else:
    total_chunks_pdf = 0
    erros_pdf = []

    with open(ARQUIVO_CHUNKS, 'a', encoding='utf-8') as f_saida:
        for caminho in tqdm(pdfs_novos, desc="Extraindo PDFs"):
            nome = os.path.basename(caminho)
            texto = extrair_texto_pdf(caminho)

            if not texto:
                erros_pdf.append(nome)
                continue

            chunks = chunkar_texto(texto, nome)
            for chunk in chunks:
                f_saida.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total_chunks_pdf += len(chunks)

    print(f"\n✅ PDFs extraídos!")
    print(f"   Chunks gerados : {total_chunks_pdf}")
    print(f"   PDFs sem texto : {len(erros_pdf)} (provavelmente escaneados)")
    if erros_pdf:
        print(f"   Primeiros erros: {erros_pdf[:5]}")
    print(f"   Salvo em: {ARQUIVO_CHUNKS}")


# %%  CÉLULA 4 — Criar Banco Vetorial com ChromaDB
# ⏱️ Tempo estimado: 10-30 minutos
# ✅ Com checkpoint: retoma de onde parou se Colab desconectar!

from sentence_transformers import SentenceTransformer
import chromadb
import json
import os
from tqdm import tqdm

# ─── Carregar modelo de embeddings ────────────────────────────
print("⏳ Carregando modelo de embeddings multilíngue...")
# multilingual-e5-base: excelente qualidade para português
modelo_emb = SentenceTransformer('intfloat/multilingual-e5-base')
print("✅ Modelo de embeddings pronto!")

# ─── Inicializar ChromaDB persistente ─────────────────────────
os.makedirs(PASTA_CHROMADB, exist_ok=True)
client = chromadb.PersistentClient(path=PASTA_CHROMADB)
colecao = client.get_or_create_collection(
    name="ufop_documentos",
    metadata={"hnsw:space": "cosine"}
)
print(f"📊 Documentos já no banco: {colecao.count()}")

# ─── Carregar todos os chunks ──────────────────────────────────
todos_chunks = []
with open(ARQUIVO_CHUNKS, 'r', encoding='utf-8') as f:
    for i, linha in enumerate(f):
        try:
            chunk = json.loads(linha)
            chunk['_id'] = f"chunk_{i}"
            todos_chunks.append(chunk)
        except Exception:
            pass

print(f"📚 Total de chunks carregados: {len(todos_chunks)}")

# ─── Verificar quais já foram indexados (checkpoint) ──────────
ids_no_banco = set()
if colecao.count() > 0:
    resultado = colecao.get(include=[])
    ids_no_banco = set(resultado['ids'])

chunks_novos = [c for c in todos_chunks if c['_id'] not in ids_no_banco]
print(f"📥 Chunks para indexar (novos): {len(chunks_novos)}")

if not chunks_novos:
    print("✅ Todos os chunks já estão indexados!")
else:
    # ─── Indexar em lotes ─────────────────────────────────────
    LOTE = 64  # Número de chunks por lote

    for i in tqdm(range(0, len(chunks_novos), LOTE), desc="Indexando no ChromaDB"):
        lote = chunks_novos[i:i + LOTE]

        # Prefixo "passage:" para o modelo multilingual-e5
        textos_emb = [f"passage: {c['texto']}" for c in lote]
        embeddings  = modelo_emb.encode(textos_emb, batch_size=32, show_progress_bar=False)

        colecao.add(
            embeddings=embeddings.tolist(),
            documents=[c['texto'] for c in lote],
            ids=[c['_id'] for c in lote],
            metadatas=[{"fonte": c['fonte']} for c in lote],
        )

    print(f"\n✅ Banco vetorial criado com sucesso!")
    print(f"   Total de documentos indexados: {colecao.count()}")
    print(f"   Salvo em: {PASTA_CHROMADB}")


# %%  CÉLULA 5 — Teste do sistema RAG (validação)

from sentence_transformers import SentenceTransformer

def buscar_docs(pergunta: str, n: int = 3) -> list:
    """Retorna os N trechos mais relevantes para a pergunta."""
    emb = modelo_emb.encode([f"query: {pergunta}"])
    res = colecao.query(
        query_embeddings=emb.tolist(),
        n_results=n,
        include=['documents', 'metadatas', 'distances']
    )
    return list(zip(res['documents'][0], res['metadatas'][0], res['distances'][0]))

# ─── Testar com algumas perguntas típicas ─────────────────────
perguntas_teste = [
    "Quais são os requisitos para participar do PIBIC?",
    "Qual é o prazo para entrega de relatórios?",
    "Como funciona a bolsa de iniciação científica?",
]

for pergunta in perguntas_teste:
    print(f"\n🔍 Pergunta: {pergunta}")
    resultados = buscar_docs(pergunta, n=2)
    for i, (doc, meta, dist) in enumerate(resultados, 1):
        similaridade = 1 - dist
        print(f"  [{i}] Similaridade: {similaridade:.2%} | Fonte: {meta['fonte']}")
        print(f"       {doc[:200]}...")
    print()

print("="*60)
print("✅ ETAPA 1 CONCLUÍDA!")
print()
print("Próximos passos:")
print("  → Para USAR o modelo agora (mais rápido):")
print("    Baixe a pasta 'ufop_ia_dados/chromadb' do Drive e siga o README")
print()
print("  → Para TREINAR o modelo (melhora qualidade, mas demora mais):")
print("    Execute o script 'finetuning/treino_qlora.py' no Colab")
