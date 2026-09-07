# IA UFOP — RAG e Ajuste Fino de LLMs para Automação de Processos Administrativos Públicos

> Projeto de Iniciação Científica desenvolvido na Universidade Federal de Ouro Preto (UFOP), entre fevereiro e agosto de 2026.
>
> **Discente:** Tchairô Niddan Ruiz Gimenez Leão
> **Orientador:** Prof. Thiago Fontes Santos
> **Status:** Encerrado (ciclo de IC) — trabalhos futuros mapeados na seção [Roadmap](#roadmap)

---

## Sumário

- [Visão geral](#visão-geral)
- [Motivação](#motivação)
- [Arquitetura](#arquitetura)
- [Pipeline em detalhe](#pipeline-em-detalhe)
  - [1. Coleta de dados](#1-coleta-de-dados)
  - [2. Base de conhecimento vetorial (RAG)](#2-base-de-conhecimento-vetorial-rag)
  - [3. Ajuste fino do modelo de linguagem](#3-ajuste-fino-do-modelo-de-linguagem)
  - [4. Implantação e restrições de hardware](#4-implantação-e-restrições-de-hardware)
- [Stack técnica](#stack-técnica)
- [Resultados](#resultados)
- [Limitações e lições aprendidas](#limitações-e-lições-aprendidas)
- [Roadmap](#roadmap)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como reproduzir](#como-reproduzir)
- [Publicação](#publicação)
- [Agradecimentos](#agradecimentos)
- [Licença](#licença)

---

## Visão geral

Este projeto investiga em que medida a combinação entre **recuperação de informação semântica (RAG)** e **modelos de linguagem de código aberto** é capaz de automatizar, com fidelidade aos documentos oficiais, a extração de informações e a resposta a perguntas sobre normas administrativas de uma universidade federal.

O sistema final constrói um acervo digital de documentos oficiais da UFOP, indexa esse acervo em uma base vetorial de busca semântica e integra essa base a um modelo de linguagem (família **Qwen2.5**) por meio de um assistente conversacional capaz de responder perguntas sobre resoluções, editais e normas internas.

## Motivação

A administração pública brasileira — e universidades federais em particular, produz um volume expressivo de documentos normativos cuja localização, interpretação e síntese consomem tempo considerável de servidores, docentes e discentes. A ausência de mecanismos automatizados de busca contribui para atrasos processuais, retrabalho e, em alguns casos, a aplicação de normas já revogadas.

O objetivo geral do projeto foi adaptar um modelo de linguagem de código aberto ao domínio da administração pública da UFOP, automatizando a extração e a consulta a documentos oficiais.

## Arquitetura

```
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│   Coleta (SOC)   │ --> │  Limpeza + Chunking   │ --> │  Embeddings (e5-base)  │
│  scraper Python  │     │  PyMuPDF / pdfplumber │     │  multilingual-e5-base  │
└─────────────────┘     └──────────────────────┘     └───────────┬───────────┘
                                                                   │
                                                                   v
┌──────────────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│  Resposta ancorada em │ <-- │  LLM (Qwen2.5 7B /   │ <-- │  Base vetorial     │
│  documentos oficiais  │     │  1.5B + QLoRA)        │     │  ChromaDB (cosine) │
└──────────────────────┘     └─────────────────────┘     └───────────────────┘
```

O desenvolvimento foi conduzido em três ambientes computacionais complementares: um notebook pessoal (GPU de 4 GB, prototipagem), uma workstation de laboratório (sem GPU dedicada, testes de implantação) e o Google Colaboratory (Tesla T4, 16 GB VRAM, ajuste fino e validação).

## Pipeline em detalhe

### 1. Coleta de dados

Script em Python que percorre automaticamente o portal público de atos normativos da UFOP (Sistema de Órgãos Colegiados — SOC), requisitando sequencialmente as páginas correspondentes a um intervalo de números de resolução.

- Filtro automático de documentos com o marcador `(REVOGADA)`, para não indexar normas sem validade vigente.
- Nomenclatura padronizada dos arquivos pelo número da resolução, preservando rastreabilidade cronológica.
- Intervalo de 1 segundo entre requisições, como medida de uso responsável do servidor.

**Resultado:** de 5.001 números de resolução verificados, 4.700 PDFs foram efetivamente recuperados e convertidos; após descarte de arquivos corrompidos ou com falhas de extração, restaram **3.700 documentos de alta integridade**.

### 2. Base de conhecimento vetorial (RAG)

- Extração de texto com estratégia de dupla biblioteca: **PyMuPDF** como extração primária, com **pdfplumber** como alternativa para documentos com tabelas ou colunas múltiplas.
- Limpeza de ruídos característicos do sistema de tramitação eletrônica (timestamps, URLs, cabeçalhos repetidos, blocos de assinatura digital).
- Segmentação em blocos (*chunks*) de ~400 palavras, com sobreposição de 60 palavras entre blocos consecutivos, para preservar contexto em trechos de fronteira.
- Geração de embeddings com o modelo multilíngue **`intfloat/multilingual-e5-base`** (prefixos `passage:` na indexação e `query:` na busca).
- Indexação em banco vetorial persistente **ChromaDB**, configurado para similaridade de cosseno, com checkpoint para retomada em caso de desconexão do Colab.

**Resultado:** base de conhecimento com **10.314 segmentos de texto indexados**. Testes exploratórios de recuperação confirmaram que a base retorna trechos tematicamente pertinentes às consultas realizadas.

### 3. Ajuste fino do modelo de linguagem

Antes de definir o modelo de base, duas alternativas de maior porte foram avaliadas e descartadas:

| Modelo | Resultado do teste |
|---|---|
| GPT-OSS 120B (MoE, ~117B params) | Inviável — exige >200 GB de VRAM, incompatível com a infraestrutura disponível |
| GPT-OSS 20B | Tecnicamente viável, mas latência elevada e baixa precisão em português |
| **Qwen2.5 (1.5B / 7B Instruct)** | **Adotado** — melhor equilíbrio entre porte, velocidade e compreensão do português |

O ajuste fino foi realizado por **QLoRA** (adaptação de baixo posto em precisão reduzida): quantização NF4 de 4 bits com quantização dupla, LoRA rank 16 / alpha 32 aplicado às camadas de atenção e projeção (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), dropout 0,05, 3 épocas, lote efetivo de 8 amostras (acumulação de gradiente), otimizador `paged_adamw_8bit`, taxa de aprendizado 2×10⁻⁴ com escalonamento por cosseno.

O conjunto de treinamento foi formatado no padrão de conversação ChatML, com respostas de referência **geradas automaticamente** a partir de elementos identificáveis nos documentos (número da resolução, data, artigos normativos).

> **Bug real diagnosticado:** um dataset usado em uma tentativa anterior (não documentada em detalhe) tinha o campo de resposta (`output`) vazio em 100% dos registros — o que ensinava o modelo a gerar respostas vazias/truncadas. Esse diagnóstico motivou a reconstrução do dataset com geração automática de respostas de referência a partir do próprio texto dos documentos.

### 4. Implantação e restrições de hardware

A implantação local do modelo de 7B parâmetros (quantizado em 4 bits) foi testada em dois ambientes de baixo desempenho, representativos de equipamentos comuns em setores administrativos públicos:

- **Notebook (GTX 1650, 4 GB VRAM):** falha por esgotamento de memória (~24% do carregamento). VRAM mínima estimada: 5,5–6,5 GB.
- **Workstation sem GPU dedicada (CPU-only):** tempo de resposta superior a 30 minutos por consulta, mesmo com otimizações de paralelismo.
- **Google Colab (Tesla T4, 16 GB VRAM):** carregamento completo bem-sucedido, validando a viabilidade técnica da arquitetura RAG combinada ao modelo base.

Como resposta às limitações de VRAM, foi **projetada** (mesclagem do adaptador → conversão para GGUF F16 → quantização Q4_K_M → execução via `llama-cpp-python` com particionamento de camadas entre VRAM e RAM do sistema) uma estratégia de quantização híbrida GPU/CPU. Essa estratégia está documentada e pronta para execução, mas depende de um adaptador LoRA treinado especificamente para o modelo de 7B (ver [Limitações](#limitações-e-lições-aprendidas)).

## Stack técnica

`Python` · `requests` · `PyMuPDF` · `pdfplumber` · `ChromaDB` · `sentence-transformers (multilingual-e5-base)` · `Hugging Face Transformers` · `PEFT (LoRA)` · `bitsandbytes (QLoRA)` · `Qwen2.5-1.5B/7B-Instruct` · `PyTorch` · `Google Colab (Tesla T4)` · `llama.cpp / GGUF` (implantação, planejado)

## Resultados

- Base vetorial funcional com >10 mil segmentos indexados e recuperação semântica validada qualitativamente.
- Adaptador QLoRA treinado com sucesso técnico sobre o Qwen2.5-1.5B-Instruct, porém com sinais de alucinação (respostas parcialmente desalinhadas do contexto recuperado), mesmo operando junto ao RAG.
- O Qwen2.5-7B-Instruct **sem** ajuste fino adicional, combinado à mesma base de RAG, mostrou-se qualitativamente mais consistente com os documentos recuperados do que a variante menor ajustada, sugerindo que, neste estágio, a capacidade do modelo base influencia a fidelidade das respostas tanto quanto (ou mais que) o ajuste fino aplicado.
- Diagnóstico e correção de uma falha crítica de implantação: a primeira versão do script de produção carregava apenas o modelo base, sem nunca aplicar o adaptador LoRA corrigida antes dos testes com terceiros.

## Limitações e lições aprendidas

- **Nenhum adaptador LoRA foi treinado para o Qwen2.5-7B-Instruct**, a principal lacuna em aberto ao final do ciclo de IC. Isso impede tanto a avaliação quantitativa formal (precisão, revocação, F1 na extração de entidades; BLEU e avaliação humana na geração) quanto a validação empírica da estratégia de implantação GGUF.
- Taxa de perda de ~21% entre PDFs convertidos e documentos de alta integridade, ainda não categorizada por tipo de falha.
- Scraper sem mecanismo de nova tentativa (retry) nem verificação de duplicidade.
- Comparação sistemática com/sem ajuste fino, prevista nos objetivos originais do projeto, permanece como trabalho futuro.

## Roadmap

- [ ] Concluir o ajuste fino QLoRA sobre o Qwen2.5-7B-Instruct.
- [ ] Avaliação quantitativa formal (precisão, revocação, F1, BLEU, avaliação humana) comparando configurações com e sem ajuste fino.
- [ ] Validar empiricamente o pipeline de conversão GGUF em hardware de baixo desempenho.
- [ ] Investigar e quantificar as causas da perda de ~21% de documentos na etapa de coleta.
- [ ] Implementar retry e verificação de duplicidade no scraper.
- [ ] Protótipo de chatbot integrado a fluxos de atendimento da comunidade acadêmica.

## Estrutura do repositório

```
.
├── coleta/
│   └── scraper_soc_ufop.py            # Etapa 1 — coleta dos documentos no portal SOC UFOP
├── rag/
│   └── construir_base_conhecimento.py # Etapa 2 — extração, chunking, embeddings e ChromaDB
├── finetuning/
│   └── treino_qlora.py                # Etapa 3 — dataset de treino + QLoRA (Qwen2.5-1.5B)
├── implantacao/
│   ├── ufop_ia.py                     # Assistente conversacional standalone (RAG + Qwen2.5-7B + adapter LoRA)
│   ├── iniciar.bat                    # Atalho de execução no Windows
│   └── README.txt                     # Instruções de instalação e teste em máquina de terceiros
├── docs/                              # Artigo científico e relatório técnico
├── requirements.txt
└── README.md
```

> As pastas `chromadb/` e `adapter/` (dados do banco vetorial e pesos do ajuste fino) não ficam neste repositório, são geradas pela Etapa 2 e Etapa 3, respectivamente, e devem ser colocadas ao lado de `ufop_ia.py` antes de rodar `iniciar.bat` (ver seção abaixo).

### Sobre o `implantacao/ufop_ia.py`

Script standalone que roda o assistente 100% localmente: verifica GPU/CUDA, carrega o `Qwen2.5-7B-Instruct` em 4 bits, aplica o adapter LoRA (se presente em `./adapter`) e responde perguntas com base no contexto recuperado do ChromaDB (`./chromadb`).

Pontos que valem destaque:
- Falha graciosamente com mensagens explicativas (sem GPU, sem VRAM suficiente, pasta `chromadb`/`adapter` ausente ou corrompida), pensado para ser testado por terceiros sem conhecimento técnico, incluindo o orientador.
- Se a pasta `adapter/` não existir, avisa e pergunta se o usuário quer continuar com o modelo base sem ajuste fino, em vez de falhar silenciosamente (correção da falha descrita em [Limitações](#limitações-e-lições-aprendidas): a primeira versão aplicava o base "cru" sem avisar).
- **Atenção:** como nenhum adapter LoRA foi treinado ainda para o `Qwen2.5-7B-Instruct` (só para o 1.5B — ver [Limitações](#limitações-e-lições-aprendidas)), colocar o adaptador `modelo_ufop_adapter_v2` (treinado sobre o 1.5B) em `./adapter` causa erro de incompatibilidade de dimensões ("size mismatch"), capturado e reportado pelo script em vez de travar sem explicação.

## Como reproduzir

Os scripts de `rag/` e `finetuning/` foram escritos como notebooks do Google Colab (marcados por comentários `# %% CÉLULA X`); abra-os no Colab ou em qualquer editor com suporte a células (VS Code, Jupyter) e execute célula por célula.

1. **Coleta** (`coleta/scraper_soc_ufop.py`) — roda localmente, não precisa de GPU:
   ```bash
   pip install requests
   python coleta/scraper_soc_ufop.py
   ```
   Baixa os PDFs para a pasta `arquivos/`. Ajuste `primeiro`/`ultimo` para outro intervalo de resoluções.

2. **Base RAG** (`rag/construir_base_conhecimento.py`) — recomendado no Google Colab:
   - Suba os PDFs coletados (ou o `dataset_ufop.jsonl`, se já tiver um) para uma pasta no Google Drive.
   - Abra o script no Colab, ajuste `PASTA_PDFS` na Célula 2 e execute célula por célula.
   - Gera `chunks.jsonl` e o banco vetorial persistente em `ufop_ia_dados/chromadb`.

3. **Fine-tuning** (`finetuning/treino_qlora.py`) — Google Colab com GPU (T4 ou superior):
   - Ambiente de execução → Alterar tipo → GPU.
   - Execute célula por célula; o adaptador LoRA final é salvo em `ufop_ia_dados/modelo_ufop_adapter_v2`.
   - Leva de 3 a 5 horas; há checkpoint automático no Drive caso o Colab desconecte.

```bash
pip install -r requirements.txt
```

4. **Uso local do assistente** (`implantacao/ufop_ia.py`) — Windows, com GPU NVIDIA (≥6 GB VRAM recomendado):
   - Copie a pasta `chromadb/` gerada na Etapa 2 e (opcionalmente) a pasta do adapter gerada na Etapa 3 para dentro de `implantacao/`.
   - Dê duplo clique em `iniciar.bat` (ou rode `python ufop_ia.py`).
   - Na primeira execução, as dependências e o modelo (~5 GB) são baixados automaticamente.
   - Instruções completas de teste em `implantacao/README.txt`.

## Publicação

Artigo científico submetido à **Revista Educação & Ensino** (Centro Universitário Ateneu — UniAteneu), ISSN 2594-4444:

> LEÃO, T. N. R. G.; SANTOS, T. F. **Adaptação de modelos de linguagem de grande porte para automação de processos administrativos públicos: um sistema de recuperação aumentada por geração aplicado à Universidade Federal de Ouro Preto.** Revista Educação & Ensino, Fortaleza, [em avaliação].

## Agradecimentos

- **Guilherme Augusto Anício Drummond do Nascimento**, pela contribuição decisiva no desenvolvimento do script de coleta de dados.
- **Brendha Alexania Pereira Godinho**, pelo apoio ao longo de todas as etapas do projeto.
- **Prof. Thiago Fontes Santos**, pela orientação.

## Licença

A definir. Os modelos de base utilizados (família Qwen2.5) são distribuídos sob licença aberta pelos respectivos mantenedores.
