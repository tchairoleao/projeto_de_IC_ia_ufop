# Assistente IA UFOP — Instruções de Teste

Este programa é um assistente que responde perguntas sobre resoluções, editais e normas da UFOP, com base em ~5 mil documentos oficiais indexados. Ele roda 100% localmente no computador (nenhum dado é enviado para a internet, exceto o download inicial do modelo).

## Requisitos

- **Placa de vídeo NVIDIA** com pelo menos **6GB de VRAM** (recomendado)
- Driver NVIDIA atualizado
- Python 3.10 ou superior instalado ([python.org](https://www.python.org/downloads/))
- ~15GB de espaço livre em disco (para o download do modelo, na primeira execução)
- Conexão com a internet (apenas na primeira execução, para baixar o modelo)

## Como rodar

1. Extraia a pasta `Assistente_UFOP` em qualquer lugar do computador
2. Dê duplo clique em `iniciar.bat`
3. Na primeira execução, o programa vai instalar automaticamente as bibliotecas necessárias e baixar o modelo de linguagem (~5GB) — isso pode levar de 10 a 30 minutos, dependendo da internet
4. Depois de carregado, basta digitar perguntas sobre normas da UFOP no terminal (por exemplo: *"Quais são os requisitos para participar do PIBIC?"*)
5. Digite `sair` para encerrar

## Se der erro de memória (VRAM insuficiente)

O programa vai avisar claramente se a placa de vídeo não tiver memória suficiente para carregar o modelo. Isso é um resultado de teste válido — por favor me avise se acontecer, com a mensagem de erro exibida, pois essa informação já ajuda a documentar os requisitos mínimos de hardware do projeto.

## O que anotar durante o teste

Para eu conseguir documentar os resultados corretamente, se possível anote:
- Modelo da placa de vídeo e quantidade de VRAM
- Se o programa carregou com sucesso ou deu erro
- Tempo aproximado de resposta por pergunta
- Qualidade das respostas (fazem sentido? estão corretas?)
