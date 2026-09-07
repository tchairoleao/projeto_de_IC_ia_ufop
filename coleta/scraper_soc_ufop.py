# ================================================================
#  IA UFOP — ETAPA 1: CÓDIGO DE COLETÂNEA DE DADOS
# ================================================================
#
# Coleta automatizada de resoluções oficiais publicadas no portal
# público do Sistema de Órgãos Colegiados (SOC) da UFOP.
#
# - Percorre um intervalo de números de resolução.
# - Descarta automaticamente documentos marcados como "(REVOGADA)".
# - Baixa o PDF correspondente, nomeado pelo número da resolução.
# - Aguarda 1 segundo entre requisições (uso responsável do servidor).
# ================================================================

import requests
import re
import time
import os


def requestArquivoIntegra(numero, pastaDestino="arquivos"):
    """
    Realiza a requisição ao portal SOC UFOP, verifica se o documento está ativo
    e baixa o PDF correspondente.
    """
    if not os.path.exists(pastaDestino):
        os.makedirs(pastaDestino)

    urlBase = "https://www.soc.ufop.br/public/"
    url = f"{urlBase}resolucao/mostrar/{numero:010d}"

    try:
        # Requisição à página da resolução
        html = requests.get(url, timeout=10)
    except Exception as e:
        print(f"Erro de conexão no arquivo {numero}: {e}")
        return

    if html.status_code == 200:
        # Filtro de documentos obsoletos: ignora se houver "(REVOGADA)" no texto
        if re.search(r"\(REVOGADA\)", html.text):
            print(f"Arquivo {numero} ignorado: Documento REVOGADO encontrado no enunciado.")
            return

        # Busca pelo link do PDF no código HTML
        pattern = r'href\s*=\s*"/public/files/(\w+\.pdf)'
        m = re.search(pattern, html.text)

        if (m):
            pdfName = m.group(1)
            pdfUrl = urlBase + "files/" + pdfName

            try:
                # Download do arquivo PDF
                pdf = requests.get(pdfUrl, timeout=15)
                if pdf.status_code == 200:
                    print(f"Tamanho do PDF: {len(pdf.content)} bytes")
                    # Nomeia o arquivo com o número da resolução para manter a ordem
                    fileName = f"{pastaDestino}/{numero:010d} - {pdfName}"
                    with open(fileName, "wb") as file:
                        file.write(pdf.content)
                    print(f"PDF salvo: {pdfName}")
                else:
                    print(f"Não foi possível recuperar o PDF para o número {numero}")
            except Exception as e:
                print(f"Erro ao baixar o PDF {numero}: {e}")
        else:
            print(f"Não foi possível encontrar o link do PDF no arquivo {numero}")
    else:
        print(f"Erro {html.status_code} ao acessar a página do arquivo {numero}")


# --- Configurações do Loop de Coleta ---
# Intervalo definido para a colheita dos 5.000 documentos
primeiro = 8137
ultimo = 13137
segundosEntreArquivos = 1

if __name__ == "__main__":
    for i in range(primeiro, ultimo + 1):
        print(f"{'='*60}\n\tProcessando Arquivo {i}")
        requestArquivoIntegra(i, "arquivos")

        # Delay de segurança para evitar bloqueio por excesso de requisições
        time.sleep(segundosEntreArquivos)
