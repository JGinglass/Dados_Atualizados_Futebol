from math import e
import re
from turtle import back
import requests
import cloudscraper
import time
from dotenv import load_dotenv
from firecrawl import Firecrawl
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import os


def _read_html_tables(html_fragment: str):
    """Compat para evitar FutureWarning de passar literal a read_html."""
    return pd.read_html(StringIO(html_fragment))


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Achatamento de MultiIndex e limpeza."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x) for x in tup if str(x) != "NaN"]).strip()
            for tup in df.columns
        ]
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^\w\s/%\-\.]", "", regex=True)
        .str.strip()
    )
    return df
  
  
def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
  """
  Converte automaticamente todas as colunas de um DataFrame
  para tipo numérico (float/int) quando o conteúdo for numérico.
  - Remove vírgulas, %, e espaços.
  - Ignora colunas puramente textuais.
  """

  df_converted = df.copy()

  for col in df_converted.columns:
      # Pula colunas completamente vazias
      if df_converted[col].isna().all():
          continue

      # Tenta converter para número (substituindo vírgulas por pontos etc.)
      df_converted[col] = (
          df_converted[col]
          .astype(str)
          .str.replace(",", ".", regex=False)
          .str.replace("%", "", regex=False)
          .str.replace("nan","", regex=False)
          .str.replace("On matchday squad. but did not play", "", regex=False)
          .str.strip()
      )

      # Converte de fato para numérico se possível
      try:
            df_converted[col] = pd.to_numeric(df_converted[col])
      except ValueError:
          # Se falhar, mantém como string
          print(f"[info] Coluna '{col}' não é numérica, mantendo como string.")
          df_converted[col] = df_converted[col].astype(str)

  return df_converted


def limpa_tabela(df: pd.DataFrame) -> pd.DataFrame:
        # Gera nome dos arquivos de saída, um para cada tabela com o nome da tabela       
        #out_path = os.path.join(out_dir, f"{tabela}.csv")

        # merged aqui é nome da variavel da tabela temporária para tratamento e exportação para arquivo
        merged = df
        
        # Remove coluna que fica com nome estranho ao ler de comentário HTML
        merged.columns = merged.columns.str.replace(r'^Unnamed [0-9]+_level_[0-9]+', '', regex=True).str.strip()
        
        # Pega o nome da primeira coluna que será chave (nome de jogador na maioria dos casos)
        primeira_coluna = merged.columns[0]
        
        # Remove header no meio da tabela e linhas desnecessárias
        merged = merged[~merged[primeira_coluna].astype(str).str.contains("Player|Total|Date", na=False)]
        merged = merged[merged[primeira_coluna].notna()]
        merged = merged[merged[primeira_coluna].astype(str).str.strip() != ""]
        merged = merged[~merged[primeira_coluna].isin(['Total'])]
        
        # Converte o que for numero
        merged = convert_numeric_columns(merged)
        
        # Refaz o indice e atualiza dicionáiro de tabelas
        merged.reset_index(drop=True, inplace=True)
        
        return  merged
      
def carrega_e_limpa_dfs(dirin: str) -> list[pd.DataFrame]:
  lista_dfs = []
  for file in os.listdir(dirin):
    if file.endswith(".csv"):
      df = pd.read_csv(os.path.join(dirin, file))
      df_limpo = limpa_tabela(df)
      lista_dfs.append(df_limpo)
      df_limpo.to_csv(os.path.join(dirin, file), index=False)
  return lista_dfs

def carrega_dados_partidas(url: str)-> list[pd.DataFrame]:
  load_dotenv()
  
  retorno = []

  app = Firecrawl()

  all_competitions_page = app.scrape(url, formats=["html"], include_tags=["table"])
  soup = BeautifulSoup(all_competitions_page.html, "lxml")
  table_tag = soup.find("table", id="stats_standard_combined")

  links_extraidos = {}

  if table_tag:
    # Percorre todas as linhas do corpo da tabela (tbody)
    for row in table_tag.find("tbody").find_all("tr"):
        # Seleciona a célula desejada. 
        # Exemplo: suponha que os links estejam na 1ª ou 2ª coluna
        cells = row.find_all("td")
        cells_player = row.find_all("th")
        tamanho = len(cells)
        if len(cells) > 0:
            # Busca qualquer link dentro dessa célula
            jogador = cells_player[0].get_text()
            link_tag = cells[tamanho-1].find("a", href=True)
            if link_tag:
                # Captura o href absoluto
                link = link_tag["href"]
                links_extraidos[jogador] = link

  for jogador in links_extraidos.keys():
    print(f"Acessando {jogador} = link: {links_extraidos[jogador]}")
    doc = app.scrape(links_extraidos[jogador], formats=["html"], include_tags=["table"])  
    soup = BeautifulSoup(doc.html, "lxml")
    table_tag = soup.find("table", id="matchlogs_all")
    if table_tag:
      df = _read_html_tables(str(table_tag))[0]
      df = _flatten_cols(df)
      df = limpa_tabela(df)
      df.to_csv(f"out/{jogador}.csv", index=False)
      retorno.append(df)
      print(df)
      time.sleep(2)
  return retorno

def junta_csvs_em_um(out_dir: str, arquivo_saida: str):
  dfs = []
  for file in os.listdir(out_dir):
    print(f"Processando arquivo: {file}")
    if file.endswith(".csv"):
      print(f"Lendo arquivo: {file}")
      df = pd.read_csv(os.path.join(out_dir, file))
      df.insert(0, "Player", file.replace(".csv",""))
      print(df.head(3))
      dfs.append(df)
  df_combinado = pd.concat(dfs, ignore_index=True)
  df_combinado.to_csv(os.path.join(out_dir, arquivo_saida), index=False)

if __name__ == "__main__":
  #dfs = carrega_dados_partidas("https://fbref.com/en/squads/639950ae/2025/all_comps/Flamengo-Stats-All-Competitions")
  #dfs = carrega_e_limpa_dfs("out/")
  junta_csvs_em_um("out/", "stats/juncaojogadores.csv")
  #for df in dfs:
  #print(df)


