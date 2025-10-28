from heapq import merge
import os
import time
import re
import warnings
from io import StringIO
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import pandas as pd
from bs4 import BeautifulSoup, Comment
import cloudscraper

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# =========================================
# CONFIG
# =========================================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.6478.114 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://google.com",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

POSSIBLE_KEYS = ["Player"]  # nomes que o FBref usa

# =========================================
# HELPERS
# =========================================
def get_html(url: str, max_retries: int = 3, backoff: float = 1.5) -> str:
    """Baixa HTML usando cloudscraper (evita 403)."""
    scraper = cloudscraper.create_scraper()
    for i in range(max_retries):
        r = scraper.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.text
        elif r.status_code == 403:
            print(f"[aviso] 403, aguardando {backoff*(i+1):.1f}s e tentando de novo…")
        time.sleep(backoff * (i + 1))
    r.raise_for_status()

def get_html_from_file(file_path: str) -> str:
    """Lê HTML de um arquivo local (para testes)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read() 

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

def _read_html_tables(html_fragment: str):
    """Compat para evitar FutureWarning de passar literal a read_html."""
    return pd.read_html(StringIO(html_fragment))

def fbref_read_table(url: str, table_id: str) -> pd.DataFrame:
    """Lê tabela (mesmo se estiver em comentário HTML)."""
    
    
    html = ""
    if (url.startswith("file://")):
        file_path = url[7:]  # Remove 'file://' prefix
        html = get_html_from_file(file_path)
    else:
        html = get_html(url)
    soup = BeautifulSoup(html, "lxml")

    # 1) direta
    table_tag = soup.find("table", id=table_id)
    if table_tag:
        df = _read_html_tables(str(table_tag))[0]
        return _flatten_cols(df)

    # 2) comentários
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    target_html = None
    pattern = re.compile(rf'<table[^>]*id=["\']{re.escape(table_id)}["\']', re.I)

    for c in comments:
        if pattern.search(c):
            target_html = c.replace("-->", "")
            break

    if target_html is None:
        raise ValueError(f"Tabela com id='{table_id}' não encontrada.")

    if not target_html.strip().endswith("</table>"):
        target_html += "</table>"

    tables = _read_html_tables(target_html)
    df = max(tables, key=lambda d: (d.shape[0], d.shape[1]))
    return _flatten_cols(df)

def fbref_extract_team_performance(
    url: str,
    tables=(
        "matchlogs_for",
        "stats_standard_combined",
        "stats_shooting_combined",
        "stats_passing_combined",
        "stats_defense_combined",
        "stats_possession_combined",
        "stats_misc_combined"
    ),
) -> dict:
    out = {}
    for t in tables:
        try:
            df = fbref_read_table(url, t)
            out[t] = df
            print(f"[ok] '{t}' -> {df.shape[0]} linhas, {df.shape[1]} colunas")
        except Exception as e:
            out[t] = None
            print(f"[aviso] falhou '{t}': {e}")
        #time.sleep(1.1)
    return out

def add_context_columns(df: pd.DataFrame, competition=None, season=None) -> pd.DataFrame:
    if df is None:
        return None
    df = df.copy()
    if competition:
        df["Competition"] = competition
    if season:
        df["Season"] = season
    return df

def _detect_join_key(df: pd.DataFrame):
    for k in POSSIBLE_KEYS:
        if k in df.columns:
            return k
    return None

def safe_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Merge robusto que detecta e alinha chaves ('Squad'/'Team'/'Club'/'Equipe')."""
    if left is None:
        return right
    if right is None:
        return left
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        return left

    # Mantém só colunas novas
    common_cols = [c for c in right.columns if c in left.columns and c != "Player"]
    right = right.drop(columns=common_cols)
    return left.merge(right, on="Player", how="left")


import pandas as pd

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


# === 5. Função mestre: escolhe o agente conforme a pergunta ===
def ask_agent(query: str):
    """
    Analisa o texto da pergunta e escolhe o DataFrame certo.
    """
    q = query.lower()
    if "jogador" in q or "gols" in q or "assistência" in q:
        chosen = "jogadores"
    elif "time" in q or "clube" in q:
        chosen = "times"
    elif "partida" in q or "jogo" in q:
        chosen = "partidas"
    else:
        chosen = "jogadores"  # padrão
    
    print(f"\n📊 Usando o DataFrame: {chosen}")
    response = agents[chosen].invoke(query)
    return response["output"]

def limpardadostabela(tabelas)->dict:
    for tabela in tabelas.keys():
        # Gera nome dos arquivos de saída, um para cada tabela com o nome da tabela       
        #out_path = os.path.join(out_dir, f"{tabela}.csv")

        # merged aqui é nome da variavel da tabela temporária para tratamento e exportação para arquivo
        merged = tabelas[tabela]
        
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
        
        tabelas[tabela] = merged
        
    return tabelas

def criatabelacombinada_jogadores(tabelas)->pd.DataFrame:
    # merged agora recebe o tabelao unindo todas as tabelas de estatísticas de jogadores
    merged = safe_merge(tabelas["stats_standard_combined"], tabelas["stats_shooting_combined"])
    merged = safe_merge(merged, tabelas["stats_passing_combined"])
    merged = safe_merge(merged, tabelas["stats_defense_combined"])
    merged = safe_merge(merged, tabelas["stats_possession_combined"])
    merged = safe_merge(merged, tabelas["stats_misc_combined"])
    
    return merged

def gerarcsvtabelas(tabelas, out_dir)->None:
    for tabela in tabelas.keys():
        # Gera nome dos arquivos de saída, um para cada tabela com o nome da tabela       
        out_path = os.path.join(out_dir, f"{tabela}.csv")
        
        # Salva a tabela tratada em CSV
        tabelas[tabela].to_csv(out_path, index=False)
        print(f"[ok] Salvou tabela '{tabela}' em: {out_path}")
        
        
def carregardadosfbref(url_pl: str)->dict:
  
    print(f"[info] cwd: {os.getcwd()}")
    tables = fbref_extract_team_performance(url_pl)
    
    # Garante diretório de saída e salva com caminho absoluto
    out_dir = os.path.join(os.getcwd(), "out")
    os.makedirs(out_dir, exist_ok=True)
        
    # Limpa dados das tabelas
    tabelas_limpa = limpardadostabela(tables)
    
    # Salva tabelas limpas em CSV
    gerarcsvtabelas(tabelas_limpa, out_dir)
    
    return tabelas_limpa

def perguntaagente(pd_dataframe: pd.DataFrame, query: str)->str:
    load_dotenv('.env')
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Cria agente para o DataFrame fornecido
    agente = create_pandas_dataframe_agent(llm, pd_dataframe, verbose=True, allow_dangerous_code=True,handle_parsing_errors=True)
    
    # Invoca o agente com a pergunta
    response = agente.invoke(query)
    
    return response["output"]

# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    #url_pl = "https://fbref.com/en/comps/9/Premier-League-Stats"
    #url_pl = "https://fbref.com/en/squads/639950ae/Flamengo-Stats"
    #url_pl = "file://data/Flamengo Stats, Série A _ FBref.com.html"
    url_pl = "file://data/Flamengo Stats, All Competitions _ FBref.com.html"
    
    tables = carregardadosfbref(url_pl)
    
    out_dir = os.path.join(os.getcwd(), "out") 
    
    gerarcsvtabelas(tables, out_dir)
           
    # Dataframe com info das partidas
    df_partida = tables["matchlogs_for"]
    
    # Dataframe combinado com info dos jogadores
    df_jogadores = criatabelacombinada_jogadores(tables)
  
    query = "Qual jogador tem mais desarmes com ganhos? Quantos desarmes desse tipo ele fez?"
    print (perguntaagente(df_jogadores, query))
    
    query = "Qual foi a partida com melhor resultado?"
    print (perguntaagente(df_partida, query))