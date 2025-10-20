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

POSSIBLE_KEYS = ["Squad", "Team", "Club", "Equipe"]  # nomes que o FBref usa

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
        "stats_standard_24",
        "stats_shooting_24",
        "stats_passing_24",
        "stats_defense_24",
        "stats_possession_24",
        "stats_misc_24"
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

    lk = _detect_join_key(left)
    rk = _detect_join_key(right)

    # Se nenhuma chave, não mergeia
    if lk is None and rk is None:
        print("[aviso] nenhum campo de chave encontrado; mantendo o da esquerda.")
        return left

    # Renomeia ambos para 'Squad' e junta
    if lk and lk != "Squad":
        left = left.rename(columns={lk: "Squad"})
    if rk and rk != "Squad":
        right = right.rename(columns={rk: "Squad"})

    if "Squad" not in left.columns or "Squad" not in right.columns:
        print("[aviso] ainda sem 'Squad' em ambos; mantendo o da esquerda.")
        return left

    return left.merge(right, on="Squad", how="left")

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


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    #url_pl = "https://fbref.com/en/comps/9/Premier-League-Stats"
    #url_pl = "https://fbref.com/en/squads/639950ae/Flamengo-Stats"
    url_pl = "file:///home/jpginglass/Documentos/projetosvscode/projstreamtest/Flamengo Stats, Série A _ FBref.com.html"
    
    load_dotenv('.env')
  
    print(f"[info] cwd: {os.getcwd()}")
    tables = fbref_extract_team_performance(url_pl)

    std = add_context_columns(
        tables["stats_standard_24"], competition="Serie A"
    )
    
    table_partida = tables["matchlogs_for"]
    table_shooting = tables["stats_shooting_24"]
    table_passing = tables["stats_passing_24"]
    table_defense = tables["stats_defense_24"]
    table_possession = tables["stats_possession_24"]
    table_misc = tables["stats_misc_24"]
    
    print(table_partida.head(20))
    #merged = safe_merge(std, tables["matchlogs_for"])
    #merged = safe_merge(merged, tables["stats_shooting_24"])
    #merged = safe_merge(merged, tables["stats_passing_24"])
    #merged = safe_merge(merged, tables["stats_defense_24"])
    #merged = safe_merge(merged, tables["stats_possession_24"])
    #merged = safe_merge(merged, tables["stats_misc_24"])
    
    for tabela in tables.keys():
        
        # Garante diretório de saída e salva com caminho absoluto
        out_dir = os.path.join(os.getcwd(), "out")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{tabela}.csv")

        merged = tables[tabela]
        merged.columns = merged.columns.str.replace(r'^Unnamed [0-9]+_level_[0-9]+', '', regex=True).str.strip()
        primeira_coluna = merged.columns[0]
        merged = merged[~merged[primeira_coluna].astype(str).str.contains("Player|Total|Date", na=False)]
        merged = merged[merged[primeira_coluna].notna()]
        merged = merged[merged[primeira_coluna].astype(str).str.strip() != ""]
        merged = merged[~merged[primeira_coluna].isin(['Total'])]
        merged.reset_index(drop=True, inplace=True)
        tables[tabela] = merged
        
        if isinstance(merged, pd.DataFrame) and not merged.empty:
            merged.to_csv(out_path, index=False)
            print(f"\n✅ Arquivo salvo em: {out_path}")
            print(f"Linhas: {len(merged)}, Colunas: {merged.shape[1]}")
            print(f"Colunas-chave detectadas: {', '.join([c for c in merged.columns if c in POSSIBLE_KEYS or c=='Squad'])}")
            print(merged.head(3))
            
        else:
            print("\n⚠️ Nada para salvar (DataFrame vazio). Verifique as mensagens de [aviso] acima.")
            
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # === 4. Crie um agente para cada DataFrame ===
    agents = {name: create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
            for name, df in tables.items()}

    #query = "Converta as coluna GF e GA para numerico e faça a diferença. Verifique a maior e a menor diferenca e indique quais foram as partidas e oponente. Diga também qual foi a média de GA quando o opentente (coluna Opponent) usou a formação 4-2-3-1 (coluna 'Opp Formation')"
    #response = agents["matchlogs_for"].invoke(query)
    #print (response["output"])

    query = "Considerando que o Oponente vai jogar no 3-4-3, qual a média de gols do Flamengo (GF) esperada para esse jogo?"
    #query = "Qual é a formação do oponente (coluna 'Opp Formation') em que o Flamengo tem a maior média de gols? QUal é a média de gols do oponente nessa formação?"
    response = agents["matchlogs_for"].invoke(query)
    print (response["output"])

    #query = "O numero de gol está na coluna 'Standard Gls'. Depois de converter a coluna para numérico, responda quem foi o maior goleador?"
    #response = agents["stats_shooting_24"].invoke(query)
    #print (response["output"])