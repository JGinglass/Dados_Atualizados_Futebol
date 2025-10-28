from dotenv import load_dotenv
import streamlit as st
from datasetFut import perguntaagente, carregardadosfbref, criatabelacombinada_jogadores, gerarcsvtabelas
import pandas as pd

@st.cache_data
def carregar_dados():
    print("Carregando dados...")
    df_jogadores = pd.read_csv("out/stats_combinada.csv")
    df_partidas = pd.read_csv("out/matchlogs_for.csv")
    return df_jogadores, df_partidas
  
  
load_dotenv('.env')

st.set_page_config(page_icon="⚽",layout="wide", page_title="Análise de Dados de Futebol")
st.write("Bem-vindo ao aplicativo de análise de dados de futebol!")

with st.sidebar:
    st.header("Configurações")
    mode = st.radio("Escolha o modo de análise:", ("Jogadores", "Partidas"))
    st.divider()
    

    
# -------------------------------
# HISTÓRICO DE CHAT
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "tabela_jogadores" not in st.session_state:
    st.session_state.tabela_jogadores,st.session_state.tabela_partidas = carregar_dados()

# Mostrar mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
 
# Entrada do usuário (modo chat)
pergunta = st.chat_input("Digite sua pergunta sobre futebol ⚽")

if pergunta:
    # Exibe a pergunta
    st.chat_message("user").markdown(pergunta)
    st.session_state.messages.append({"role": "user", "content": pergunta})
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("Analisando seus dados..."):
                if mode == "Partidas":
                  resposta = perguntaagente(st.session_state.tabela_partidas, pergunta)
                else:
                  resposta = perguntaagente(st.session_state.tabela_jogadores, pergunta)
                st.chat_message("assistant").markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
               
        except Exception as e:
            st.error(f"❌ Erro ao processar: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Erro: {e}"})

    

