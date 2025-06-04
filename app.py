import streamlit as st
import pandas as pd
import os
from extractor import extrair_texto
from nlp_utils import (
    criar_pipeline_com_skills,
    processar_texto_com_pipeline,
    calcular_similaridade,
    agrupar_aderencia,
    gerar_nuvem_palavras,
    plotar_nuvem
)

st.set_page_config(page_title="Analisador de Currículos", layout="wide", initial_sidebar_state="collapsed")

# Oculta o menu de Streamlit (hambúrguer superior direito)
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.expander("ℹ️ Como funciona a aderência?"):
    st.markdown("""
    **Aderência Semântica** é a compatibilidade entre o currículo e a vaga, com base no conteúdo textual.

    - **Alta**: 80% ou mais
    - **Média**: entre 50% e 79%
    - **Baixa**: abaixo de 50%
    """)

st.markdown("## 🤖 Analisador de Currículos com IA")
st.markdown("Envie currículos em PDF e compare com a vaga para obter análise de aderência, informações de contato e nuvem de palavras-chave.")

# Layout: Descrição da vaga e habilidades lado a lado
col1, col2 = st.columns([2, 1])

with col1:
    with st.expander("📌 Descrição da vaga", expanded=True):
        vaga = st.text_area("Cole a descrição completa da vaga aqui:", height=200)

with col2:
    with st.expander("🧠 Habilidades desejadas"):
        skills_input = st.text_input(
            "Digite as habilidades separadas por vírgula",
            value="Python, Excel, SQL, Power BI, Machine Learning"
        )

skills_base = [s.strip() for s in skills_input.split(",") if s.strip()]
nlp = criar_pipeline_com_skills(skills_base)

st.markdown("### 📁 Upload de Currículos (PDF)")
arquivos = st.file_uploader("Envie os arquivos PDF dos currículos:", type="pdf", accept_multiple_files=True)

if st.button("🔍 Iniciar Análise"):
    if not vaga or not arquivos:
        st.warning("⚠️ Por favor, preencha a descrição da vaga e envie pelo menos um currículo.")
    else:
        st.info("🔄 Processando currículos, aguarde...")

        dados = []
        textos_cvs = []

        progress = st.progress(0)
        for idx, arq in enumerate(arquivos):
            texto_cv = extrair_texto(arq)
            textos_cvs.append(texto_cv.lower())

            similaridade = calcular_similaridade(vaga, texto_cv)
            info_extraida = processar_texto_com_pipeline(texto_cv, nlp)

            dados.append({
                "Nome do Arquivo": arq.name,
                "Nome": info_extraida.get("nome", ""),
                "E-mail": info_extraida.get("email", ""),
                "Telefone": info_extraida.get("telefone", ""),
                "Aderência Semântica (%)": round(similaridade * 100, 2),
                "Skills": ", ".join(info_extraida.get("skills", []))
            })

            progress.progress((idx + 1) / len(arquivos))

        df = pd.DataFrame(dados)
        df = df.sort_values(by="Aderência Semântica (%)", ascending=False).reset_index(drop=True)
        df["Nível de Aderência"] = df["Aderência Semântica (%)"].apply(agrupar_aderencia)

        st.success("✅ Análise concluída com sucesso!")

        with st.expander("📊 Resultado da Análise", expanded=True):
            st.dataframe(df, use_container_width=True)

        with st.expander("☁️ Nuvem de Palavras dos Currículos"):
            wc = gerar_nuvem_palavras(textos_cvs)
            fig = plotar_nuvem(wc)
            st.pyplot(fig)

        os.makedirs("export", exist_ok=True)
        nome_arquivo = "export/ranking_curriculos.xlsx"
        df.to_excel(nome_arquivo, index=False)

        st.download_button(
            label="📥 Baixar Ranking em Excel",
            data=open(nome_arquivo, "rb"),
            file_name="ranking_curriculos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
