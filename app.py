import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from pandas.api.types import infer_dtype
import os

# Přímé vložení API klíče (pozor, jen pro testovací účely)
llm = OpenAI(api_token="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")
pandas_ai = PandasAI(llm)

st.set_page_config(page_title="AI Metadata Agent v2", layout="wide")
st.title("🤖 Pokročilý AI agent pro správu metadat")
st.markdown("Tato verze využívá jazykový model pro inteligentní analýzu dat.")

uploaded_file = st.file_uploader("Nahraj CSV soubor", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Náhled dat")
    st.dataframe(df.head())

    st.subheader("🧪 Základní kontrola kvality dat")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Počet prázdných hodnot ve sloupcích:**")
        st.write(df.isnull().sum())
    with col2:
        st.markdown("**Detekovaný typ hodnot (Pandas):**")
        st.write(df.apply(infer_dtype))

    st.subheader("🧠 AI analýza a návrh metadat")
    with st.spinner("Analyzuji data pomocí jazykového modelu..."):
        try:
            response = pandas_ai.run(df, prompt="Popiš tabulku, navrhni typy sloupců a vyznač zajímavé anomálie. Vypiš také, o jaký typ dat pravděpodobně jde.")
            st.markdown("### Shrnutí od AI")
            st.write(response)
        except Exception as e:
            st.error(f"Došlo k chybě při AI analýze: {e}")

    st.subheader("💬 Polož vlastní dotaz na data")
    user_query = st.text_input("Zeptej se na cokoli ohledně dat:")
    if user_query:
        with st.spinner("Zpracovávám dotaz pomocí AI..."):
            try:
                answer = pandas_ai.run(df, prompt=user_query)
                st.markdown("**Odpověď:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Chyba při zpracování dotazu: {e}")

    st.success("Analýza dokončena. Můžeš zkoušet další dotazy nebo nahrát jiný soubor.")
