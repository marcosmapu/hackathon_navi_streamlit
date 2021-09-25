import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objs as go

st.title("NAVI Hackathon")
st.sidebar.title("Opções")

carteira_csv = st.file_uploader("Faça o upload de sua carteira", type=["csv", "txt"])

option = st.sidebar.selectbox(
     'Que opção você quer?',
     ('Opt 1', 'Opt 2', 'Opt 3'))

st.sidebar.write('Selecionado:', option)

if carteira_csv is not None:
    carteira = pd.read_csv(carteira_csv)