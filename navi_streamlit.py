import pandas as pd
import sqlalchemy 
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import esg_score_over_time

#%matplotlib inline


engine = create_engine("cockroachdb://quant:gz43t_wo4zesYneu@free-tier.gcp-us-central1.cockroachlabs.cloud:26257/grown-iguana-3784.defaultdb?sslmode=require")

st.title("NAVI Hackathon")
st.sidebar.title("Opções")


#option = st.sidebar.selectbox(
#     'Que opção você quer?',
#     ('Opt 1', 'Opt 2', 'Opt 3'))

#st.sidebar.write('Selecionado:', option)




#CARREGANDO A CARTEIRA

carteira_csv = st.file_uploader("Faça o upload de sua carteira", type=["csv", "txt"])

start_button = st.button("Calcular",help="Clique aqui para gerar informações sobre sua carteira")

if carteira_csv is not None and start_button==True:
    
    df_weighted = esg_score_over_time.esg_over_time(carteira_csv,engine)
    
    fig = px.line(df_weighted, x='assessment_year',y='esg_score')

    st.header("esg score over time")
    st.plotly_chart(fig)