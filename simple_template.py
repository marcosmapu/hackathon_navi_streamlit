import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

st.title("NAVI Hackathon")
st.sidebar.title("Opções")

carteira_csv = st.file_uploader("Faça o upload de sua carteira", type=["csv", "txt"])

option = st.sidebar.selectbox(
     'Que opção você quer?',
     ('Opt 1', 'Opt 2', 'Opt 3'))

st.sidebar.write('Selecionado:', option)

if carteira_csv is not None:
    carteira = pd.read_csv(carteira_csv)
    st.write(carteira.head(10))

df = pd.DataFrame(np.random.randn(30,50), columns=(('coluna %d' %i for i in range(1,51))))

st.write(df)
simple_chart = px.line(df,x=['%d' % i for i in range(30)],y='coluna 3')
st.plotly_chart(simple_chart, use_container_width=True)

a= pd.DataFrame(dict(
    x = [1, 3, 2, 4],
    y = [1, 2, 3, 4]
))
st.write(a)