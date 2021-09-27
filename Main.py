from os import write
import pandas as pd
import sqlalchemy 
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import esg_score_over_time
import yfinance as yf
import scipy.optimize as so
import os
import sys

#%matplotlib inline


engine = create_engine("cockroachdb://quant:gz43t_wo4zesYneu@free-tier.gcp-us-central1.cockroachlabs.cloud:26257/grown-iguana-3784.defaultdb?sslmode=require")

st.title("NAVI Hackathon")
st.sidebar.title("Bem Vindo ao Quantesg!")
st.sidebar.write("Uma plataforma de análise quantitativas focada em ESG.")


#option = st.sidebar.selectbox(
#     'Que opção você quer?',
#     ('Opt 1', 'Opt 2', 'Opt 3'))

#st.sidebar.write('Selecionado:', option)




#CARREGANDO A CARTEIRA

carteira_csv = st.file_uploader("Faça o upload de sua carteira", type=["csv", "txt"])

start_button = st.button("Calcular",help="Clique aqui para gerar informações sobre sua carteira")

if carteira_csv is not None and start_button==True:
    
    #####
    ##LOADING DATA
    df = pd.read_sql_query("""SELECT comp.ticker,
                                            comp.company_name,
                                            comp.industry,
                                            esg.score_value,
                                            esg.score_weight,
                                            esg.aspect,
                                            esg.assessment_year
                                            FROM COMPANIES_BR comp
                                            LEFT JOIN esg_scores_history_br esg
                                                ON comp.company_id = esg.company_id
                                                AND esg.parent_aspect = 'S&P Global ESG Score'""", engine)
    
    carteira = pd.read_csv(carteira_csv)
    Inicio = "2020-01-01"
    Fim = ''

    MinConcentration = 0.03 #Alocação mínima por ativo
    MaxConcentration = 0.1 #Alocação máxima por ativo

    #Alocs = portfolio['port_weight'].to_list()
    #Assets = ['MGLU3', 'PETR4', 'PRIO3', 'RDOR3', 'ITUB4', 'BBDC4', 'CSNA3', 'EZTC3', 'VALE3', 'SUZB3', 'ITSA4', 'LCAM3','BBAS3','LAME4','ABEV3']
    Assets = carteira['ticker'].to_list()

    Acoes = Assets.copy()
    Alocs = carteira['port_weight'].to_list()
    #Alocs = [0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.04,0.04,0.04,0.04,0.04]

    for i,acao in enumerate(Acoes):
        Acoes[i] += ".SA"

        RiskFree = 0.05


    #Precos
    Precos = yf.download(Acoes, Inicio)['Adj Close'].fillna(method='ffill')
    Retornos = np.log(Precos / Precos.shift(1)).dropna(axis=0)

    # Calculando retornos da RiskFree
    RiskFree = np.log(1 + RiskFree)
    DailyRiskFree = RiskFree/252

    # Transforma os pesos do portfólio em porcentagens:
    weightsPort = np.array(Alocs)

    # Calcula os retornos esperados para o portfólio atual
    retPort = np.sum(Retornos.mean()*weightsPort*252)

    # Calcula a volatilidade do portfólio atual:
    volPort = np.sqrt(np.dot(weightsPort.T, np.dot(Retornos.cov()*252, weightsPort)))

    # Calcula o Sharpe do portfólio atual:
    sharpePort = (retPort - RiskFree)/volPort

    retorno_cumulativo_carteira = np.cumprod(np.sum(Retornos*weightsPort,axis=1)+1)-1
    # retorno_cumulativo_carteira.rename_
    # st.write(retorno_cumulativo_carteira.columns)
    fig_yield = go.Figure()
    fig_yield.add_trace(go.Scatter(x=retorno_cumulativo_carteira.index[1:], y=retorno_cumulativo_carteira, name='Atual',line = dict(width=1)))
    #st.plotly_chart(fig_yield)

    # retorno_cumulativo_carteira.plot()
    
    # np.cumprod(np(Retornos()*weightsPort[1:] + 1)-1

    # Retorna as ações e os pesos do portfólio atual:

    #Stocks = portfolio['ticker'].to_list()
    Stocks = Assets

    data = {'ticker': Stocks, 'weight': weightsPort}
    #print('\nCurrent portfolio:\n')
    #print(pd.DataFrame(data))

    #Retorna o retorno esperado, volatilidade e Sharpe Ratio do portfólio atual
    l = [retPort, sharpePort]
    m = ['Exp Return', 'Sharpe Ratio']
    data = {'Data': m, 'Numbers': l}
    #print()
    #print(pd.DataFrame(data))

    #Incluir notas de ESG da carteira atual na tabela com ticker e sharpe

    years = df['assessment_year'].unique()
    tickers = df['ticker'].unique()
    dfs = []
    for year in years:
        for ticker in tickers:
            df_year = df[(df['assessment_year'] == year) & (df['ticker'] == ticker)]
            df_year['single_esg_score'] = df_year['score_value']*df_year['score_weight']
            df_year['esg_score'] = df_year['single_esg_score'].sum()/df_year['score_weight'].sum()
            df_year = df_year.drop(['score_weight','score_value','single_esg_score','aspect'], axis = 1)
            df_year = df_year.iloc[:1,:]
            dfs.append(df_year)

    df_score = pd.concat(dfs)
    df_score = df_score[df_score['assessment_year']==2020]
    ESGscore = df_score[['ticker','esg_score']]

    dff = pd.DataFrame(Assets, columns =['ticker'])
    ESGscore = pd.merge(ESGscore,dff,on='ticker')

    
    # Calcula o retorno esperado, volatilidade e Sharpe dado uma matriz de pesos:
    exprets = Retornos.mean() * 252
    esg_score = ESGscore['esg_score']
    
    def get_ret_vol_sr(weights):
        weights = np.array(weights)
        esg = np.sum(esg_score*weights)
        ret = np.sum(exprets*weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(Retornos.cov()*252, weights)))
        # sr = ((ret - RiskFree)/vol)*esg
        sr = ((ret - RiskFree)/vol)+esg
        return np.array([ret, vol, sr])

    # Retorna 0 se a soma dos pesos for 1:
    def check_sum(weights):
        return np.sum(weights)-1

    # Retorna a volatilidade da função get_ret_vol_sr:
    def minimize_volatility(weights):
        return get_ret_vol_sr(weights)[1]


    #Padrões
    bottomBorder = 0
    topBorder = 0.5

    # limitações:
    cons = ({'type': 'eq', 'fun': check_sum})

    # Limite de concentração:
    bounds = []

    # Valores que o modelo começará testando:
    init_guess = []

    # Preenche as duas variáveis acima:
    for stock in Stocks:
        bounds.append((MinConcentration, MaxConcentration))
        init_guess.append(1/len(Stocks))

    bounds = tuple(bounds)

    ############################################################################################
    # Calcula níveis de retorno esperado:
    frontier_y = np.linspace(bottomBorder, topBorder, 10)


    frontier_x = []

    # Para cada nível de retorno, minimiza a volatilidade e coloca na lista 'frontier_x':
    for possible_return in frontier_y:
        cons = ({'type': 'eq', 'fun': check_sum},
                {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})

        result = so.minimize(minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        frontier_x.append(result['fun'])
    #    print(possible_return, result['fun'])

    # Cria um dicionário com a 'frontier_x' e a 'frontier_y' e transforma em dataframe:
    data = {'Volatility': frontier_x, 'Expected Returns': frontier_y}
    FEficiente = pd.DataFrame(data)

    #Cria uma coluna com o Sharpe de cada ponto:
    FEficiente['Sharpe Ratio'] = (FEficiente['Expected Returns'] - RiskFree) / FEficiente['Volatility']


    # Calcula o portfólio de maior Sharpe
    print('\nMax Sharpe:\n')
    cons = ({'type': 'eq', 'fun': check_sum},
            {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - FEficiente['Expected Returns'][FEficiente['Sharpe Ratio'].argmax()]})

    result = so.minimize(minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    data = {'ticker': Stocks, 'port_weight': np.round(result.x, 2)}
    
    data = pd.DataFrame(data)
    
    l = list(get_ret_vol_sr(result.x))
    m = ['Exp Return', 'Volatility', 'Shrp Ratio']
    data_shpe = {'Data': m, 'Numbers': l}

    

    MaxShpe = pd.DataFrame(data_shpe)

    retorno_cumulativo_carteira = np.cumprod(np.sum(Retornos*result.x,axis=1)+1)-1
    fig_yield.add_trace(go.Scatter(x=retorno_cumulativo_carteira.index[1:], y=retorno_cumulativo_carteira, name='Otimizado',line = dict(width=1)))
    st.header("Retorno")
    st.plotly_chart(fig_yield)
    st.subheader("Composição atual:")
    st.write(carteira)
    st.subheader("Composição Otimizada:")
    st.write(data)

    ###################
      # YIELD CHART ########################################

    #print(MaxShpe)
    #plt.scatter(MaxShpe.iloc[1]['Numbers'], MaxShpe.iloc[0]['Numbers'], c='darkorange', label='Maximum Sharpe Ratio')

    #plt.plot(frontier_x, frontier_y, 'dimgray', linewidth=2)]


    #####################################################

    df_weighted = esg_score_over_time.esg_over_time(carteira,engine)
        
    fig = px.line(df_weighted, x='assessment_year',y='esg_score')

    st.header("Nota ESG ponderada ao longo do tempo")
    st.plotly_chart(fig)

    #####
    # portfolio = pd.read_csv('IBOVDia_27-09-21.csv',skiprows=1,encoding = "ISO-8859-1", sep=';')
    # portfolio.reset_index(inplace=True)
    # cols = portfolio.columns
    # cols = cols[1:]
    # portfolio = portfolio.iloc[:,:-1]
    # portfolio.columns = cols
    # portfolio = portfolio.drop(['Ação','Tipo','Qtde. Teórica'], axis = 1)
    # portfolio = portfolio.iloc[:-2,:]
    # portfolio.rename(columns={'Código':'ticker','Part. (%)':'port_weight'},inplace=True)
    # portfolio['port_weight'] = portfolio.apply(lambda x: float(x['port_weight'].replace(',','.'))/100,axis=1)
    portfolio = carteira.copy()

    Inicio = "2020-01-01"
    Fim = ''

    Acoes = portfolio['ticker'].to_list()
    for i,acao in enumerate(Acoes):
        Acoes[i] += ".SA"
    Alocs = portfolio['port_weight'].to_list()

    Precos = yf.download(Acoes, Inicio)['Adj Close'].fillna(method='ffill')
    Retornos = np.log(Precos / Precos.shift(1)).dropna(axis=0)

    retornos = Retornos.unstack().to_frame().reset_index()
    retornos.rename(columns={'level_0':'ticker','Date':'target_date',0:'retorno'},inplace=True)
    retornos['ticker'] = retornos.apply(lambda x: x['ticker'].replace('.SA',''),axis=1)
    port = pd.merge(portfolio,retornos,on='ticker')
    # port = pd.merge(port,companies,on='ticker')

    port['retorno_weighted'] = port['port_weight']*port['retorno']

    target_dates = port['target_date'].unique()
    port_weights = []
    for target_date in target_dates:
        port_date = port[port['target_date']==target_date]
        port_date['retorno_weighted']=port_date['retorno_weighted'].sum()
        port_date = port_date.drop(['ticker','retorno','port_weight'], axis = 1)
        port_date = port_date.iloc[:1,:]
        port_weights.append(port_date)

    port_weighted = pd.concat(port_weights)
    port_weighted.set_index('target_date',inplace=True)
    port_weighted['retorno_weighted'] = port_weighted['retorno_weighted']+1
    port_weighted['retorno_weighted'] = port_weighted['retorno_weighted'].cumprod() - 1

    fig_yield = px.line(port_weighted)

    #st.header("yield chart")
    #st.plotly_chart(fig_yield)  # YIELD CHART ########################################

    companies_rated = pd.read_sql_query("""SELECT comp.ticker,
                                            comp.industry,
                                            esg.score_value,
                                            esg.score_weight,
                                            esg.aspect,
                                            esg.assessment_year
                                    FROM COMPANIES_BR comp
                                    INNER JOIN esg_scores_history_br esg
                                    ON comp.company_id = esg.company_id
                                        AND esg.parent_aspect = 'S&P Global ESG Score'
                                    WHERE esg.assessment_year = 2020""",engine)

    companies= pd.read_sql_query("""SELECT ticker, industry FROM COMPANIES_BR""",engine)

    acoes = companies_rated['ticker'].to_list()
    for i,acao in enumerate(acoes):
        acoes[i] += ".SA"

    precos = yf.download(acoes, Inicio)['Adj Close'].fillna(method='ffill')

    retornos = np.log(Precos / Precos.shift(1)).dropna(axis=0)
    retornos = Retornos.unstack().to_frame().reset_index()
    retornos.rename(columns={'level_0':'ticker','Date':'target_date',0:'retorno'},inplace=True)
    retornos['ticker'] = retornos.apply(lambda x: x['ticker'].replace('.SA',''),axis=1)


    years = companies_rated['assessment_year'].unique()
    tickers = companies_rated['ticker'].unique()
    dfs = []
    for year in years:
        for ticker in tickers:
            df_year = companies_rated[(companies_rated['assessment_year'] == year) & (companies_rated['ticker'] == ticker)]
            df_year['single_esg_score'] = df_year['score_value']*df_year['score_weight']
            df_year['esg_score'] = df_year['single_esg_score'].sum()/df_year['score_weight'].sum()
            df_year = df_year.drop(['score_weight','score_value','single_esg_score','aspect','assessment_year'], axis = 1)
            df_year = df_year.iloc[:1,:]
            dfs.append(df_year)

    df_score = pd.concat(dfs)


    retornos_rated = pd.merge(df_score,retornos,on='ticker')

    port_rated = pd.merge(portfolio,df_score,on='ticker')
    scored_sectors =  port_rated['industry'].unique() #esses caras q vao ter q ser carregados no select

    option = st.selectbox('Setor:',scored_sectors)

    sector_port = port_rated[port_rated['industry']=='Banks'] #banks nesse exemplo mas vai ter q ser uma variavel q vem do select
    tickers_sector = sector_port['ticker'].to_list()

    retornos_sector_port = retornos_rated[(retornos_rated['industry']=='Banks') & (retornos_rated['ticker'].isin(tickers_sector))]
    retornos_sector_port = pd.merge(retornos_sector_port,sector_port[['ticker','port_weight']],on='ticker')

    def get_retorno(port):
        target_dates = port['target_date'].unique()
        port_weights = []
        port['retorno_weighted'] = port['retorno']*port['port_weight']
        for target_date in target_dates:
            port_date = port[port['target_date']==target_date]
            port_date['retorno_weighted']=port_date['retorno_weighted'].sum()/port_date['port_weight'].sum()
            port_date = port_date.drop(['ticker','retorno','port_weight'], axis = 1)
            port_date = port_date.iloc[:1,:]
            port_weights.append(port_date)

        port_weighted = pd.concat(port_weights)
        port_weighted.set_index('target_date',inplace=True)
        port_weighted['retorno_weighted'] = port_weighted['retorno_weighted']+1
        port_weighted['retorno_weighted'] = port_weighted['retorno_weighted'].cumprod() - 1 
        return port_weighted

    #retorno do setor
    df = get_retorno(retornos_sector_port)
    #st.write(retornos_sector_port)
    sector_yield = px.line(df['retorno_weighted'])
    st.header("Retorno por setor")
    st.plotly_chart(sector_yield)

    ######
    ##### INICIO PORT SECTOR

    #CARREGANDO A CARTEIRA

    # portfolio = pd.read_csv('IBOVDia_27-09-21.csv',skiprows=1,encoding = "ISO-8859-1", sep=';')
    # portfolio.reset_index(inplace=True)
    # cols = portfolio.columns
    # cols = cols[1:]
    # portfolio = portfolio.iloc[:,:-1]
    # portfolio.columns = cols
    # portfolio = portfolio.drop(['Ação','Tipo','Qtde. Teórica'], axis = 1)
    # portfolio = portfolio.iloc[:-2,:]
    # portfolio.rename(columns={'Código':'ticker','Part. (%)':'port_weight'},inplace=True)
    # portfolio['port_weight'] = portfolio.apply(lambda x: float(x['port_weight'].replace(',','.'))/100,axis=1)
    portfolio = carteira.copy()
    hist = pd.read_sql_query("""SELECT comp.ticker,
                                    hist.sector_name,
                                    hist.data_item_name,
                                    hist.data_item_value,
                                    hist.fiscal_year
                                FROM COMPANIES_BR comp
                                INNER JOIN physical_risk_history_br hist
                                ON comp.company_id = hist.company_id""",engine)
    fut = pd.read_sql_query("""SELECT comp.ticker,
                                    fore.sector_name,
                                    fore.data_item_name,
                                    fore.data_item_value,
                                    fore.scenario_level,
                                    fore.forecast_year,
                                    fore.fiscal_year
                                FROM COMPANIES_BR comp
                                INNER JOIN physical_risk_forecast_br fore
                                ON comp.company_id = fore.company_id""",engine)



    def change_scenario(row):
        scenario = row['scenario_level']
        if scenario == 'High':
            return 3
        elif scenario == 'Medium':
            return 2
        else:
            return 1

    fut['scenario_level'] = fut.apply(change_scenario,axis=1)

    tickers = fut['ticker'].unique()
    # ticker = fut[fut['ticker']=='GOLL4']
    dfs_fut = []
    #pegando o fiscal_year mais recente e fazendo a media
    for ticker in tickers:
        ticker_fut = fut[fut['ticker']==ticker]
        ticker_fut = ticker_fut[ticker_fut['fiscal_year']==ticker_fut['fiscal_year'].max()]
        fut_years = ticker_fut['forecast_year'].unique()
        print(fut_years)
    for fut_year in fut_years:
        ticker_fut_ind = ticker_fut[ticker_fut['forecast_year']==fut_year]
        ticker_fut_ind['data_item_value'] = ticker_fut_ind['data_item_value']*ticker_fut_ind['scenario_level']
        ticker_fut_ind['data_item_value'] = ticker_fut_ind['data_item_value'].sum()/ticker_fut_ind['scenario_level'].sum()
        ticker_fut_ind = ticker_fut_ind.iloc[:1,:]
        ticker_fut_ind = ticker_fut_ind.drop(['scenario_level','data_item_name'], axis = 1)
        dfs_fut.append(ticker_fut_ind)

    df_fut = pd.concat(dfs_fut)
    df_fut = df_fut.drop(['fiscal_year'], axis = 1)
    df_fut.rename(columns={'forecast_year':'fiscal_year'},inplace=True)

    tickers = hist['ticker'].unique()
    hist_years = hist['fiscal_year'].unique()

    dfs_hist = []
    #pegando o fiscal_year mais recente e fazendo a media
    for ticker in tickers:
        for hist_year in hist_years:
            ticker_hist = hist[(hist['fiscal_year']==hist_year) & (hist['ticker']==ticker)]
            ticker_hist['data_item_value'] = ticker_hist['data_item_value'].sum()/len(ticker_hist)
            ticker_hist = ticker_hist.iloc[:1,:]
            ticker_hist = ticker_hist.drop(['data_item_name'], axis = 1)
            dfs_hist.append(ticker_hist)

    df_hist = pd.concat(dfs_hist)

    port_fut = pd.merge(portfolio,df_fut,on='ticker')
    port_hist = pd.merge(portfolio,df_hist,on='ticker')

    def get_env_charts(port):
        sectors = port['sector_name'].unique()
        years = port['fiscal_year'].unique()
        port_sectors = []
        port_weights = []

        for year in years:
            for sector in sectors:
                df_year = port[(port['fiscal_year'] == year) & (port['sector_name'] == sector)]
                df_year['data_item_value'] = df_year['port_weight']*df_year['data_item_value']
                df_year['data_item_value'] = df_year['data_item_value'].sum()/df_year['port_weight'].sum()
                df_year['port_weight'] = df_year['port_weight'].sum()
                df_year = df_year.drop(['ticker'], axis = 1)
                df_year = df_year.iloc[:1,:]
                port_sectors.append(df_year)

        port_sector = pd.concat(port_sectors)

        for year in years:
            df_year = port_sector[port_sector['fiscal_year'] == year]
            df_year['data_item_value'] = df_year['port_weight']*df_year['data_item_value']
            df_year['data_item_value'] = df_year['data_item_value'].sum()/df_year['port_weight'].sum()
            df_year['port_weight'] = df_year['port_weight'].sum()
            df_year = df_year.drop(['sector_name'], axis = 1)
            df_year = df_year.iloc[:1,:]
            port_weights.append(df_year)

        port_weighted = pd.concat(port_weights)

        return port_weighted, port_sector

    # (port_hist_weighted, port_hist_sector) = get_env_charts(port_hist)
    # (port_fut_weighted, port_fut_sector) = get_env_charts(port_fut)

    # sector_list = port_hist_sector['sector_name'].unique()
    # current_years = port_hist_weighted['fiscal_year'].unique()
    # future_years = port_fut_weighted['fiscal_year'].unique()
    # years = np.concatenate([current_years,future_years])
    # years = np.unique(years)



    # port_sector_fig = go.Figure()
    # # Create and style traces
    # for i,sector in enumerate(sector_list):
    #     x1 = port_hist_sector[port_hist_sector['sector_name']==sector]['fiscal_year'].to_list()
    #     for i,year in enumerate(x1):
    #         x1[i] = str(year)
    #     x2 = port_fut_sector[port_fut_sector['sector_name']==sector]['fiscal_year']
    #     for i,year in enumerate(x2):
    #         x2[i] = str(year)
    #     port_sector_fig.add_trace(go.Scatter(x=x1, y=port_hist_sector[port_hist_sector['sector_name']==sector]['data_item_value'], name=sector,line = dict(width=1)))
    #     port_sector_fig.add_trace(go.Scatter(x=x2, y=port_fut_sector[port_fut_sector['sector_name']==sector]['data_item_value'], name=sector,line = dict(width=1, dash='dot')))
        
    #     if i>9:
    #         break
    #     #fig.add_trace(go.Scatter(x=years, y=y['port_weight'], name=sector))

    # port_sector_weighted = go.Figure()
    # port_sector_weighted.add_trace(go.Scatter(x=x1, y=port_hist_weighted['data_item_value'], name='historico',line = dict(width=1)))
    # port_sector_weighted.add_trace(go.Scatter(x=x2, y=port_fut_weighted['data_item_value'], name='futuro',line = dict(width=1, dash='dot')))

    # port_chart_selector = st.selectbox('Chart Selector:',
    #     ('Default', 'Weighted'))

    # if port_chart_selector == 'Default':
    #     st.plotly_chart(port_sector_fig)
    # elif port_chart_selector == 'Weighted':
    #     st.plotly_chart(port_sector_weighted)

    # ### FIM PORT SECTOR
   