import pandas as pd
import sqlalchemy 
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


#engine = create_engine("cockroachdb://quant:gz43t_wo4zesYneu@free-tier.gcp-us-central1.cockroachlabs.cloud:26257/grown-iguana-3784.defaultdb?sslmode=require")

#CARREGANDO A CARTEIRA
def port_risk(carteira_csv, engine):
    portfolio = pd.read_csv(carteira_csv)
    # portfolio.reset_index(inplace=True)
    # cols = portfolio.columns
    # cols = cols[1:]
    # portfolio = portfolio.iloc[:,:-1]
    # portfolio.columns = cols
    # portfolio = portfolio.drop(['Ação','Tipo','Qtde. Teórica'], axis = 1)
    # portfolio = portfolio.iloc[:-2,:]
    # portfolio.rename(columns={'Código':'ticker','Part. (%)':'port_weight'},inplace=True)
    # portfolio['port_weight'] = portfolio.apply(lambda x: float(x['port_weight'].replace(',','.'))/100,axis=1)

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

      
    return get_env_charts(port_hist), get_env_charts(port_fut)