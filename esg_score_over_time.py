import pandas as pd
import sqlalchemy 
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def esg_over_time(portfolio, engine):
    
    

    #portfolio = pd.read_csv(carteira_csv,skiprows=1,encoding = "ISO-8859-1", sep=';')
    #portfolio = pd.read_csv('IBOVDia_27-09-21.csv',skiprows=1,encoding = "ISO-8859-1", sep=';')
    # portfolio.reset_index(inplace=True)
    # cols = portfolio.columns
    # cols = cols[1:]
    # portfolio = portfolio.iloc[:,:-1]
    # portfolio.columns = cols
    # portfolio = portfolio.drop(['Ação','Tipo','Qtde. Teórica'], axis = 1)
    # portfolio = portfolio.iloc[:-2,:]
    # portfolio.rename(columns={'Código':'ticker','Part. (%)':'port_weight'},inplace=True)
    # portfolio['port_weight'] = portfolio.apply(lambda x: float(x['port_weight'].replace(',','.'))/100,axis=1)
    #portfolio = pd.read_csv(carteira_csv)
    esg_score = pd.read_sql_query("""SELECT comp.ticker,
                                            comp.industry,
                                            esg.score_value,
                                            esg.score_weight,
                                            esg.aspect,
                                            esg.assessment_year
                                    FROM COMPANIES_BR comp
                                    INNER JOIN esg_scores_history_br esg
                                    ON comp.company_id = esg.company_id
                                        AND esg.parent_aspect = 'S&P Global ESG Score'""",engine)

    df = pd.merge(portfolio,esg_score,on='ticker')
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
    # df_score.drop_duplicates(inplace=True)

    sectors = df_score['industry'].unique()
    df_sector = []
    for year in years:
        for sector in sectors:
            df_year = df_score[(df_score['assessment_year'] == year) & (df_score['industry'] == sector)]
            df_year['esg_sector_ind'] = df_year['port_weight']*df_year['esg_score']
            df_year['esg_sector'] = df_year['esg_sector_ind'].sum()/df_year['port_weight'].sum()
            df_year['port_weight'] = df_year['port_weight'].sum()
            df_year = df_year.drop(['ticker','esg_score','esg_sector_ind'], axis = 1)
            df_year = df_year.iloc[:1,:]
            df_sector.append(df_year)

    df_sector_score = pd.concat(df_sector)

    df_weights = []
    for year in years:
        df_year = df_sector_score[df_sector_score['assessment_year'] == year]
        df_year['esg_score'] = df_year['port_weight']*df_year['esg_sector']
        df_year['esg_score'] = df_year['esg_score'].sum()/df_year['port_weight'].sum()
        df_year['port_weight'] = df_year['port_weight'].sum()
        df_year = df_year.drop(['industry','esg_sector'], axis = 1)
        df_year = df_year.iloc[:1,:]
        df_weights.append(df_year)

    df_weighted = pd.concat(df_weights)
    df_weighted.sort_values(by='assessment_year', inplace = True)
    return df_weighted