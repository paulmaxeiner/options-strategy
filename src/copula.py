import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy.stats import rankdata, norm, kendalltau, spearmanr, t
from datetime import datetime, time, timedelta
import streamlit as st
import pytz as pytz
from copulas.visualization import compare_3d, scatter_3d
from copulas.multivariate import GaussianMultivariate
from copulas.datasets import sample_trivariate_xyz

data = yf.download(['SPY','^VIX'], start='2025-09-21', end='2025-09-22', interval='1m')
data.reset_index(inplace=True)
data.dropna(inplace=True)

df = pd.DataFrame(data={
    'SPY': data['Close']['SPY'],
    'VIX': data['Close']['^VIX'],
    'SPY_ret': data['Close']['SPY'].pct_change(),
    'VIX_ret': data['Close']['^VIX'].pct_change()
})

print(data)

st.write(data)
st.write(df)

rets = df[['SPY_ret', 'VIX_ret']].replace([np.inf, -np.inf], np.nan).dropna().copy()

tau = kendalltau(rets['SPY_ret'], rets['VIX_ret']).correlation
rho_s = spearmanr(rets['SPY_ret'], rets['VIX_ret']).correlation

if len(rets) < 20:
    st.warning("not large enoughy dataset")
else:
    
    #Fit Gaussian copula
    model = GaussianMultivariate()
    model.fit(rets)  #columns:'SPY_ret', 'VIX_ret'

    # make sample from fitted copula
    m = len(rets)
    
    sim = model.sample(m)
    # ensure same column order/names
    sim = sim[['SPY_ret', 'VIX_ret']].dropna()




    #SPY
    st.subheader("SPY: Real vs Expected (Copula-Simulated) Returns")
    fig_spy = plt.figure(figsize=(6, 4))
    bins = 50
    plt.hist(rets['SPY_ret'].values, bins=bins, density=True, alpha=0.5, label="Real")
    plt.hist(sim['SPY_ret'].values, bins=bins, density=True, alpha=0.5, label="Expected (Copula Sim)")
    plt.xlabel("SPY minute return")
    plt.ylabel("Density")
    plt.title("SPY Returns — Real vs Copula Expected")
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_spy)
    plt.close(fig_spy)

    #VIX
    st.subheader("VIX: Real vs Expected (Copula-Simulated) Returns")
    fig_vix = plt.figure(figsize=(6, 4))
    plt.hist(rets['VIX_ret'].values, bins=bins, density=True, alpha=0.5, label="Real")
    plt.hist(sim['VIX_ret'].values, bins=bins, density=True, alpha=0.5, label="Expected (Copula Sim)")
    plt.xlabel("VIX minute return")
    plt.ylabel("Density")
    plt.title("VIX Returns — Real vs Copula Expected")
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_vix)
    plt.close(fig_vix)
    
    st.subheader("Scatter Plot: SPY and VIX Copula-Simulated Returns")
    st.scatter_chart(sim,x='SPY_ret',y='VIX_ret')
    
    rets['SPY_sim'] = sim['SPY_ret'].values
    rets['VIX_sim'] = sim['VIX_ret'].values
    
    
    
    fig_line = plt.figure(figsize=(10, 4))
    plt.plot(rets['SPY_ret'].values, label='SPY Real Returns', alpha=0.7)
    plt.plot(rets['SPY_sim'].values, label='SPY Copula Sim Returns', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("SPY Returns")
    plt.title("SPY Real vs Copula Simulated Returns")
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_line)
    plt.close(fig_line)

