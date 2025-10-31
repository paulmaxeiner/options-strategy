# app.py
# Streamlit ML app: predict next-day stock direction & backtest a simple strategy
# pip install streamlit yfinance scikit-learn pandas numpy matplotlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from dataclasses import dataclass
from typing import Tuple, List

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt


SPY = yf.download('SPY', start='2025-01-01', end='2025-12-31', interval='1d')
VIX = yf.download('^VIX', start='2025-01-01', end='2025-12-31', interval='1d')

X = SPY['Close'].pct_change().dropna().values.reshape(-1, 1)
Y = VIX['Close'].pct_change().dropna().values.reshape(-1, 1)



fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X, Y, alpha=0.5)
ax.set_title("Cumulative Returns of SPY and VIX in 2025")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.legend()
ax.grid(True)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X.flatten(), Y.flatten(), alpha=0.5)
ax.set_title("Cumulative Returns of SPY and VIX in 2025")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.legend()
ax.grid(True)
st.pyplot(fig)

correlation = np.corrcoef(X.flatten(), Y.flatten())[0, 1]
st.write(f"Correlation between SPY returns and VIX returns in 2025: {correlation:.2f}")

