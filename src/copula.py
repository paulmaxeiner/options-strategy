import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats as st
from datetime import datetime, time, timedelta
import streamlit as st
import pytz as pytz
from copulas.multivariate import GaussianMultivariate

np.random.seed(1024)

data = yf.download(['SPY','^VIX'], start='2025-08-15', end='2025-08-16', interval='1m')
data = data.dropna()

print(data)


