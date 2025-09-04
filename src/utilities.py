import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.stats import norm
from scipy import stats as st
from datetime import datetime, time, timedelta
import streamlit as st
import pytz as pytz

def minutes_to_close(datetime_series):
    close_time = 16 # 4 PM
    datetime_series = datetime_series.dt.tz_convert('America/New_York')
    four_pm_today = datetime_series.dt.normalize() + pd.Timedelta(hours=close_time)
    time_difference = four_pm_today - datetime_series
    minutes = time_difference.dt.total_seconds() / 60

    return minutes /60/24/365

def getStockData(ticker, start, end, interval):
    data = yf.download([ticker], start=start, end=end, interval=interval)
    data = data.dropna()
    data = data.reset_index()
    data['Time'] = pd.to_datetime(data['Datetime'], utc=True, format='%H:%M:%S.%f').dt.tz_convert('America/New_York').dt.time
    stock = pd.DataFrame(data={
        'Datetime': data['Datetime'],
        'Time': data['Time'],
        'Close': data['Close'][ticker],
        'Open': data['Open'][ticker],
        'High': data['High'][ticker],
        'Low': data['Low'][ticker],
        'Volume': data['Volume'][ticker],
        'Return': data['Close'][ticker].pct_change()
    })
    stock['Datetime'] = pd.to_datetime(data['Datetime'], utc=True)
    stock['Time'] = pd.to_datetime(data['Datetime'], utc=True, format='%H:%M:%S.%f').dt.tz_convert('America/New_York').dt.time
    stock['T'] = data.apply(lambda row: minutes_to_close(row['Datetime']), axis=1)
    return stock

def C(S,K,T, r=0.03, sigma=0.5):
    S = float(S)
    T = float(T)
    if T == 0:
        return max(S - K, 0)
    if T < 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def P(S,K,T, r=0.03, sigma=0.5):
    price = C(S,K,T, r=r, sigma=sigma) - S + K * np.exp(-r * T) #put call parity!
    return price







