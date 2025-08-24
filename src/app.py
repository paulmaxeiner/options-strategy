import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, time, timedelta
import streamlit as st
import pytz as pytz

K = st.number_input("Insert a value for K",min_value=1, value=625)
st.write("K:", K)

r=0.03
sigma=0.5

def black_scholes_call(S,T,K):
    S = float(S)
    T = float(T)
    if T == 0:
        return max(S - K, 0)  # Handle expiration edge case
    if T < 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi






dt1_str = "2025-07-15 20:30:00+00:00"
est = pytz.timezone('US/Eastern')
utc = pytz.utc
fmt = '%Y-%m-%d %H:%M:%S %Z%z'
dt1 = datetime.fromisoformat(dt1_str)


data = yf.download('SPY', start='2025-08-14', end='2025-08-15', interval='1m')
data = data.reset_index()

data['Date'] = data['Datetime'].dt.strftime(fmt)
data['Pct Change']=data['Close'].pct_change()

data['Time'] = pd.to_datetime(data['Datetime'], utc=True, format='%H:%M:%S.%f').dt.tz_convert('America/New_York').dt.time
st.write(data)

def minutes_to_4pm(datetime_series):
    # Ensure datetime is timezone-aware (in New York time)
    datetime_series = datetime_series.dt.tz_convert('America/New_York')

    # Create a datetime object for 4 PM on each day
    four_pm_today = datetime_series.dt.normalize() + pd.Timedelta(hours=16)

    # Compute the difference in minutes
    time_difference = four_pm_today - datetime_series
    minutes = time_difference.dt.total_seconds() / 60

    return minutes /60/24/365

data['T'] = data.apply(lambda row: minutes_to_4pm(row['Datetime']), axis=1)



##data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
data.columns = [' '.join(col).strip() for col in data.columns.values]
data['Call Price 625'] = data.apply(lambda row:black_scholes_call(row['Close SPY'],row['T'],625), axis=1)
data['RSI'] = calculate_rsi(data['Close SPY'])
data['Call Price 623'] = data.apply(lambda row:black_scholes_call(row['Close SPY'],row['T'],623), axis=1)
data['Call Price 624'] = data.apply(lambda row:black_scholes_call(row['Close SPY'],row['T'],624), axis=1)
data['Call Price 626'] = data.apply(lambda row:black_scholes_call(row['Close SPY'],row['T'],626), axis=1)
data['Call Price 630'] = data.apply(lambda row:black_scholes_call(row['Close SPY'],row['T'],630), axis=1)
data['Call Price 628'] = data.apply(lambda row:black_scholes_call(row['Close SPY'],row['T'],628), axis=1)


def rsi_signal(rsi):
    if rsi < 30:
        return 1     # Buy signal
    elif rsi > 70:
        return -1    # Sell signal
    else:
        return 0     # Hold

data['Signal'] = data['RSI'].apply(rsi_signal)

# Forward-fill signal to maintain position
data['Position'] = data['Signal'].replace(to_replace=0, method='ffill')

data['Position'] = data['Position'].shift()
data['Return'] = data['Close SPY'].pct_change()

data['Strategy_Return'] = data['Position'] * data['Return']



# Cumulative returns
data['Cumulative_Market'] = (1 + data['Return']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
plt.figure(figsize=(12, 6))
plt.plot(data['Cumulative_Market'], label='Market Return')
plt.plot(data['Cumulative_Strategy'], label='RSI Strategy Return')

st.write(data)

st.subheader("Strategy Performance")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['Cumulative_Market'], label='Market Return')
ax.plot(data['Cumulative_Strategy'], label='RSI Strategy Return')
ax.set_title("Backtest: RSI Strategy")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.legend()
ax.grid(True)
st.pyplot(fig)


fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['Close SPY'], label='Market Return')
st.pyplot(fig)
st.pyplot(fig)

st.line_chart(data,x='Datetime',y=['RSI','Call Price 625'])

st.line_chart(data,x='Datetime',y='Close SPY')
st.line_chart(data,x='Datetime',y=['Call Price 625','Call Price 626','Call Price 623','Call Price 624','Call Price 630','Call Price 628'])


##data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]









