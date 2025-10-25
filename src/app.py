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
from copulas.multivariate import GaussianMultivariate

with st.sidebar:
    st.title("0DTE Options Strategy Simulator")
    st.markdown("""
    This app simulates a simple 0DTE call option trading strategy based on RSI signals / VIX Copula Signals.
    It uses the Black-Scholes model to price options and evaluates strategy performance over intraday data.
    """)
    d = st.date_input("Select Date", value=datetime(2025, 10, 14), min_value=datetime(2025, 1, 1), max_value=datetime(2025, 12, 31))

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



def minutes_to_close(datetime_series):
    close_time = 16
    datetime_series = datetime_series.dt.tz_convert('America/New_York')
    four_pm_today = datetime_series.dt.normalize() + pd.Timedelta(hours=close_time)
    time_difference = four_pm_today - datetime_series
    minutes = time_difference.dt.total_seconds() / 60

    return minutes /60/24/365



dt1_str = "2025-10-14 20:30:00+00:00"
est = pytz.timezone('US/Eastern')
utc = pytz.utc
fmt = '%Y-%m-%d %H:%M:%S %Z%z'
dt1 = datetime.fromisoformat(dt1_str)


data = yf.download(['SPY','^VIX'], start='2025-10-14', end='2025-10-15', interval='1m')

data = data.dropna()
data = data.reset_index()

data['Time'] = pd.to_datetime(data['Datetime'], utc=True, format='%H:%M:%S.%f').dt.tz_convert('America/New_York').dt.time

spy = pd.DataFrame(data={
    'Datetime': data['Datetime'],
    'Time': data['Time'],
    'Close': data['Close']['SPY'],
    'Open': data['Open']['SPY'],
    'High': data['High']['SPY'],
    'Low': data['Low']['SPY'],
    'Volume': data['Volume']['SPY'],
    'Return': data['Close']['SPY'].pct_change()
})


spy['Datetime'] = pd.to_datetime(spy['Datetime'], utc=True)
spy['Time'] = pd.to_datetime(data['Datetime'], utc=True, format='%H:%M:%S.%f').dt.tz_convert('America/New_York').dt.time
spy['T'] = data.apply(lambda row: minutes_to_close(row['Datetime']), axis=1)
st.write(spy)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(spy['Close'], label='SPY Price')
ax.set_title("SPY")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)


atm_price = round(spy['Close'].iloc[-1])
st.write("ATM Price:", atm_price)

call_prices = pd.DataFrame(data={
    atm_price + x: spy.apply(lambda row: black_scholes_call(row['Close'], row['T'], atm_price + x), axis=1)
    for x in range(-10, 10)
})

call_returns = pd.DataFrame(data={     
    atm_price + x: (spy.apply(lambda row: black_scholes_call(row['Close'], row['T'], atm_price + x), axis=1).pct_change()+1).cumprod()-1
    for x in range(-10, 10)
}) 

st.write(call_prices)
st.write(call_returns)


S0 = spy['Close'].rolling(60, min_periods=1).mean().iloc[-1]
atm = int(round(S0))

K_values = np.linspace(atm - 20, atm + 20, 20) # k strike prices around the atm price
T_eps = 1e-6
T_values = np.linspace(max(T_eps, spy['T'].min()), max(T_eps, spy['T'].max()), 20)

points = []
for K in K_values:
    for T in T_values:
        C = black_scholes_call(S0, float(T), float(K))
        points.append((K, T, C))

K_arr, T_arr, C_arr = np.array(points).T


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(K_arr, T_arr, C_arr, c=C_arr, cmap=cm.viridis, s=30)
ax.set_xlabel('K')
ax.set_ylabel('Time to Expiry T (years)')
ax.set_zlabel('Call Price C')
ax.set_title('Black–Scholes Call Price Points 0DTE')
fig.colorbar(sc, shrink=0.6, aspect=14, label='Call Price')

st.pyplot(fig)






st.line_chart(call_prices)
st.line_chart(call_returns)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(call_prices, label=call_prices.columns)
ax.set_title("0DTE Call Prices")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)


data['T'] = data.apply(lambda row: minutes_to_close(row['Datetime']), axis=1)

##data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
data.columns = [' '.join(col).strip() for col in data.columns.values]
data['RSI'] = calculate_rsi(data['Close SPY'])



### Actual strategy backtest

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
st.line_chart(data['RSI'])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['SPY_ret'],data['VIX_ret'], linestyle='', marker='o', markersize=2, alpha=0.5)
st.pyplot(fig)


# options backtest

CONTRACT_MULT = 100 # standard options contract multiplier, 100 shares per contract

def call_px(S, T, K): # black scholes call price with T=0 handling
    if T <= 0:
        return max(S - K, 0.0)
    return black_scholes_call(S, T, K)

position = 0
entry_strike = None
entry_px = 0.0
entry_time = None
entry_rsi = None

cum_pnl = 0.0
pnl_series = []
trades = []

for _, row in data.iterrows():
    S   = float(row['Close SPY'])
    T   = float(row['T'])
    rsi = float(row['RSI'])
    ts  = row['Datetime'] #time

    #sell if RSI is high
    if position == 1 and (rsi > 70 or T <= 0):
        
        exit_px = call_px(S, T, entry_strike)
        
        trade_pnl = (exit_px - entry_px) * CONTRACT_MULT
        cum_pnl += trade_pnl
        
        trades.append({
            'Open Time':  entry_time,
            'Close Time': ts,
            'Strike': int(entry_strike),
            'Entry RSI': round(entry_rsi, 2),
            'Exit RSI': round(rsi, 2),
            'Entry Px': round(entry_px, 4),
            'Exit Px': round(exit_px, 4),
            'PnL ($)': round(trade_pnl, 2),
        })
        position = 0
        
        entry_strike = None
        
        entry_px = 0.0
        
        entry_time = None
        entry_rsi = None

    #BUY if RSI is low
    if position == 0 and rsi < 30 and T > 0:
        entry_strike = int(round(S)) # ATM strike price
        entry_px = call_px(S, T, entry_strike)
        entry_time = ts
        entry_rsi = rsi
        position = 1

    pnl_series.append(cum_pnl)

# close at end of trading day
if position == 1:
    last = data.iloc[-1]
    
    S_end, T_end, rsi_end, ts_end = float(last['Close SPY']), float(last['T']), float(last['RSI']), last['Datetime']
    exit_px = call_px(S_end, T_end, entry_strike)
    
    trade_pnl = (exit_px - entry_px) * CONTRACT_MULT
    cum_pnl += trade_pnl
    
    
    trades.append({
        'Open Time': entry_time,
        'Close Time': ts_end,
        'Strike': int(entry_strike),
        'Entry RSI': round(entry_rsi, 2),
        'Exit RSI': round(rsi_end, 2),
        'Entry Px': round(entry_px, 4),
        'Exit Px': round(exit_px, 4),
        'PnL ($)': round(trade_pnl, 2),
    })
    if pnl_series: pnl_series[-1] = cum_pnl
    position = 0

data['CumPnL_$'] = pnl_series


## DISPLAY RESULTS

st.metric("Cumulative P&L in $", value=round(data['CumPnL_$'].iloc[-1], 2))
trades_df = pd.DataFrame(trades)

st.subheader("Trades")

if trades_df.empty:
    st.write("No trades executed.")
else:
    st.dataframe(trades_df)
    
    


