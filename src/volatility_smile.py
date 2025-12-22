import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import norm
from datetime import datetime, time, timedelta

import yfinance as yf
import pandas as pd
from matplotlib import cm
import pytz as pytz
from matplotlib.ticker import LinearLocator



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



def minutes_to_close(datetime_series):
    close_time = 16
    datetime_series = datetime_series.dt.tz_convert('America/New_York')
    four_pm_today = datetime_series.dt.normalize() + pd.Timedelta(hours=close_time)
    time_difference = four_pm_today - datetime_series
    minutes = time_difference.dt.total_seconds() / 60

    return minutes /60/24/365



dt1_str = "2025-12-15 20:30:00+00:00"
est = pytz.timezone('US/Eastern')
utc = pytz.utc
fmt = '%Y-%m-%d %H:%M:%S %Z%z'
dt1 = datetime.fromisoformat(dt1_str)


data = yf.download(['SPY','^VIX'], start='2025-12-15', end='2025-12-16', interval='1m')

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

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X = call_prices.columns
Y = spy['Datetime']
X, Y = np.meshgrid(X, Y)
Z = call_prices.values 
st.write(X,Y,Z)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

### From matplotlib tutorials

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)



# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

st.pyplot(fig)
plt.show()