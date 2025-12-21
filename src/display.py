# intraday_single_day.py
import math
from datetime import datetime, date, time, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from scipy.stats import norm
import matplotlib.pyplot as plt
import pytz

NY = pytz.timezone("America/New_York")

# ---------------------------
# Black–Scholes (Call)
# ---------------------------
def bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    sigma = max(float(sigma), 1e-9)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

# ---------------------------
# Data
# ---------------------------
def clamp_period_for_interval(max_days_back: int, interval: str) -> str:
    lim = 7 if interval == "1m" else 60  # rough Yahoo intraday limits
    return f"{min(max_days_back, lim)}d"

@st.cache_data(show_spinner=False)
def load_intraday_period(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    # Ensure tz-aware New York time
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(NY)
    else:
        df.index = df.index.tz_convert(NY)
    return df

# ---------------------------
# Trade structure
# ---------------------------
@dataclass
class Trade:
    date: pd.Timestamp
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_S: float
    exit_S: float
    K: float
    T_entry: float
    T_exit: float
    entry_px: float
    exit_px: float
    pnl: float

# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="Single-Day Intraday ATM Call", layout="wide")
st.title("Single-Day Intraday ATM Call — Buy at Open, Sell 1h Before Close")

with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Ticker", "SPY")
    interval = st.selectbox("Bar interval", ["1m", "2m", "5m", "15m"], index=2)

    today_ny = datetime.now(tz=NY).date()
    trade_day = st.date_input("Trading day (New York date)", value=today_ny)

    const_sigma = st.number_input("Constant σ (annualized)", value=0.20, step=0.01, format="%.2f")
    r = st.number_input("Risk-free rate (annual)", value=0.02, step=0.005, format="%.3f")
    close_buffer_min = st.number_input("Sell minutes before close", value=60, min_value=5, max_value=240, step=5)
    contracts = st.number_input("# contracts", value=1, min_value=1, step=1)
    multiplier = st.number_input("Contract multiplier", value=100, min_value=1, step=1)
    slippage = st.number_input("Per-leg slippage/fees ($)", value=0.50, step=0.05)
    init_capital = st.number_input("Initial Capital ($)", value=100000, step=1000)

# Fetch enough recent history, then filter to the selected day
days_back_from_today = (datetime.now(tz=NY).date() - trade_day).days + 1
period = clamp_period_for_interval(days_back_from_today if days_back_from_today > 0 else 1, interval)
idata = load_intraday_period(ticker, period=period, interval=interval)

if idata.empty:
    st.error("No intraday data returned. If you chose 1m bars, the day must be within ~7 days; for 2m/5m/15m, within ~60 days.")
    st.stop()

# Keep only selected NY date and regular session
start_t = time(9, 30)
end_t = time(16, 0)

df_day = idata[idata.index.date == trade_day]
df_day = df_day[(df_day.index.time >= start_t) & (df_day.index.time <= end_t)][["Close"]].dropna()

if df_day.empty:
    st.error("No regular-session bars for the selected day. It may be outside provider limits or a non-trading day.")
    st.stop()

# ---------------------------
# Build the ONE trade for this day
# ---------------------------
entry_ts = df_day.index[0]  # first bar >= 09:30
official_close = NY.localize(datetime.combine(trade_day, end_t))
target_exit = official_close - timedelta(minutes=int(close_buffer_min))
before_exit = df_day[df_day.index <= target_exit]
if before_exit.empty:
    st.error("No bar found ≤ (close − buffer). Try a larger interval or smaller buffer.")
    st.stop()
exit_ts = before_exit.index[-1]

if exit_ts <= entry_ts:
    st.error("Exit bar is not after entry bar. Adjust buffer/interval.")
    st.stop()

S_entry = float(df_day.loc[entry_ts, "Close"])
S_exit = float(df_day.loc[exit_ts, "Close"])
K = S_entry  # ATM at entry

# Time to expiry -> minutes to 16:00 same day; convert to years via 252*390 minutes
minutes_entry_to_close = max(0, int((official_close - entry_ts).total_seconds() // 60))
minutes_exit_to_close = max(0, int((official_close - exit_ts).total_seconds() // 60))
T_entry = minutes_entry_to_close / (252.0 * 390.0)
T_exit = minutes_exit_to_close / (252.0 * 390.0)

call_entry = bs_call_price(S_entry, K, T_entry, r, const_sigma)
call_exit  = bs_call_price(S_exit,  K, T_exit,  r, const_sigma)

pnl_per_contract = (call_exit - call_entry) * multiplier - 2.0 * slippage
pnl = pnl_per_contract * int(contracts)
final_capital = float(init_capital) + pnl

trade = Trade(
    date=pd.Timestamp(trade_day),
    entry_ts=entry_ts, exit_ts=exit_ts,
    entry_S=S_entry, exit_S=S_exit,
    K=K, T_entry=T_entry, T_exit=T_exit,
    entry_px=call_entry * multiplier, exit_px=call_exit * multiplier,
    pnl=pnl
)

# ---------------------------
# Intraday mark-to-market equity curve (for this day)
# ---------------------------
times = df_day.index

# 1) Cash after buying at entry (apply one leg of slippage)
cash_after_entry = float(init_capital) - (call_entry * multiplier * int(contracts)) - slippage

# 2) Build MTM only from entry..exit, using pandas Series (avoid ndarray shape issues)
mtm_series = pd.Series(np.nan, index=times, dtype=float)

post_entry = times[(times >= entry_ts) & (times <= exit_ts)]
if len(post_entry):
    # Remaining minutes to close for each bar as plain ints (robust conversion)
    rem_td = official_close - post_entry  # TimedeltaIndex
    # convert to integer minutes safely:
    rem_minutes = (rem_td // pd.Timedelta(minutes=1)).astype(int).to_numpy()

    T_vec = rem_minutes / (252.0 * 390.0)
    S_vec = df_day.loc[post_entry, "Close"].to_numpy(dtype=float)

    # Vectorized pricing via list comprehension → pandas Series aligned by index
    opt_vals = [bs_call_price(float(S_vec[i]), float(K), float(T_vec[i]), r, const_sigma)
                for i in range(len(S_vec))]
    opt_vals = (np.asarray(opt_vals, dtype=float) * multiplier * int(contracts))
    opt_series = pd.Series(opt_vals, index=post_entry, dtype=float)

    # Update MTM (avoids "setting array element with a sequence")
    mtm_series.update(opt_series)

# 3) Equity curve:
equity_curve = pd.Series(init_capital, index=times, dtype=float)
if len(post_entry):
    equity_curve.loc[post_entry] = cash_after_entry + mtm_series.loc[post_entry]
# Snap equity at exit onward to final capital (reflecting sale + second leg slippage)
equity_curve.loc[exit_ts:] = final_capital

# ---------------------------
# Output
# ---------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("P&L ($)", f"{trade.pnl:,.2f}")
c2.metric("Entry Option ($)", f"{trade.entry_px:,.2f}")
c3.metric("Exit Option ($)", f"{trade.exit_px:,.2f}")
c4.metric("Final Equity ($)", f"{final_capital:,.2f}")

st.subheader("Equity Curve (selected day only)")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(equity_curve.index, equity_curve.values, label="Equity (MTM)")
ax1.axvline(trade.entry_ts, linestyle="--", linewidth=1)
ax1.axvline(trade.exit_ts, linestyle="--", linewidth=1)
ax1.grid(True, alpha=0.3)
ax1.legend()
st.pyplot(fig1)

st.subheader("Price with Entry/Exit Marks")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df_day.index, df_day["Close"], label=f"{ticker} Close")
ax2.scatter(trade.entry_ts, trade.entry_S, marker="^", s=90, label="Buy 09:30", zorder=3)
ax2.scatter(trade.exit_ts,  trade.exit_S,  marker="v", s=90, label=f"Sell {int(close_buffer_min)}m pre-close", zorder=3)
ax2.grid(True, alpha=0.3)
ax2.legend()
st.pyplot(fig2)

st.subheader("Trade Log (1 trade)")
tdf = pd.DataFrame([trade.__dict__])
tdf["date"] = tdf["date"].dt.date
tdf["entry_px"] = tdf["entry_px"].round(2)
tdf["exit_px"] = tdf["exit_px"].round(2)
tdf["pnl"] = tdf["pnl"].round(2)
st.dataframe(tdf[[
    "date","entry_ts","exit_ts","entry_S","exit_S","K",
    "T_entry","T_exit","entry_px","exit_px","pnl"
]], use_container_width=True)

with st.expander("Notes"):
    st.markdown(f"""
- **Exactly one trade** on the selected New York date.
- **Entry:** first regular-session bar (≥ 09:30 ET).
- **Exit:** last bar ≤ 16:00 − {int(close_buffer_min)} minutes.
- **Option:** ATM call (K = spot at entry), **0DTE** expiring 16:00 same day.
- **Volatility:** constant σ = {const_sigma:.2f} (annualized).
- **T (years):** minutes to 16:00 divided by 252×390.
- **PnL:** \[(exit − entry) × {int(multiplier)} × {int(contracts)}\] − 2×slippage.
- **Provider limits:** {interval} bars typically available for ~{'7' if interval=='1m' else '60'} days from today.
""")
