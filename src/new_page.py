import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, time as dtime
import streamlit as st

st.set_page_config(page_title="Short Strangle Backtester", layout="wide")

NY_TZ = "America/New_York"
CONTRACT_MULT = 100
INTRADAY_LIMIT_DAYS = 60

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("Short Strangle Backtester (Sell Call + Put)")

    ticker = st.text_input("Underlying ticker", value="SPY").strip().upper()

    start_date = st.date_input("Start date", value=datetime(2025, 12, 1))
    end_date = st.date_input("End date", value=datetime(2025, 12, 20))

    interval = st.selectbox("Interval", ["1m", "2m", "5m", "15m", "1d"], index=2)

    st.subheader("Intraday hours (NY) (ignored for 1d)")
    entry_start = st.time_input("Entry start", value=dtime(9, 35))
    entry_end = st.time_input("Entry end", value=dtime(15, 30))
    forced_exit_time = st.time_input("Forced exit", value=dtime(15, 59))

    st.subheader("Strikes")
    otm_pct = st.slider("OTM % (both sides)", 0.25, 10.0, 1.0, 0.25)
    strike_round = st.selectbox("Strike rounding", [0.5, 1.0, 5.0], index=1)

    st.subheader("Sizing")
    contracts = st.number_input("Contracts", 1, 1000, 1, 1)

    st.subheader("Pricing inputs (toy BS)")
    r = st.number_input("Risk-free r", 0.0, 0.25, 0.03, 0.005)
    sigma = st.number_input("Flat IV sigma", 0.01, 3.0, 0.50, 0.05)

    st.subheader("Exit rules (per trade)")
    use_tp = st.checkbox("Use take-profit", value=True)
    tp_pct = st.slider("TP (% of credit)", 5, 95, 50, 5)

    use_sl = st.checkbox("Use stop-loss", value=True)
    sl_pct = st.slider("SL (% of credit)", 50, 600, 200, 25)

    st.subheader("Realism")
    slippage_cents = st.number_input("Slippage (cents per leg per fill)", 0.0, 50.0, 1.0, 0.5)
    fee_per_contract = st.number_input("Fee ($ per contract per leg)", 0.0, 5.0, 0.0, 0.05)

    st.subheader("Re-entry")
    overlap_trades = st.checkbox("Allow overlapping trades (intraday only)", value=False)
    max_open_trades = st.number_input("Max simultaneous open trades", 1, 50, 1, 1)

# ============================================================
# Validation (intraday Yahoo limit)
# ============================================================
if start_date > end_date:
    st.error("Start date must be <= end date.")
    st.stop()

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

is_intraday = interval in {"1m", "2m", "5m", "15m"}
today_utc = pd.Timestamp.utcnow().normalize()
oldest_allowed_utc = today_utc - pd.Timedelta(days=INTRADAY_LIMIT_DAYS)

if is_intraday and start_ts < oldest_allowed_utc.tz_localize(None):
    st.warning(
        f"Yahoo intraday data (1m–15m) is typically limited to the last ~{INTRADAY_LIMIT_DAYS} days.\n\n"
        f"Your start date {start_date} is older than {oldest_allowed_utc.date()}.\n"
        "Switch to interval='1d' or pick a more recent range."
    )
    if not st.button("Auto-switch to 1d and continue"):
        st.stop()
    interval = "1d"
    is_intraday = False

# ============================================================
# Helpers
# ============================================================
def make_unique_columns(cols):
    if isinstance(cols, pd.MultiIndex):
        cols = [" ".join([str(x) for x in tup]).strip() for tup in cols.to_flat_index()]
    else:
        cols = [str(c) for c in cols]

    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out

def normalize_ohlc_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    If yfinance returns 'Open TICKER' etc, rename to plain 'Open' 'High' 'Low' 'Close' 'Volume'.
    Works if columns are already plain.
    """
    # Already plain?
    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        return df

    # Common yfinance multi/suffixed forms
    mapping = {}
    for base in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        suff = f"{base} {ticker}"
        if suff in df.columns and base not in df.columns:
            mapping[suff] = base

    if mapping:
        df = df.rename(columns=mapping)

    return df

def bs_call(S, T, K, r, sigma):
    S = float(S); T = float(T); K = float(K)
    if T <= 0:
        return max(S - K, 0.0)
    if S <= 0 or K <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, T, K, r, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    C = bs_call(S, T, K, r, sigma)
    return C - S + K * np.exp(-r * T)

def round_strike(x, step):
    step = float(step)
    return step * round(float(x) / step)

def strangle_mid_per_share(S, T, Kc, Kp, r, sigma):
    return bs_call(S, T, Kc, r, sigma) + bs_put(S, T, Kp, r, sigma)

def time_to_forced_exit_years_intraday(dt_utc: pd.Series, forced_exit: dtime) -> pd.Series:
    dt_ny = dt_utc.dt.tz_convert(NY_TZ)
    forced_dt = dt_ny.dt.normalize() + pd.Timedelta(hours=forced_exit.hour, minutes=forced_exit.minute)
    minutes = (forced_dt - dt_ny).dt.total_seconds() / 60.0
    return (minutes / (60 * 24 * 365)).clip(lower=0)

# ============================================================
# Download
# ============================================================
start = pd.Timestamp(start_date)
end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

raw = yf.download(
    ticker,
    start=start.strftime("%Y-%m-%d"),
    end=end.strftime("%Y-%m-%d"),
    interval=interval,
    auto_adjust=False,
    progress=False,
).dropna()

if raw.empty:
    st.error("No data returned. Try different dates or interval=1d.")
    st.stop()

raw = raw.reset_index()
raw.columns = make_unique_columns(raw.columns)

# Find datetime column robustly
dt_candidates = [c for c in raw.columns if c.lower() in ("datetime", "date")]
if not dt_candidates:
    dt_candidates = [c for c in raw.columns if c.lower() == "index"]
if not dt_candidates:
    st.error(f"Couldn't find datetime column. Columns: {raw.columns.tolist()}")
    st.stop()
dt_col = dt_candidates[0]

raw[dt_col] = pd.to_datetime(raw[dt_col], utc=True)
raw["DatetimeNY"] = raw[dt_col].dt.tz_convert(NY_TZ)

# Normalize OHLC column names (this fixes your error)
raw = normalize_ohlc_columns(raw, ticker)

need = ["Open", "High", "Low", "Close"]
missing = [c for c in need if c not in raw.columns]
if missing:
    st.error(f"Missing columns: {missing}. Columns: {raw.columns.tolist()}")
    st.stop()

raw = raw.sort_values(dt_col).reset_index(drop=True)

# T and day columns
raw["NY_date"] = raw["DatetimeNY"].dt.date
if is_intraday:
    raw["T"] = time_to_forced_exit_years_intraday(raw[dt_col], forced_exit_time)
else:
    raw["T"] = 1.0 / 365.0  # daily approximation

# ============================================================
# Backtest engine (scalar-safe)
# ============================================================
contracts = int(contracts)
slip = float(slippage_cents) / 100.0
fee_leg = float(fee_per_contract)

def per_trade_fees():
    return fee_leg * contracts * 2

def credit_fill(mid_per_share):
    return max(0.0, float(mid_per_share) - 2.0 * slip)

def debit_fill(mid_per_share):
    return max(0.0, float(mid_per_share) + 2.0 * slip)

def tp_level(credit_dollars):
    return (float(tp_pct) / 100.0) * float(credit_dollars)

def sl_level(credit_dollars):
    return -(float(sl_pct) / 100.0) * float(credit_dollars)

# positional indices
col_index = {c: i for i, c in enumerate(raw.columns)}
i_dt_ny = col_index["DatetimeNY"]
i_day = col_index["NY_date"]
i_close = col_index["Close"]
i_T = col_index["T"]

open_positions = []
trades = []
cum_pnl = 0.0
equity_curve = []

for tup in raw.itertuples(index=False, name=None):
    ts_ny = tup[i_dt_ny]
    day = tup[i_day]
    S = float(tup[i_close])
    T = float(tup[i_T])

    if is_intraday:
        t_ny = ts_ny.time()
        can_enter = (entry_start <= t_ny <= entry_end) and (T > 0.0)
        must_force_exit = (t_ny >= forced_exit_time) or (T <= 0.0)
    else:
        # daily: enter + exit each bar (0DTE-ish approximation)
        can_enter = True
        must_force_exit = True

    # exits
    if open_positions:
        keep = []
        for pos in open_positions:
            mid = strangle_mid_per_share(S, T, pos["Kc"], pos["Kp"], r, sigma)
            debit = debit_fill(mid) * CONTRACT_MULT * contracts
            pnl = pos["credit_$"] - debit - per_trade_fees()

            reason = None
            if use_tp and pnl >= tp_level(pos["credit_$"]):
                reason = f"TP hit ({tp_pct}% credit)"
            if use_sl and pnl <= sl_level(pos["credit_$"]):
                reason = f"SL hit ({sl_pct}% credit)"
            if must_force_exit:
                reason = "Forced exit"

            if reason is not None:
                cum_pnl += pnl
                trades.append({
                    "Ticker": ticker,
                    "Entry Time (NY)": pos["entry_time_ny"],
                    "Exit Time (NY)": ts_ny,
                    "Entry S": round(pos["entry_S"], 4),
                    "Exit S": round(S, 4),
                    "Call Strike": pos["Kc"],
                    "Put Strike": pos["Kp"],
                    "Credit ($)": round(pos["credit_$"], 2),
                    "Debit ($)": round(debit, 2),
                    "PnL ($)": round(pnl, 2),
                    "Exit Reason": reason,
                })
            else:
                keep.append(pos)
        open_positions = keep

    # entries (continuous re-entry)
    if can_enter:
        if is_intraday and (not overlap_trades) and open_positions:
            pass
        else:
            if len(open_positions) < int(max_open_trades):
                Kc = round_strike(S * (1 + otm_pct / 100.0), strike_round)
                Kp = round_strike(S * (1 - otm_pct / 100.0), strike_round)

                mid = strangle_mid_per_share(S, T, Kc, Kp, r, sigma)
                credit = credit_fill(mid) * CONTRACT_MULT * contracts

                open_positions.append({
                    "NY_date": day,
                    "entry_time_ny": ts_ny,
                    "entry_S": S,
                    "Kc": Kc,
                    "Kp": Kp,
                    "credit_$": float(credit),
                })

    equity_curve.append((ts_ny, cum_pnl))

equity_df = pd.DataFrame(equity_curve, columns=["DatetimeNY", "CumPnL_$"]).drop_duplicates("DatetimeNY")
trades_df = pd.DataFrame(trades)

# ============================================================
# Display
# ============================================================
st.subheader("Results")
st.metric("Cumulative P&L ($)", round(float(equity_df["CumPnL_$"].iloc[-1]), 2))
st.metric("Trades", int(len(trades_df)))

if not trades_df.empty:
    st.metric("Win rate", f"{(trades_df['PnL ($)'] > 0).mean()*100:.1f}%")
    st.metric("Avg PnL / trade ($)", f"{trades_df['PnL ($)'].mean():.2f}")
    st.metric("Worst trade ($)", f"{trades_df['PnL ($)'].min():.2f}")

st.subheader("Equity curve")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(equity_df["DatetimeNY"], equity_df["CumPnL_$"])
ax.set_title("Cumulative P&L ($)")
ax.set_xlabel("Time (NY)")
ax.set_ylabel("PnL ($)")
ax.grid(True)
st.pyplot(fig)

st.subheader("Underlying")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(raw["DatetimeNY"], raw["Close"])
ax.set_title(f"{ticker} Close ({interval})")
ax.set_xlabel("Time (NY)")
ax.set_ylabel("Price")
ax.grid(True)
st.pyplot(fig)

st.subheader("Trades")
if trades_df.empty:
    st.write("No trades executed.")
else:
    show = trades_df.copy()
    show["Entry Time (NY)"] = pd.to_datetime(show["Entry Time (NY)"]).dt.strftime("%Y-%m-%d %H:%M:%S%z")
    show["Exit Time (NY)"] = pd.to_datetime(show["Exit Time (NY)"]).dt.strftime("%Y-%m-%d %H:%M:%S%z")
    st.dataframe(show, use_container_width=True)

with st.expander("Debug: columns"):
    st.write(raw.columns.tolist())
