# app.py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date, datetime, timedelta, time as dtime
from utilities import getStockData, C

#ui
st.set_page_config(page_title="RSI Call Strategy (0-DTE, BS pricing)", layout="wide")
st.title("RSI Strategy on SPY Calls (Black-Scholes theoretical pricing)")

with st.sidebar:
    st.subheader("Data")
    trading_day = st.date_input("Trading day (ET)", value=date.today() - timedelta(days=1))
    interval = st.selectbox("Interval", ["1m", "2m", "5m"], index=0)

    st.subheader("RSI & Signals")
    rsi_period = st.slider("RSI period", 5, 50, 14, 1)
    buy_threshold = st.slider("Buy when RSI crosses above", 5, 50, 30, 1)
    sell_threshold = st.slider("Sell when RSI crosses below", 50, 95, 70, 1)

    st.subheader("Option Pricing")
    r_rate = st.number_input("Risk-free rate (annual, dec.)", value=0.03, step=0.005, format="%.3f")
    use_dynamic_sigma = st.checkbox("Use rolling (intraday) sigma", value=True)
    sigma_const = st.number_input("Fallback sigma (annualized)", value=0.35, step=0.05, format="%.2f")
    sigma_window = st.slider("Sigma rolling window (minutes)", 5, 120, 30, 1)

    st.subheader("Execution")
    contracts = st.number_input("# of contracts per trade", min_value=1, value=1, step=1)
    fee_per_side = st.number_input("Fee/slippage per side ($)", min_value=0.0, value=2.50, step=0.25)
    run_btn = st.button("Run Backtest")

st.caption("This sim uses **ATM** calls (strike = round(spot at entry)) that **expire today at 4:00 PM ET**. "
           "Pricing uses Black–Scholes with your R and sigma choices.")

# ---------------- helpers ----------------
def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral start
    return rsi

@st.cache_data(show_spinner=False)
def load_data(day: date, interval: str) -> pd.DataFrame:
    """Fetch intraday SPY with 'T' (years to 4pm ET) from utilities.getStockData, filter RTH."""
    start = pd.Timestamp(day).strftime("%Y-%m-%d")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = getStockData("SPY", start=start, end=end, interval=interval)
    # Filter to regular trading hours 9:30–16:00 ET using the pre-made 'Time' col (timezone-aware conversion was done)
    rth = df[(df["Time"] >= dtime(9, 30)) & (df["Time"] <= dtime(16, 0))].copy()
    rth.sort_values("Datetime", inplace=True)
    rth.reset_index(drop=True, inplace=True)
    return rth

def annualize_sigma_from_intraday(returns: pd.Series, interval_str: str, window: int) -> pd.Series:
    """
    Approximate annualization for intraday std:
    trading minutes/year ≈ 252 * 6.5 * 60 = 98,280
    """
    if interval_str.endswith("m"):
        mins_per_bar = int(interval_str[:-1])
    else:
        mins_per_bar = 1
    bars_per_year = (252 * 6.5 * 60) / mins_per_bar
    rolling_std = returns.rolling(window).std()
    sigma_ann = rolling_std * np.sqrt(bars_per_year)
    return sigma_ann

def price_call_series_fixed_strike(S: pd.Series, T_years: pd.Series, K: float, r: float,
                                   sigma_series_or_scalar) -> pd.Series:
    """
    Price a fixed-strike call across a time window. sigma can be scalar or Series aligned with S.
    """
    prices = []
    for i in range(len(S)):
        s_i = float(S.iloc[i])
        t_i = max(0.0, float(T_years.iloc[i]))  # no negative T
        if isinstance(sigma_series_or_scalar, pd.Series):
            sig = float(sigma_series_or_scalar.iloc[i]) if not np.isnan(sigma_series_or_scalar.iloc[i]) else np.nan
            if np.isnan(sig) or sig <= 0:
                sig = sigma_const
        else:
            sig = float(sigma_series_or_scalar)
        # clamp sigma to something sane
        sig = max(0.01, min(sig, 3.0))
        prices.append(C(s_i, K, t_i, r=r, sigma=sig))
    return pd.Series(prices, index=S.index, dtype=float)

def backtest_rsi_calls(df: pd.DataFrame,
                       rsi_period: int,
                       buy_thr: int,
                       sell_thr: int,
                       r: float,
                       sigma_const: float,
                       use_dyn_sigma: bool,
                       sigma_window: int,
                       interval: str,
                       contracts: int,
                       fee_per_side: float):
    """
    Event-driven sim: long 1x (or N) ATM call on RSI cross-up; exit on cross-down or 4pm.
    Prices are theoretical (BS) using your utilities.C. Strike fixed at entry.
    """
    data = df.copy()
    data["RSI"] = compute_rsi(data["Close"], rsi_period)

    if use_dyn_sigma:
        data["sigma_dyn"] = annualize_sigma_from_intraday(data["Return"].fillna(0), interval, sigma_window).fillna(sigma_const)
        data["sigma_dyn"] = data["sigma_dyn"].clip(0.01, 3.0)
    else:
        data["sigma_dyn"] = sigma_const

    # Pre-allocate equity & position tracking
    data["position"] = 0  # 0 or +contracts
    data["equity"] = 0.0
    data["call_mtm"] = np.nan  # mark-to-market option price when in a trade (same strike while open)

    trades = []
    in_pos = False
    K = None
    entry_idx = None
    entry_price = None
    realized_pnl = 0.0

    for i in range(1, len(data)):
        prev_rsi, rsi_now = data["RSI"].iloc[i-1], data["RSI"].iloc[i]
        S_now = data["Close"].iloc[i]
        T_now = data["T"].iloc[i]
        sigma_now = data["sigma_dyn"].iloc[i]

        # entry: cross above buy threshold
        if (not in_pos) and (prev_rsi < buy_thr) and (rsi_now >= buy_thr) and (T_now > 0):
            in_pos = True
            entry_idx = i
            K = round(S_now)  # ATM strike
            entry_price = C(float(S_now), float(K), float(T_now), r=r, sigma=float(sigma_now))
            data.at[data.index[i], "position"] = contracts
            # mark current MTM
            data.at[data.index[i], "call_mtm"] = entry_price

        elif in_pos:
            # mark-to-market valuation with fixed strike K
            mtm = C(float(S_now), float(K), max(0.0, float(T_now)), r=r, sigma=float(sigma_now))
            data.at[data.index[i], "position"] = contracts
            data.at[data.index[i], "call_mtm"] = mtm

            should_exit_signal = (prev_rsi > sell_thr) and (rsi_now <= sell_thr)
            should_exit_eod = (T_now <= 0)

            if should_exit_signal or should_exit_eod or i == len(data) - 1:
                exit_price = mtm
                gross = (exit_price - entry_price) * contracts
                fees = fee_per_side * 2.0
                pnl = gross - fees

                trades.append({
                    "EntryTime": data["Datetime"].iloc[entry_idx],
                    "ExitTime": data["Datetime"].iloc[i],
                    "Strike": K,
                    "EntrySpot": data["Close"].iloc[entry_idx],
                    "ExitSpot": S_now,
                    "EntryCall": entry_price,
                    "ExitCall": exit_price,
                    "Contracts": contracts,
                    "GrossPnL": gross,
                    "Fees": fees,
                    "NetPnL": pnl
                })

                realized_pnl += pnl
                # flat after exit
                in_pos = False
                K = None
                entry_idx = None
                entry_price = None
                data.at[data.index[i], "position"] = 0  # position closed AFTER this bar

        # equity curve (mark-to-market)
        if in_pos and entry_price is not None:
            # include open PnL but not yet exit fees (those are realized on close)
            open_mtm = data["call_mtm"].iloc[i]
            open_pnl = (open_mtm - entry_price) * contracts
            data.at[data.index[i], "equity"] = realized_pnl + open_pnl
        else:
            data.at[data.index[i], "equity"] = realized_pnl

    trades_df = pd.DataFrame(trades)
    return data, trades_df

# ---------------- run ----------------
if run_btn:
    try:
        df = load_data(trading_day, interval)
        if df.empty:
            st.warning("No data returned for that day/interval. Try a recent weekday.")
            st.stop()

        bt_df, trades_df = backtest_rsi_calls(
            df,
            rsi_period=rsi_period,
            buy_thr=buy_threshold,
            sell_thr=sell_threshold,
            r=r_rate,
            sigma_const=sigma_const,
            use_dyn_sigma=use_dynamic_sigma,
            sigma_window=sigma_window,
            interval=interval,
            contracts=contracts,
            fee_per_side=fee_per_side
        )

        # --------- Metrics ---------
        total_pnl = float(trades_df["NetPnL"].sum()) if not trades_df.empty else 0.0
        wins = int((trades_df["NetPnL"] > 0).sum()) if not trades_df.empty else 0
        ntrades = len(trades_df)
        win_rate = (wins / ntrades * 100.0) if ntrades else 0.0
        avg_trade = float(trades_df["NetPnL"].mean()) if ntrades else 0.0
        max_dd = float((bt_df["equity"].cummax() - bt_df["equity"]).max()) if not bt_df.empty else 0.0

        # Trade-return Sharpe (per trade) for a quick feel
        if ntrades:
            trade_rets = trades_df["NetPnL"] / trades_df["EntryCall"].replace(0, np.nan)
            trade_rets = trade_rets.replace([np.inf, -np.inf], np.nan).dropna()
            sharpe = (trade_rets.mean() / trade_rets.std()) * math.sqrt(len(trade_rets)) if trade_rets.std() > 0 else np.nan
        else:
            sharpe = np.nan

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Net PnL ($)", f"{total_pnl:,.2f}")
        c2.metric("Trades", f"{ntrades}")
        c3.metric("Win rate", f"{win_rate:.1f}%")
        c4.metric("Avg trade ($)", f"{avg_trade:,.2f}")
        c5.metric("Max Drawdown ($)", f"{max_dd:,.2f}")

        # --------- Charts ---------
        st.subheader("Equity Curve")
        fig_eq, ax_eq = plt.subplots(figsize=(10, 4))
        ax_eq.plot(bt_df["Datetime"], bt_df["equity"])
        ax_eq.set_xlabel("Time (ET)")
        ax_eq.set_ylabel("Equity ($)")
        ax_eq.grid(True, alpha=0.3)
        st.pyplot(fig_eq)

        st.subheader("SPY Price & RSI (entries/exits)")
        fig, (ax_p, ax_r) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax_p.plot(bt_df["Datetime"], bt_df["Close"], label="SPY Close")
        # mark entries & exits from trades_df
        if not trades_df.empty:
            ax_p.scatter(trades_df["EntryTime"], trades_df["EntrySpot"], marker="^", s=60, label="Buy Call", zorder=3)
            ax_p.scatter(trades_df["ExitTime"], trades_df["ExitSpot"], marker="v", s=60, label="Sell Call", zorder=3)
        ax_p.set_ylabel("Price")
        ax_p.legend()
        ax_p.grid(True, alpha=0.3)

        ax_r.plot(bt_df["Datetime"], bt_df["RSI"], label="RSI")
        ax_r.axhline(buy_threshold, linestyle="--", linewidth=1)
        ax_r.axhline(sell_threshold, linestyle="--", linewidth=1)
        ax_r.set_ylabel("RSI")
        ax_r.set_xlabel("Time (ET)")
        ax_r.grid(True, alpha=0.3)
        st.pyplot(fig)

        # --------- Trades Table ---------
        st.subheader("Trades")
        if trades_df.empty:
            st.info("No trades.")
        else:
            show_cols = ["EntryTime","ExitTime","Strike","EntrySpot","ExitSpot","EntryCall","ExitCall","Contracts","GrossPnL","Fees","NetPnL"]
            st.dataframe(trades_df[show_cols].style.format({
                "EntrySpot": "{:.2f}", "ExitSpot": "{:.2f}",
                "EntryCall": "{:.2f}", "ExitCall": "{:.2f}",
                "GrossPnL": "{:.2f}", "Fees": "{:.2f}", "NetPnL": "{:.2f}"
            }), use_container_width=True)

    except Exception as e:
        st.error(f"Backtest failed: {e}")

