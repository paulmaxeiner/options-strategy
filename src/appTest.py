# app.py
import math
import numpy as np
import pandas as pd
from matplotlib import cm
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
    fee_per_side = st.number_input("Fee/slippage per side ($)", min_value=0.0, value=0.03, step=0.1)
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
            entry_price = max(0.01, C(float(S_now), float(K), float(T_now), r=r, sigma=float(sigma_now)))
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
        total_return = (bt_df["equity"].iloc[-1] - bt_df["equity"].iloc[0]) if not bt_df.empty else 0.0

        # Trade-return Sharpe (per trade) for a quick feel
        if ntrades:
            trade_rets = trades_df["NetPnL"] / trades_df["EntryCall"].replace(0, np.nan)
            trade_rets = trade_rets.replace([np.inf, -np.inf], np.nan).dropna()
            sharpe = (trade_rets.mean() / trade_rets.std()) * math.sqrt(len(trade_rets)) if trade_rets.std() > 0 else np.nan
        else:
            sharpe = np.nan

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Net PnL ($)", f"{total_pnl:,.2f}")
        c2.metric("Trades", f"{ntrades}")
        c3.metric("Win rate", f"{win_rate:.1f}%")
        c4.metric("Avg trade ($)", f"{avg_trade:,.2f}")
        c5.metric("Max Drawdown ($)", f"{max_dd:,.2f}")
        c6.metric("Return", f"{total_return*100:.2f}%")

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

# ==== New: Single-Day RSI Threshold Surface (3D) =============================
# ==== Fixed: 30-Day Aggregated RSI Threshold Surface (3D) ====================
st.markdown("---")
st.header("RSI Threshold Surface — Aggregated Over Last 30 Market Days")

with st.expander("Configure grid"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        buy_min = st.number_input("Buy RSI min", 5, 95, 20, 1, key="agg_buy_min")
        buy_max = st.number_input("Buy RSI max", 5, 95, 40, 1, key="agg_buy_max")
        buy_step = st.number_input("Buy RSI step", 1, 20, 5, 1, key="agg_buy_step")
    with col2:
        sell_min = st.number_input("Sell RSI min", 5, 95, 60, 1, key="agg_sell_min")
        sell_max = st.number_input("Sell RSI max", 5, 95, 90, 1, key="agg_sell_max")
        sell_step = st.number_input("Sell RSI step", 1, 20, 5, 1, key="agg_sell_step")
    with col3:
        n_days = st.number_input("# market days (max 30)", 5, 30, 30, 1, key="agg_n_days")
    with col4:
        show_top_table = st.checkbox("Show top combos table", value=True, key="agg_show_top")

run_surface = st.button("Run Grid Across Last N Market Days", key="agg_run_surface")

# ... rest of the code unchanged ...

@st.cache_data(show_spinner=False)
def _surface_for_day(
    the_day: date,
    interval: str,
    buy_vals: list[int],
    sell_vals: list[int],
    r_rate: float,
    sigma_const: float,
    use_dynamic_sigma: bool,
    sigma_window: int,
    contracts: int,
    fee_per_side: float,
    rsi_period: int
):
    # Load day once
    df = load_data(the_day, interval)
    if df.empty:
        return None, None, None, None

    # Pre-allocate result table
    records = []
    # Compute total NetPnL for each (buy, sell)
    for b in buy_vals:
        for s in sell_vals:
            if b >= s:           # invalid regime (won't logically trigger exit); skip
                pnl = np.nan
                ntrades = 0
            else:
                bt_df, trades_df = backtest_rsi_calls(
                    df,
                    rsi_period=rsi_period,
                    buy_thr=b,
                    sell_thr=s,
                    r=r_rate,
                    sigma_const=sigma_const,
                    use_dyn_sigma=use_dynamic_sigma,
                    sigma_window=sigma_window,
                    interval=interval,
                    contracts=contracts,
                    fee_per_side=fee_per_side
                )
                pnl = float(trades_df["NetPnL"].sum()) if not trades_df.empty else 0.0
                ntrades = int(len(trades_df))
            records.append({"Buy": b, "Sell": s, "NetPnL": pnl, "Trades": ntrades})

    res_df = pd.DataFrame(records)

    # Build Z matrix aligned to (Buy x Sell) grid
    buy_sorted = sorted(set(res_df["Buy"]))
    sell_sorted = sorted(set(res_df["Sell"]))
    Z = np.full((len(buy_sorted), len(sell_sorted)), np.nan)
    for i, b in enumerate(buy_sorted):
        for j, s in enumerate(sell_sorted):
            val = res_df.loc[(res_df["Buy"] == b) & (res_df["Sell"] == s), "NetPnL"]
            if not val.empty:
                Z[i, j] = float(val.iloc[0])

    # Best combo
    best_row = res_df.loc[res_df["NetPnL"].idxmax()] if res_df["NetPnL"].notna().any() else None
    return res_df, np.array(buy_sorted), np.array(sell_sorted), Z, best_row

if run_surface:
    # Build ranges
    buy_vals  = list(range(buy_min,  buy_max + 1,  buy_step))
    sell_vals = list(range(sell_min, sell_max + 1, sell_step))

    with st.spinner("Sweeping RSI thresholds…"):
        res = _surface_for_day(
            trading_day,
            interval,
            buy_vals,
            sell_vals,
            r_rate,
            sigma_const,
            use_dynamic_sigma,
            sigma_window,
            contracts,
            fee_per_side,
            rsi_period
        )

    if res is None or res[0] is None:
        st.warning("No data for that day. Try a different weekday.")
    else:
        res_df, B, S, Z, best = res

        # Summary of best combo
        if best is not None and np.isfinite(best["NetPnL"]):
            st.success(f"Best combo: Buy {int(best['Buy'])}, Sell {int(best['Sell'])} → NetPnL ${best['NetPnL']:.2f} (Trades: {int(best['Trades'])})")
        else:
            st.info("No profitable combination found (or all invalid).")

        # --- 3D Surface ---
        st.subheader("3D Profit Surface")
        fig3d = plt.figure(figsize=(9, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')

        # Create meshgrid using Buy (X) and Sell (Y)
        X, Y = np.meshgrid(S, B)   # shape matches Z: rows=B (buy), cols=S (sell)
        surf = ax3d.plot_surface(X, Y, Z, cmap=cm.RdYlGn, linewidth=0, antialiased=True)
        ax3d.set_xlabel("Sell RSI")
        ax3d.set_ylabel("Buy RSI")
        ax3d.set_zlabel("Total Profit ($)")
        ax3d.set_title(f"Day: {trading_day.isoformat()} | Interval: {interval}")
        fig3d.colorbar(surf, shrink=0.6, aspect=15)
        st.pyplot(fig3d, clear_figure=True)

        # Optional: show top combos
        if show_table:
            st.subheader("Top Combinations")
            top = res_df.dropna().sort_values("NetPnL", ascending=False).head(15)
            st.dataframe(
                top.style.format({"NetPnL": "{:.2f}"}),
                use_container_width=True
            )
# ==== End Single-Day RSI Threshold Surface ===================================


# ==== New: 30-day auto backtest ==============================================
st.markdown("---")
st.header("30-Day Auto Backtest")

def _day_metrics(
    the_day: date,
    interval: str,
    rsi_period: int,
    buy_threshold: int,
    sell_threshold: int,
    r_rate: float,
    sigma_const: float,
    use_dynamic_sigma: bool,
    sigma_window: int,
    contracts: int,
    fee_per_side: float,
):
    """Run a single-day backtest and aggregate daily metrics."""
    df = load_data(the_day, interval)
    if df.empty:
        return None  # market closed or no data

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

    if trades_df.empty:
        return {
            "Date": pd.to_datetime(the_day),
            "Trades": 0,
            "PremiumSpent": 0.0,
            "PremiumReceived": 0.0,
            "NetPnL": 0.0,
            "ReturnPct": 0.0,
            "WinRatePct": 0.0,
        }

    # Premium spent/received (ex-fees); we compute return on premium put at risk
    premium_spent = float((trades_df["EntryCall"] * trades_df["Contracts"]).sum())
    premium_recv  = float((trades_df["ExitCall"]  * trades_df["Contracts"]).sum())
    net_pnl       = float(trades_df["NetPnL"].sum())
    ntrades       = int(len(trades_df))
    wins          = int((trades_df["NetPnL"] > 0).sum())
    win_rate_pct  = (wins / ntrades) * 100.0 if ntrades else 0.0

    # Define daily "return" as NetPnL divided by total premium spent that day
    denom = premium_spent if premium_spent > 0 else np.nan
    ret_pct = (net_pnl / denom) * 100.0 if denom and not np.isnan(denom) else 0.0

    return {
        "Date": pd.to_datetime(the_day),
        "Trades": ntrades,
        "PremiumSpent": premium_spent,
        "PremiumReceived": premium_recv,
        "NetPnL": net_pnl,
        "ReturnPct": ret_pct,
        "WinRatePct": win_rate_pct,
    }

def _last_n_market_days(n: int, interval: str, lookback_days: int = 90):
    """
    Heuristic: scan back up to `lookback_days` calendar days, keep days
    where load_data() returns RTH bars. Stops after collecting n days.
    """
    days = []
    today = pd.Timestamp.today(tz="America/New_York").date()
    for d in pd.date_range(end=today, periods=lookback_days, freq="D")[::-1]:
        the_day = d.date()
        # quick probe: if load_data not empty, treat as market day
        try:
            df = load_data(the_day, interval)
            if not df.empty:
                days.append(the_day)
        except Exception:
            # ignore fetch errors for broken days
            pass
        if len(days) >= n:
            break
    return days

cA, cB = st.columns([1, 2])
with cA:
    run_batch = st.button("Run Last 30 Market Days")
with cB:
    st.caption("Runs the same RSI/BS call strategy for each of the most recent 30 market days with available data.")

if run_batch:
    with st.spinner("Backtesting 30 market days…"):
        days = _last_n_market_days(20, interval)
        results = []
        for d in days:
            m = _day_metrics(
                the_day=d,
                interval=interval,
                rsi_period=rsi_period,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                r_rate=r_rate,
                sigma_const=sigma_const,
                use_dynamic_sigma=use_dynamic_sigma,
                sigma_window=sigma_window,
                contracts=contracts,
                fee_per_side=fee_per_side,
            )
            if m is not None:
                results.append(m)

        if not results:
            st.warning("No valid days found in the recent window.")
        else:
            res_df = pd.DataFrame(results).sort_values("Date")
            res_df["Date"] = res_df["Date"].dt.date  # cleaner display

            # Summary row
            total_pnl = float(res_df["NetPnL"].sum())
            total_spent = float(res_df["PremiumSpent"].sum())
            # geometric aggregation is tricky; report arithmetic average of daily returns:
            avg_ret = float(res_df["ReturnPct"].mean()) if len(res_df) else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Net PnL ($)", f"{total_pnl:,.2f}")
            c2.metric("Total Premium Spent ($)", f"{total_spent:,.2f}")
            c3.metric("Avg Daily Return (%)", f"{avg_ret:,.2f}")

            st.subheader("Daily Results (Last 30 Market Days)")
            st.dataframe(
                res_df[
                    ["Date", "Trades", "PremiumSpent", "PremiumReceived", "NetPnL", "ReturnPct", "WinRatePct"]
                ].style.format({
                    "PremiumSpent": "{:.2f}",
                    "PremiumReceived": "{:.2f}",
                    "NetPnL": "{:.2f}",
                    "ReturnPct": "{:.2f}",
                    "WinRatePct": "{:.1f}",
                }),
                use_container_width=True
            )

            # Download
            csv = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="rsi_bs_calls_30d_results.csv",
                mime="text/csv",
            )
# ==== End 30-day auto backtest ===============================================

# ==== New: 30-Day Aggregated RSI Threshold Surface (3D) ======================
st.markdown("---")
st.header("RSI Threshold Surface — Aggregated Over Last 30 Market Days")

with st.expander("Configure grid"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        buy_min = st.number_input("Buy RSI min", 5, 95, 20, 1)
        buy_max = st.number_input("Buy RSI max", 5, 95, 40, 1)
        buy_step = st.number_input("Buy RSI step", 1, 20, 5, 1)
    with col2:
        sell_min = st.number_input("Sell RSI min", 5, 95, 60, 1)
        sell_max = st.number_input("Sell RSI max", 5, 95, 90, 1)
        sell_step = st.number_input("Sell RSI step", 1, 20, 5, 1)
    with col3:
        n_days = st.number_input("# market days (max 30)", 5, 30, 30, 1)
    with col4:
        show_top_table = st.checkbox("Show top combos table", value=True)

run_agg_surface = st.button("Run Grid Across Last N Market Days")

@st.cache_data(show_spinner=False)
def _prefetch_days_data(days: list, interval: str):
    """Load once per day; ignore empty days."""
    loaded = []
    for d in days:
        df = load_data(d, interval)
        if not df.empty:
            loaded.append((d, df))
    return loaded

@st.cache_data(show_spinner=False)
def _agg_surface_over_days(
    days: list,
    interval: str,
    buy_vals: list[int],
    sell_vals: list[int],
    r_rate: float,
    sigma_const: float,
    use_dynamic_sigma: bool,
    sigma_window: int,
    contracts: int,
    fee_per_side: float,
    rsi_period: int
):
    """Sweep (buy, sell) grid, sum NetPnL across all provided days."""
    # prefetch day data to avoid repeated downloads
    day_data = _prefetch_days_data(days, interval)
    if not day_data:
        return None, None, None, None, None  # no valid days

    records = []
    # Iterate grid
    for b in buy_vals:
        for s in sell_vals:
            if b >= s:
                # invalid region (won't logically allow a cross-down exit); mark NaN
                records.append({"Buy": b, "Sell": s, "NetPnL": np.nan, "Trades": 0, "DaysUsed": 0})
                continue

            total_pnl = 0.0
            total_trades = 0
            days_used = 0

            for (the_day, df) in day_data:
                try:
                    bt_df, trades_df = backtest_rsi_calls(
                        df,
                        rsi_period=rsi_period,
                        buy_thr=b,
                        sell_thr=s,
                        r=r_rate,
                        sigma_const=sigma_const,
                        use_dyn_sigma=use_dynamic_sigma,
                        sigma_window=sigma_window,
                        interval=interval,
                        contracts=contracts,
                        fee_per_side=fee_per_side
                    )
                    if trades_df is not None:
                        total_pnl += float(trades_df["NetPnL"].sum()) if not trades_df.empty else 0.0
                        total_trades += int(len(trades_df))
                        days_used += 1
                except Exception:
                    # skip broken day gracefully
                    pass

            records.append({
                "Buy": b,
                "Sell": s,
                "NetPnL": total_pnl,
                "Trades": total_trades,
                "DaysUsed": days_used
            })

    res_df = pd.DataFrame(records)

    # Build Z matrix aligned with (Buy x Sell) where rows=Buy, cols=Sell
    buy_sorted = sorted(set(res_df["Buy"]))
    sell_sorted = sorted(set(res_df["Sell"]))
    Z = np.full((len(buy_sorted), len(sell_sorted)), np.nan)
    for i, b in enumerate(buy_sorted):
        for j, s in enumerate(sell_sorted):
            val = res_df.loc[(res_df["Buy"] == b) & (res_df["Sell"] == s), "NetPnL"]
            if not val.empty:
                Z[i, j] = float(val.iloc[0])

    # Best combo across all days
    best_row = res_df.loc[res_df["NetPnL"].idxmax()] if res_df["NetPnL"].notna().any() else None
    return res_df, np.array(buy_sorted), np.array(sell_sorted), Z, best_row

if run_agg_surface:
    # Build ranges (ensure sensible bounds)
    buy_vals  = list(range(buy_min,  buy_max + 1,  buy_step))
    sell_vals = list(range(sell_min, sell_max + 1, sell_step))

    with st.spinner("Sweeping RSI thresholds across market days…"):
        # Use your existing helper to detect last N market days
        days = _last_n_market_days(int(n_days), interval)
        res = _agg_surface_over_days(
            days=days,
            interval=interval,
            buy_vals=buy_vals,
            sell_vals=sell_vals,
            r_rate=r_rate,
            sigma_const=sigma_const,
            use_dynamic_sigma=use_dynamic_sigma,
            sigma_window=sigma_window,
            contracts=contracts,
            fee_per_side=fee_per_side,
            rsi_period=rsi_period
        )

    if res is None or res[0] is None:
        st.warning("No valid market days found in the recent window.")
    else:
        res_df, B, S, Z, best = res

        if best is not None and np.isfinite(best["NetPnL"]):
            st.success(
                f"Best aggregated combo over {int(res_df['DaysUsed'].max())} days: "
                f"Buy {int(best['Buy'])}, Sell {int(best['Sell'])} "
                f"→ NetPnL ${best['NetPnL']:.2f} (Trades: {int(best['Trades'])})"
            )
        else:
            st.info("No profitable combination found (or all invalid).")

        # --- 3D Surface: X=Buy RSI, Y=Sell RSI, Z=Total NetPnL ---
        st.subheader("3D Aggregated Profit Surface (Last N Market Days)")
        fig3d = plt.figure(figsize=(9, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')

        # Meshgrid with X=Buy, Y=Sell; Z shape rows=len(Buy), cols=len(Sell)
        X, Y = np.meshgrid(B, S, indexing="ij")  # X rows follow B, Y cols follow S
        surf = ax3d.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
        ax3d.set_xlabel("Buy RSI")
        ax3d.set_ylabel("Sell RSI")
        ax3d.set_zlabel("Total Profit ($)")
        ax3d.set_title(f"Aggregated over last {len(days)} market days | Interval: {interval}")
        fig3d.colorbar(surf, shrink=0.6, aspect=15)
        st.pyplot(fig3d, clear_figure=True)

        # Optional table of top combos
        if show_top_table:
            st.subheader("Top Combinations (Aggregated)")
            top = res_df.dropna().sort_values("NetPnL", ascending=False).head(20)
            st.dataframe(
                top.style.format({"NetPnL": "{:.2f}"}),
                use_container_width=True
            )

# ==== End 30-Day Aggregated RSI Threshold Surface ============================
