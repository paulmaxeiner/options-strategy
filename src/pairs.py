import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import plotly.graph_objects as go


# -------------------------
# Black-Scholes helpers
# -------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


# -------------------------
# Data + Vol
# -------------------------
def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df.rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def compute_rolling_vol(df: pd.DataFrame, window: int) -> pd.Series:
    close = df["Close"].astype(float)
    rets = np.log(close / close.shift(1))
    vol_daily = rets.rolling(window).std()
    vol_annual = vol_daily * math.sqrt(252)
    return vol_annual


# -------------------------
# Strategy backtest (N-trading-day cycles)
# -------------------------
@dataclass
class BacktestParams:
    ticker: str
    start: str
    end: str
    otm_pct: float
    dte_days: int           # trading days until close
    vol_window: int
    iv_mult: float
    r: float
    fee_per_contract_leg: float  # fee per leg (sell + buy)
    shares: int = 100


def build_cycles(df: pd.DataFrame, dte_days: int) -> pd.DataFrame:
    """
    Build non-overlapping cycles so only one call is open at a time.
    Entry: day i (Open)
    Exit: day i + dte_days - 1 (Close)
    Next entry: day after exit.
    """
    idx = df.index.to_list()
    n = len(idx)
    i = 0
    rows = []
    while i + dte_days - 1 < n:
        entry_dt = idx[i]
        exit_dt = idx[i + dte_days - 1]
        rows.append({
            "entry_date": entry_dt,
            "exit_date": exit_dt,
            "entry_open": float(df.loc[entry_dt, "Open"]),
            "exit_close": float(df.loc[exit_dt, "Close"]),
            "days_held": dte_days
        })
        i = i + dte_days  # next cycle starts after exit
    return pd.DataFrame(rows)


def run_backtest(df: pd.DataFrame, params: BacktestParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      equity: daily equity curve for CoveredCall and BuyHold
      trades: per-cycle option details + per-cycle returns
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    vol = compute_rolling_vol(df, params.vol_window)
    cycles = build_cycles(df, params.dte_days)
    if cycles.empty:
        return pd.DataFrame(), pd.DataFrame()

    close = df["Close"].astype(float)
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()

    # Track cash from option trades; shares held constant at 100
    cash_steps = pd.Series(0.0, index=df.index)

    trade_records = []
    for _, tr in cycles.iterrows():
        entry_date = pd.to_datetime(tr["entry_date"])
        exit_date = pd.to_datetime(tr["exit_date"])
        S0 = float(tr["entry_open"])
        ST = float(tr["exit_close"])

        # vol estimate at entry
        sigma = float(vol.loc[entry_date]) if entry_date in vol.index else float("nan")
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 0.25  # fallback early sample
        sigma_iv = sigma * params.iv_mult

        K = S0 * (1.0 + params.otm_pct)
        T = params.dte_days / 252.0  # time to expiration in years

        premium_sold = bs_call_price(S=S0, K=K, T=T, r=params.r, sigma=sigma_iv, q=0.0)

        # Conservative close: intrinsic at exit close
        close_cost = max(ST - K, 0.0)

        fees = 2.0 * params.fee_per_contract_leg
        option_pnl = premium_sold - close_cost - fees

        # realize on exit_date
        if exit_date in cash_steps.index:
            cash_steps.loc[exit_date] += option_pnl

        # per-cycle returns (covered call vs buyhold)
        # start/end portfolio values measured at entry open and exit close
        cc_start = params.shares * S0
        cc_end = params.shares * ST + option_pnl  # option pnl realized within the cycle
        cc_ret = cc_end / cc_start - 1.0

        bh_start = params.shares * S0
        bh_end = params.shares * ST
        bh_ret = bh_end / bh_start - 1.0

        trade_records.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "S_entry_open": S0,
            "S_exit_close": ST,
            "K": K,
            "days_held": params.dte_days,
            "sigma_realized": sigma,
            "sigma_iv_used": sigma_iv,
            "premium_sold": premium_sold,
            "close_cost_intrinsic": close_cost,
            "fees": fees,
            "option_pnl": option_pnl,
            "cc_cycle_return": cc_ret,
            "bh_cycle_return": bh_ret,
        })

    trades = pd.DataFrame(trade_records)
    trades["cum_option_pnl"] = trades["option_pnl"].cumsum()

    cash_cc_daily = cash_steps.cumsum()
    equity = pd.DataFrame(index=df.index)
    equity["BuyHold"] = params.shares * close
    equity["CoveredCall"] = params.shares * close + cash_cc_daily
    equity["CC_Cash"] = cash_cc_daily

    equity["BuyHold_ret"] = equity["BuyHold"].pct_change()
    equity["CoveredCall_ret"] = equity["CoveredCall"].pct_change()
    return equity, trades


def perf_stats(series: pd.Series) -> dict:
    series = series.dropna()
    if series.empty:
        return {}
    rets = series.pct_change().dropna()
    if rets.empty:
        return {}

    total_return = series.iloc[-1] / series.iloc[0] - 1.0
    ann_return = (1.0 + total_return) ** (252.0 / max(len(series) - 1, 1)) - 1.0
    ann_vol = rets.std() * math.sqrt(252)
    sharpe = np.nan if ann_vol == 0 else ann_return / ann_vol

    peak = series.cummax()
    dd = (series / peak) - 1.0
    max_dd = dd.min()

    return {
        "Total return": total_return,
        "Annualized return": ann_return,
        "Annualized vol": ann_vol,
        "Sharpe (rf≈0)": sharpe,
        "Max drawdown": max_dd,
    }


# -------------------------
# 3D Surface / Scatter Data
# -------------------------
def compute_3d_points(df: pd.DataFrame, base_params: BacktestParams, otm_grid: np.ndarray) -> pd.DataFrame:
    """
    For each OTM in otm_grid, run cycles and record (OTM, exit_date, cc_cycle_return).
    Returns a long dataframe with columns: otm, date, ret, cumret
    """
    points = []
    for otm in otm_grid:
        p = BacktestParams(
            ticker=base_params.ticker,
            start=base_params.start,
            end=base_params.end,
            otm_pct=float(otm),
            dte_days=base_params.dte_days,
            vol_window=base_params.vol_window,
            iv_mult=base_params.iv_mult,
            r=base_params.r,
            fee_per_contract_leg=base_params.fee_per_contract_leg,
            shares=base_params.shares,
        )
        _, trades = run_backtest(df, p)
        if trades.empty:
            continue
        
        # Calculate cumulative return for this OTM level
        cumret = 1.0
        for _, row in trades.iterrows():
            ret = float(row["cc_cycle_return"])
            cumret *= (1 + ret)
            points.append({
                "otm": float(otm),
                "date": pd.to_datetime(row["exit_date"]),
                "ret": ret,
                "cumret": cumret - 1.0,  # Convert back to return percentage
            })
    return pd.DataFrame(points)


def plot_3d(points: pd.DataFrame) -> go.Figure:
    """
    3D surface: X=OTM, Y=Date, Z=Cumulative Return
    Converts scattered points to a gridded surface.
    """
    # Pivot data to create a grid using cumulative return
    # Group by OTM and Date, taking mean if multiple values
    pivot_df = points.pivot_table(
        index='date',
        columns='otm',
        values='cumret',
        aggfunc='mean'
    )
    
    # Fill NaN values with mean to create continuous surface
    z_grid = pivot_df.to_numpy()
    z_filled = np.nan_to_num(z_grid, nan=np.nanmean(z_grid))
    
    # Create meshgrid for X (OTM) and Y (Date)
    x_vals = pivot_df.columns.to_numpy()  # OTM values
    y_vals = pivot_df.index.to_numpy()    # Date values
    
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x_vals,
            y=y_vals,
            z=z_filled,
            colorscale='Viridis',
            hovertemplate="OTM=%{x:.2%}<br>Date=%{y|%Y-%m-%d}<br>Cumulative Return=%{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="OTM %",
            yaxis_title="Date",
            zaxis_title="Cumulative % Return",
        ),
        height=650,
    )
    return fig


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Covered Call Backtester (Adjustable Frequency + 3D)", layout="wide")
st.title("Covered Call Backtester (yfinance + Streamlit)")
st.caption(
    "Sell a %OTM call at entry day open, close at expiration day close (intrinsic), repeat with non-overlapping cycles. "
    "Option premium is simulated using Black–Scholes with rolling realized vol × IV multiplier."
)

with st.sidebar:
    st.header("Inputs")

    ticker = st.text_input("Ticker", value="SPY")
    start = st.date_input("Start date", value=pd.to_datetime("2015-01-01"))
    end = st.date_input("End date", value=pd.to_datetime("today"))

    otm_pct = st.slider("Primary OTM% (for main backtest)", 0.0, 0.30, 0.05, 0.005)

    dte_days = st.slider(
        "Frequency / Time to Expiration (trading days)",
        min_value=2,
        max_value=40,
        value=5,
        step=1,
        help="If set to 10, each call is held for ~10 trading days. Only one call open at a time."
    )

    vol_window = st.slider("Volatility lookback (trading days)", 5, 120, 20, 5)
    iv_mult = st.slider("IV multiplier (IV = realized * multiplier)", 0.5, 2.5, 1.2, 0.05)

    r = st.number_input("Risk-free rate (annual, decimal)", min_value=0.0, max_value=0.10, value=0.02, step=0.005)
    fee = st.number_input("Fee/slippage per contract leg ($)", min_value=0.0, max_value=50.0, value=0.50, step=0.25)

    st.divider()
    st.subheader("3D plot settings")
    otm_min, otm_max = st.slider("OTM% range for 3D plot", 0.0, 0.50, (0.0, 0.20), 0.01)
    otm_step = st.select_slider("OTM step size", options=[0.005, 0.01, 0.02, 0.025, 0.05], value=0.01)

    run_btn = st.button("Run backtest", type="primary")


if run_btn:
    params = BacktestParams(
        ticker=ticker.strip().upper(),
        start=str(start),
        end=str(end),
        otm_pct=float(otm_pct),
        dte_days=int(dte_days),
        vol_window=int(vol_window),
        iv_mult=float(iv_mult),
        r=float(r),
        fee_per_contract_leg=float(fee),
        shares=100,
    )

    with st.spinner("Downloading price data..."):
        df = load_prices(params.ticker, params.start, params.end)

    if df.empty:
        st.error("No data returned. Check ticker and date range.")
        st.stop()

    equity, trades = run_backtest(df, params)
    if equity.empty or trades.empty:
        st.error("Not enough data to build cycles for the selected DTE. Try a longer date range or smaller DTE.")
        st.stop()

    # Metrics
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Performance (Covered Call)")
        stats_cc = perf_stats(equity["CoveredCall"])
        for k, v in stats_cc.items():
            st.metric(k, f"{v:.2%}" if ("return" in k.lower() or "vol" in k.lower() or "drawdown" in k.lower()) else f"{v:.2f}")

    with c2:
        st.subheader("Performance (Buy & Hold)")
        stats_bh = perf_stats(equity["BuyHold"])
        for k, v in stats_bh.items():
            st.metric(k, f"{v:.2%}" if ("return" in k.lower() or "vol" in k.lower() or "drawdown" in k.lower()) else f"{v:.2f}")

    st.divider()

    # Curves
    st.subheader("Equity Curves")
    st.line_chart(equity[["CoveredCall", "BuyHold"]])

    st.subheader("Covered Call Cash From Options (Cumulative)")
    st.line_chart(equity[["CC_Cash"]])

    st.divider()

    # Trades table
    st.subheader("Cycle Trades (Option Details)")
    st.dataframe(trades, use_container_width=True)

    # Downloads
    trades_csv = trades.to_csv(index=False).encode("utf-8")
    equity_csv = equity.to_csv(index=True).encode("utf-8")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download trades CSV", trades_csv, file_name=f"{params.ticker}_covered_call_trades.csv", mime="text/csv")
    with d2:
        st.download_button("Download equity CSV", equity_csv, file_name=f"{params.ticker}_covered_call_equity.csv", mime="text/csv")

    st.divider()

    # 3D plot
    st.subheader("3D Plot: OTM%, Date, Cumulative %Return (Covered Call)")
    otm_grid = np.arange(otm_min, otm_max + 1e-12, float(otm_step))
    if len(otm_grid) > 200:
        st.warning("Your OTM grid is very large; consider increasing step size to speed things up.")

    with st.spinner("Computing 3D points across OTM grid..."):
        points = compute_3d_points(df, params, otm_grid)

    if points.empty:
        st.warning("No 3D points generated (not enough cycles). Try widening the date range or reducing DTE.")
    else:
        fig = plot_3d(points)
        st.plotly_chart(fig, use_container_width=True)

        # Quick summary heat table (optional helpful view)
        st.caption("Tip: the 3D surface is easiest to interpret if you rotate and zoom. Hover over the surface for exact values.")

    st.info(
        "Modeling notes:\n"
        "- Premium is simulated using Black–Scholes with rolling realized vol × IV multiplier.\n"
        "- Close price is approximated by intrinsic at expiration close (conservative; ignores time value).\n"
        "- Cycles are non-overlapping: only one contract open at a time.\n"
        "- This ignores early assignment, bid/ask spreads beyond your fee input, and volatility skew."
    )

else:
    st.write("Choose inputs on the left, then click **Run backtest**.")
