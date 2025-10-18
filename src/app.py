import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.stats import norm
import scipy.stats as scistats
import streamlit as st
import pytz
from datetime import datetime, date, timedelta

r = 0.03
sigma = 0.5
TRADING_DAYS = 252
MIN_PER_DAY = 6.5 * 60
MIN_PER_YEAR = TRADING_DAYS * MIN_PER_DAY

def black_scholes_call(S, T, K):
    return black_scholes_call_sigma(S, T, K, sigma)

def black_scholes_call_sigma(S, T, K, sigma_use):
    S = float(S); T = float(T)
    if T <= 0:
        return max(S - K, 0.0)
    if sigma_use <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_use**2) * T) / (sigma_use * np.sqrt(T))
    d2 = d1 - sigma_use * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def minutes_to_close_series(dts):
    ny = pytz.timezone("America/New_York")
    s = pd.to_datetime(dts, utc=True)
    dts_ny = s.dt.tz_convert(ny)
    four_pm = dts_ny.dt.normalize() + pd.Timedelta(hours=16)
    mins = (four_pm - dts_ny).dt.total_seconds() / 60
    return mins / 60 / 24 / 365

def rank_cdf(x):
    return (x.rank(method="average") - 0.5) / len(x)

def inverse_ecdf(values, u):
    v_sorted = np.sort(values)
    n = len(v_sorted)
    q = (np.arange(1, n + 1) - 0.5) / n
    return np.interp(u, q, v_sorted)

def conditional_sigma_from_vix(spy_returns_window, vix_returns_window, vix_ret_current, n_samples=2000):
    if len(spy_returns_window) < 50 or len(vix_returns_window) < 50:
        return np.nan
    df = pd.DataFrame({"spy": spy_returns_window, "vix": vix_returns_window}).dropna()
    if len(df) < 50:
        return np.nan
    U_spy = rank_cdf(df["spy"])
    U_vix = rank_cdf(df["vix"])
    Z_spy = scistats.norm.ppf(U_spy.clip(1e-6, 1 - 1e-6))
    Z_vix = scistats.norm.ppf(U_vix.clip(1e-6, 1 - 1e-6))
    rho = np.corrcoef(Z_spy, Z_vix)[0, 1]
    rho = np.clip(rho, -0.999, 0.999)
    u2_curr = (np.sum(df["vix"].values <= vix_ret_current) + 0.5) / (len(df) + 1)
    u2_curr = float(np.clip(u2_curr, 1e-6, 1 - 1e-6))
    z2_curr = scistats.norm.ppf(u2_curr)
    mu_c = rho * z2_curr
    var_c = 1 - rho**2
    if var_c <= 0:
        return np.nan
    z1_samples = np.random.normal(loc=mu_c, scale=np.sqrt(var_c), size=n_samples)
    u1_samples = scistats.norm.cdf(z1_samples)
    spy_sim_returns = inverse_ecdf(df["spy"].values, u1_samples)
    return float(np.std(spy_sim_returns, ddof=1))

def annualize_minute_sigma(sig_minute):
    if np.isnan(sig_minute) or sig_minute <= 0:
        return np.nan
    return float(sig_minute * np.sqrt(MIN_PER_YEAR))

def load_intraday_session(session_date: date):
    ny = pytz.timezone("America/New_York")
    start_ny = datetime.combine(session_date, datetime.min.time()).replace(tzinfo=ny)
    end_ny = start_ny + timedelta(days=1)
    start_utc = start_ny.astimezone(pytz.UTC)
    end_utc = end_ny.astimezone(pytz.UTC)
    spy = yf.download("SPY", start=start_utc, end=end_utc, interval="1m", prepost=False, progress=False, auto_adjust=True)
    vix = yf.download("^VIX", start=start_utc, end=end_utc, interval="1m", prepost=False, progress=False, auto_adjust=True)
    if spy.empty or vix.empty:
        return None, None
    return spy.dropna(), vix.dropna()

def flatten_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [' '.join([str(x) for x in tup if str(x) != '']).strip() for tup in df.columns]
    return df

def find_col(df, include_any, exclude_any=None):
    cols = []
    for c in df.columns:
        cl = c.lower()
        if all(k.lower() in cl for k in include_any):
            if not exclude_any or all(ex.lower() not in cl for ex in exclude_any):
                cols.append(c)
    return cols

def squeeze_1d(obj):
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            obj = obj.iloc[:, 0]
        else:
            obj = obj.squeeze(axis=1)
    return pd.to_numeric(pd.Series(obj), errors="coerce")

st.set_page_config(page_title="0DTE BS + VIX Copula (Single Day)", layout="wide")
st.title("0DTE BS + VIX Copula (Single Day)")

default_date = (datetime.now(pytz.timezone("America/New_York")) - timedelta(days=1)).date()
pick_date = st.sidebar.date_input("Session Date (NY)", value=default_date, min_value=default_date - timedelta(days=60), max_value=default_date)

spy_raw, vix_raw = load_intraday_session(pick_date)
if spy_raw is None or vix_raw is None or spy_raw.empty or vix_raw.empty:
    st.error("No 1m data for the selected date. Try another trading day.")
    st.stop()

spy_raw = flatten_cols(spy_raw)
vix_raw = flatten_cols(vix_raw)

df = spy_raw.copy()
vix_close_name_map = {c: "VIX_Close" for c in vix_raw.columns if "close" in c.lower()}
df = df.join(vix_raw[[c for c in vix_raw.columns if "close" in c.lower()]].rename(columns=vix_close_name_map), how="inner")
df.reset_index(inplace=True)
if "Datetime" not in df.columns:
    if "index" in df.columns:
        df.rename(columns={"index": "Datetime"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "Datetime"}, inplace=True)
df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
df["Time"] = df["Datetime"].dt.tz_convert("America/New_York").dt.time

close_candidates = (find_col(df, ["close"], exclude_any=["vix"]) or find_col(df, ["close"]))
open_candidates  = (find_col(df, ["open"],  exclude_any=["vix"]) or find_col(df, ["open"]))
high_candidates  = (find_col(df, ["high"],  exclude_any=["vix"]) or find_col(df, ["high"]))
low_candidates   = (find_col(df, ["low"],   exclude_any=["vix"]) or find_col(df, ["low"]))
vol_candidates   = (find_col(df, ["volume"],exclude_any=["vix"]) or find_col(df, ["volume"]))
vix_candidates   = (find_col(df, ["vix", "close"]) or find_col(df, ["vix_close"]) or [c for c in df.columns if c.lower() == "vix_close"])

missing = []
if not close_candidates: missing.append("SPY Close")
if not open_candidates:  missing.append("SPY Open")
if not high_candidates:  missing.append("SPY High")
if not low_candidates:   missing.append("SPY Low")
if not vol_candidates:   missing.append("SPY Volume")
if not vix_candidates:   missing.append("VIX Close")
if missing:
    st.error(f"Could not locate required columns: {', '.join(missing)}. Found columns: {list(df.columns)}")
    st.stop()

close_col = close_candidates[0]
open_col  = open_candidates[0]
high_col  = high_candidates[0]
low_col   = low_candidates[0]
vol_col   = vol_candidates[0]
vix_col   = vix_candidates[0]

close_series = squeeze_1d(df[close_col])
open_series  = squeeze_1d(df[open_col])
high_series  = squeeze_1d(df[high_col])
low_series   = squeeze_1d(df[low_col])
vol_series   = squeeze_1d(df[vol_col])
vix_series   = squeeze_1d(df[vix_col])

spy = pd.DataFrame({
    "Datetime": df["Datetime"],
    "Time": df["Time"],
    "Close": close_series,
    "Open": open_series,
    "High": high_series,
    "Low": low_series,
    "Volume": vol_series,
})
spy["Return"] = spy["Close"].pct_change()

vix = pd.DataFrame({
    "Datetime": df["Datetime"],
    "Close": vix_series,
})
vix["Return"] = vix["Close"].pct_change()

spy["T"] = minutes_to_close_series(spy["Datetime"])

st.caption(f"Session: {pick_date} (NY)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(spy["Close"])
ax.set_title("SPY Close")
st.pyplot(fig)

atm_price = int(round(spy["Close"].iloc[-1]))
st.write("ATM Price:", atm_price)

call_prices = pd.DataFrame({
    atm_price + x: spy.apply(lambda row: black_scholes_call(row["Close"], row["T"], atm_price + x), axis=1)
    for x in range(-10, 10)
})
call_returns = pd.DataFrame({
    atm_price + x: (call_prices[atm_price + x].pct_change() + 1).cumprod() - 1
    for x in range(-10, 10)
})

ROLL_MIN = 120
N_SAMPLES = 1500
sig_minute_cond_list = []
sig_annual_cond_list = []

for i in range(len(spy)):
    start = max(0, i - ROLL_MIN)
    spy_win = spy["Return"].iloc[start:i].dropna()
    vix_win = vix["Return"].iloc[start:i].dropna()
    if i == 0 or len(spy_win) < 50 or len(vix_win) < 50 or pd.isna(vix["Return"].iloc[i]):
        sig_minute_cond_list.append(np.nan)
        sig_annual_cond_list.append(np.nan)
        continue
    vix_ret_now = float(vix["Return"].iloc[i])
    sig_minute = conditional_sigma_from_vix(spy_win, vix_win, vix_ret_now, n_samples=N_SAMPLES)
    sig_annual = annualize_minute_sigma(sig_minute)
    sig_minute_cond_list.append(sig_minute)
    sig_annual_cond_list.append(sig_annual)

spy["sigma_minute_copula"] = sig_minute_cond_list
spy["sigma_annual_copula"] = sig_annual_cond_list

def bs_with_copula_sigma(row, K):
    s = float(row["Close"]); T = float(row["T"])
    sig = row["sigma_annual_copula"]
    if np.isnan(sig) or sig <= 0:
        sig = sigma
    return black_scholes_call_sigma(s, T, K, sig)

call_prices_copula = pd.DataFrame({
    atm_price + x: spy.apply(lambda row: bs_with_copula_sigma(row, atm_price + x), axis=1)
    for x in range(-10, 10)
})
call_returns_copula = pd.DataFrame({
    atm_price + x: (call_prices_copula[atm_price + x].pct_change() + 1).cumprod() - 1
    for x in range(-10, 10)
})

st.subheader("Annualized σ from VIX–SPY Copula")
st.line_chart(spy[["sigma_annual_copula"]])

st.subheader("ATM Call: Fixed σ vs Copula σ")
compare = pd.DataFrame({
    "Fixed σ (ATM)": call_prices[atm_price],
    "Copula σ (ATM)": call_prices_copula[atm_price]
})
st.line_chart(compare)

S0 = spy["Close"].rolling(60, min_periods=1).mean().iloc[-1]
atm_center = int(round(S0))
K_values = np.linspace(atm_center - 20, atm_center + 20, 20)
T_eps = 1e-6
T_values = np.linspace(max(T_eps, spy["T"].min()), max(T_eps, spy["T"].max()), 20)
points = []
for K in K_values:
    for T in T_values:
        C = black_scholes_call(S0, float(T), float(K))
        points.append((K, T, C))
K_arr, T_arr, C_arr = np.array(points).T

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(K_arr, T_arr, C_arr, c=C_arr, cmap=cm.viridis, s=25)
ax.set_xlabel("K")
ax.set_ylabel("T (years)")
ax.set_zlabel("C")
ax.set_title("BS Surface (Fixed σ)")
fig.colorbar(sc, shrink=0.6, aspect=12)
st.pyplot(fig)

sigma_last = spy["sigma_annual_copula"].dropna().iloc[-1] if spy["sigma_annual_copula"].notna().any() else sigma

def bs_surface_with_sigma(S_base, Ks, Ts, sigma_use):
    Z = np.zeros((len(Ts), len(Ks)))
    for i, T in enumerate(Ts):
        for j, K in enumerate(Ks):
            Z[i, j] = black_scholes_call_sigma(S_base, T, K, sigma_use)
    return Z

Xg, Yg = np.meshgrid(np.unique(K_arr), np.unique(T_arr))
Zc = bs_surface_with_sigma(S0, np.unique(K_arr), np.unique(T_arr), sigma_last)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(Xg, Yg, Zc, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter("{x:.02f}")
ax.set_title("BS Surface (Copula σ)")
fig.colorbar(surf, shrink=0.5, aspect=5)
st.pyplot(fig)

st.subheader("Call Price Panels")
st.line_chart(call_prices)
st.line_chart(call_returns)
st.line_chart(call_prices_copula)
st.line_chart(call_returns_copula)

rsi_df = pd.DataFrame({"Close": spy["Close"]})
rsi_df["RSI"] = calculate_rsi(rsi_df["Close"])
def rsi_signal(v):
    if v < 30: return 1
    elif v > 70: return -1
    else: return 0
rsi_df["Signal"] = rsi_df["RSI"].apply(rsi_signal)
rsi_df["Position"] = rsi_df["Signal"].replace(to_replace=0, method="ffill").shift()
rsi_df["Return"] = rsi_df["Close"].pct_change()
rsi_df["Strategy_Return"] = rsi_df["Position"] * rsi_df["Return"]
rsi_df["Cumulative_Market"] = (1 + rsi_df["Return"]).cumprod()
rsi_df["Cumulative_Strategy"] = (1 + rsi_df["Strategy_Return"]).cumprod()

st.subheader("RSI Strategy Backtest")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(rsi_df["Cumulative_Market"], label="Market")
ax.plot(rsi_df["Cumulative_Strategy"], label="RSI Strategy")
ax.legend()
st.pyplot(fig)
