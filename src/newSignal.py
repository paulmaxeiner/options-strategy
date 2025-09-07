# vol_surface_copula_engine.py
# --------------------------------------------------------------
# Intraday Vol Surface + Copula-Based Risk Engine (SPY/VIX)
# Robust version:
#  - UTC-safe timestamps (no tz-naive/aware mismatch)
#  - Skips expired maturities; handles weekends/holidays
#  - Combines calls+puts, liquidity filters
#  - Backfills missing IVs by inverting BS from midprices
#  - SVI slice fitting with bounds & safeguards
#  - Fallback: polynomial smile if SVI fails
#  - Copula-driven MC stress test (Gaussian + Student-t if available)
#  - Streamlit UI (optional)
# --------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from scipy.optimize import minimize, brentq
import yfinance as yf

# Optional UI/plot deps
try:
    import streamlit as st
    STREAMLIT = True
except Exception:
    STREAMLIT = False

try:
    import plotly.graph_objects as go
    PLOTLY = True
except Exception:
    PLOTLY = False

# Copulas
from copulas.multivariate import GaussianMultivariate
try:
    from copulas.multivariate import StudentTMultivariate
    HAS_T_COPULA = True
except Exception:
    HAS_T_COPULA = False


# ---------------------------
# Timezone helpers
# ---------------------------
def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Ensure a pandas Timestamp is tz-aware in UTC."""
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# ---------------------------
# Black–Scholes
# ---------------------------
def bs_call_price(S, K, T, r=0.03, q=0.0, sigma=0.2):
    """Black–Scholes call with continuous dividend yield q."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S*np.exp(-q*T) - K*np.exp(-r*T), 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma*sigma) * T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_call_greeks(S, K, T, r=0.03, q=0.0, sigma=0.2):
    """Returns (delta, gamma, vega, theta, rho) for a call."""
    if T <= 0 or sigma <= 0:
        # Finite-difference fallback near edges
        eps = 1e-4
        base = bs_call_price(S, K, T, r, q, sigma)
        delta = (bs_call_price(S+eps, K, T, r, q, sigma) - base)/eps
        gamma = (bs_call_price(S+eps, K, T, r, q, sigma) - 2*base + bs_call_price(S-eps, K, T, r, q, sigma))/eps**2
        vega  = (bs_call_price(S, K, T, r, q, sigma+eps) - base)/eps
        theta = (bs_call_price(S, K, max(T-1/365, 1e-8), r, q, sigma) - base)*365
        rho   = (bs_call_price(S, K, T, r+eps, q, sigma) - base)/eps
        return delta, gamma, vega, theta, rho

    d1 = (np.log(S/K) + (r - q + 0.5*sigma*sigma) * T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    Nd1 = norm.cdf(d1)
    nd1 = norm.pdf(d1)
    delta = np.exp(-q*T) * Nd1
    gamma = np.exp(-q*T) * nd1 / (S*sigma*np.sqrt(T))
    vega  = S*np.exp(-q*T) * nd1 * np.sqrt(T)
    theta = (-S*np.exp(-q*T)*nd1*sigma/(2*np.sqrt(T))
             + q*S*np.exp(-q*T)*Nd1
             - r*K*np.exp(-r*T)*norm.cdf(d2))
    rho   = K*T*np.exp(-r*T)*norm.cdf(d2)
    return delta, gamma, vega, theta, rho

def invert_iv_call(S, K, T, r, q, price, lo=1e-4, hi=5.0):
    """Invert BS to get IV from price via Brent root-finding."""
    target = np.clip(price, 1e-6, None)
    def f(sig):
        return bs_call_price(S, K, T, r, q, sig) - target
    try:
        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0:
            hi2 = hi * 2
            f_hi2 = f(hi2)
            if f_lo * f_hi2 > 0:
                return np.nan
            return brentq(f, lo, hi2, maxiter=200)
        return brentq(f, lo, hi, maxiter=200)
    except Exception:
        return np.nan


# ---------------------------
# Data: SPY & VIX 1-minute
# ---------------------------
def get_intraday_data(days=5):
    """Pull ~last N days of 1m data. yfinance caps 1m to ~7 days."""
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=days)
    data = yf.download(["SPY", "^VIX"], start=start, end=end, interval="1m",
                       auto_adjust=False, progress=False)
    data = data.dropna()
    spy_close = data["Close"]["SPY"].rename("SPY")
    vix_close = data["Close"]["^VIX"].rename("VIX")
    df = pd.concat([spy_close, vix_close], axis=1).dropna()
    df["SPY_ret"] = df["SPY"].pct_change()
    df["VIX_ret"] = df["VIX"].pct_change()
    df = df.dropna()
    return df


# ---------------------------------------
# Copula fitting (Gaussian & Student-t)
# ---------------------------------------
@dataclass
class CopulaFit:
    family: str
    model: object
    tau: float
    tail_dep_est: Optional[float]

def fit_copulas(returns_df: pd.DataFrame) -> List[CopulaFit]:
    # pseudo-observations (ranks -> uniforms)
    Udf = returns_df[["SPY_ret", "VIX_ret"]].rank(method="average") / (len(returns_df) + 1.0)
    U = Udf.values
    fits = []

    tau = returns_df[["SPY_ret", "VIX_ret"]].corr(method="kendall").iloc[0,1]

    # Gaussian copula
    gauss = GaussianMultivariate()
    gauss.fit(pd.DataFrame(U, columns=["SPY_ret", "VIX_ret"]))
    fits.append(CopulaFit("Gaussian", gauss, tau=float(tau), tail_dep_est=None))

    # Student-t copula if available
    if HAS_T_COPULA:
        tcop = StudentTMultivariate()
        tcop.fit(pd.DataFrame(U, columns=["SPY_ret", "VIX_ret"]))
        tail_dep = None  # exact tail dep depends on rho, nu; we leave as None
        fits.append(CopulaFit("Student-t", tcop, tau=float(tau), tail_dep_est=tail_dep))

    return fits

def sample_from_copula(fit: CopulaFit, n=10000, seed=42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    samples = fit.model.sample(n).values  # uniforms in [0,1]
    return np.clip(samples, 1e-6, 1-1e-6)


# ---------------------------
# Options Chain (robust) + SVI
# ---------------------------
def load_spy_options_slices(max_expiries=6) -> Tuple[float, List[Tuple[pd.DataFrame, pd.Timestamp]]]:
    """
    Pulls a few near expiries, combines calls+puts, filters, and backfills IV by inversion when missing.
    Returns: (spot, [(df_filtered_for_fit, expiry_ts_utc), ...])
    """
    tkr = yf.Ticker("SPY")
    expiries = tkr.options or []
    now = pd.Timestamp.now(tz="UTC")

    # Spot
    hist = tkr.history(period="1d")
    if hist.empty:
        raise RuntimeError("No SPY spot data.")
    spot = float(hist["Close"].iloc[-1])

    picks: List[Tuple[pd.DataFrame, pd.Timestamp]] = []
    for e in expiries[:max_expiries]:
        try:
            exp_ts = to_utc(pd.to_datetime(e, utc=True))
            T_years = (exp_ts - now).total_seconds() / (365.0*24*3600.0)
            if T_years <= 1e-6:
                continue

            chain = tkr.option_chain(e)
            calls = chain.calls.copy()
            puts  = chain.puts.copy()

            def prep(df):
                if df is None or df.empty:
                    return pd.DataFrame()
                # Some yfinance columns can vary; guard with get
                cols = ["strike","lastPrice","bid","ask","impliedVolatility"]
                cols = [c for c in cols if c in df.columns]
                base = df[cols].copy()
                # Ensure expected columns exist
                for c in ["strike","lastPrice","bid","ask","impliedVolatility"]:
                    if c not in base.columns:
                        base[c] = np.nan
                base["mid"] = (base["bid"].fillna(0) + base["ask"].fillna(0))/2
                base["expiry"] = exp_ts
                return base

            calls = prep(calls)
            puts  = prep(puts)
            df = pd.concat([calls, puts], axis=0, ignore_index=True)
            df = df.dropna(subset=["strike"])
            df["strike"] = df["strike"].astype(float)

            # Liquidity filter
            df = df[(df["bid"] > 0) & (df["ask"] > 0)]
            if df.empty:
                continue

            # Restrict to reasonable log-moneyness band
            r, q = 0.03, 0.0
            F = spot * math.exp((r - q)*max(T_years, 1e-8))
            k = np.log(df["strike"].to_numpy(dtype=float) / F)
            band = np.abs(k) <= 0.6
            df_band = df.loc[band].copy()
            if df_band.empty:
                continue

            # Backfill IVs by inversion where needed
            iv = df_band["impliedVolatility"].to_numpy(dtype=float)
            need_invert = (~np.isfinite(iv)) | (iv <= 1e-6)
            if need_invert.any():
                K_arr = df_band["strike"].to_numpy(dtype=float)
                mid   = df_band["mid"].to_numpy(dtype=float)
                iv_new = iv.copy()
                for i in np.where(need_invert)[0]:
                    iv_new[i] = invert_iv_call(spot, K_arr[i], T_years, r, q, mid[i])
                df_band["impliedVolatility"] = iv_new

            # Final clean
            mask = np.isfinite(df_band["impliedVolatility"].to_numpy()) & (df_band["impliedVolatility"] > 1e-4)
            df_band = df_band.loc[mask]
            if len(df_band) < 12:
                continue

            picks.append((df_band.reset_index(drop=True), exp_ts))
        except Exception:
            continue

    return float(spot), picks

@dataclass
class SVISlice:
    T: float
    params: Tuple[float, float, float, float, float]  # (a,b,rho,m,sigma)

def fit_svi_slice(spot: float, df_calls: pd.DataFrame, expiry_ts: pd.Timestamp, r=0.03, q=0.0) -> Optional[SVISlice]:
    now = pd.Timestamp.now(tz="UTC")
    expiry_ts = to_utc(expiry_ts)
    T = (expiry_ts - now).total_seconds() / (365.0*24*3600.0)
    if T <= 1e-6:
        return None

    F = spot * math.exp((r - q) * T)
    strikes = df_calls["strike"].to_numpy(dtype=float)
    iv      = df_calls["impliedVolatility"].to_numpy(dtype=float)

    mask = np.isfinite(strikes) & np.isfinite(iv) & (iv > 1e-5) & (strikes > 1e-8)
    strikes, iv = strikes[mask], iv[mask]
    if len(strikes) < 10:
        return None

    k = np.log(strikes / F)
    w_obs = (iv**2) * T

    def svi_w(k_, a,b,rho,m,sig):
        return a + b*(rho*(k_-m) + np.sqrt((k_-m)**2 + sig**2))

    def loss(theta):
        a,b,rho,m,sig = theta
        # feasibility
        if b <= 0 or sig <= 0 or abs(rho) >= 0.999:
            return 1e6
        w_fit = svi_w(k, a,b,rho,m,sig)
        if np.any(w_fit <= 0) or np.any(~np.isfinite(w_fit)):
            return 1e6
        return np.mean((w_fit - w_obs)**2)

    guess = np.array([0.01, 0.2, 0.0, 0.0, 0.2])
    bounds = [(-0.5, 2.0), (1e-6, 5.0), (-0.999, 0.999), (-2.0, 2.0), (1e-4, 5.0)]
    res = minimize(loss, guess, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    if not res.success:
        return None

    a,b,rho,m,sig = res.x
    return SVISlice(T=float(T), params=(float(a), float(b), float(rho), float(m), float(sig)))

def interpolate_svi_slices(slices: List[SVISlice]):
    """Linear interpolation of (a,b,rho,m,sigma) across maturities; returns iv_func(S,K,T, r,q)."""
    if not slices:
        raise ValueError("No SVI slices fitted.")
    slices_sorted = sorted(slices, key=lambda s: s.T)
    T_arr = np.array([s.T for s in slices_sorted])
    P_arr = np.array([s.params for s in slices_sorted])  # [n,5]

    def svi_w(k, a,b,rho,m,sig):
        return a + b*( rho*(k - m) + np.sqrt((k - m)**2 + sig**2) )

    def params_at(Tq):
        if Tq <= T_arr[0]:
            return P_arr[0]
        if Tq >= T_arr[-1]:
            return P_arr[-1]
        i = np.searchsorted(T_arr, Tq) - 1
        t0, t1 = T_arr[i], T_arr[i+1]
        w = (Tq - t0)/(t1 - t0)
        return (1-w)*P_arr[i] + w*P_arr[i+1]

    def iv_func(S, K, T, r=0.03, q=0.0):
        T = max(T, 1e-8)
        F = S * math.exp((r - q) * T)
        k = np.log(np.array(K, ndmin=1, dtype=float)/F)
        a,b,rho,m,sig = params_at(T)
        w = svi_w(k, a,b,rho,m,sig)  # total variance
        iv = np.sqrt(np.maximum(w, 1e-10)/T)
        return iv if np.ndim(K)>0 else float(iv[0])

    return iv_func

def fallback_surface_from_chain(spot, df_calls, expiry_ts, r=0.03, q=0.0):
    """
    Fallback IV surface: quadratic fit in log-moneyness for one expiry, flat across T.
    Keeps the engine running if SVI fails for all maturities.
    """
    now = pd.Timestamp.now(tz="UTC")
    expiry_ts = to_utc(expiry_ts)
    T0 = max(1e-6, (expiry_ts - now).total_seconds() / (365.0*24*3600.0))
    F0 = spot * math.exp((r - q) * T0)

    strikes = df_calls["strike"].to_numpy(dtype=float)
    iv      = df_calls["impliedVolatility"].to_numpy(dtype=float)
    k = np.log(strikes / F0)
    mask = np.isfinite(k) & np.isfinite(iv)
    k, iv = k[mask], iv[mask]
    if len(k) < 8:
        raise RuntimeError("Not enough data for fallback surface.")

    X = np.column_stack([np.ones_like(k), k, k*k])
    coef, *_ = np.linalg.lstsq(X, iv, rcond=None)

    def iv_func(S, K, T, r=0.03, q=0.0):
        T = max(T, 1e-8)
        F = S * math.exp((r - q) * T)
        kk = np.log(np.array(K, ndmin=1, dtype=float)/F)
        iv_hat = coef[0] + coef[1]*kk + coef[2]*kk*kk
        iv_hat = np.clip(iv_hat, 1e-4, 5.0)
        return iv_hat if np.ndim(K)>0 else float(iv_hat[0])

    return iv_func


# ---------------------------------
# Scenario engine (copula + options)
# ---------------------------------
def empirical_quantile_map(sample_u: np.ndarray, ref_returns: pd.Series) -> np.ndarray:
    """Map uniform(0,1) samples to returns using inverse empirical CDF."""
    ref_sorted = np.sort(ref_returns.values)
    ranks = (sample_u * (len(ref_sorted)-1)).astype(int)
    return ref_sorted[np.clip(ranks, 0, len(ref_sorted)-1)]

def simulate_pnl(
    copula_fit: CopulaFit,
    ref_returns: pd.DataFrame,
    S0: float,
    K: float,
    T_years: float,
    r: float,
    q: float,
    iv_surface,                # function S,K,T -> IV
    n_sims=10000,
    seed=42,
    vix_beta_to_iv=4.0         # simple: dIV ≈ beta * dVIX (% terms)
):
    U = sample_from_copula(copula_fit, n=n_sims, seed=seed)  # (SPY, VIX) uniforms
    dS = empirical_quantile_map(U[:,0], ref_returns["SPY_ret"])
    dVIX = empirical_quantile_map(U[:,1], ref_returns["VIX_ret"])

    S1 = S0 * (1.0 + dS)
    base_iv = float(iv_surface(S0, K, T_years, r=r, q=q))
    iv1 = np.clip(base_iv * (1.0 + vix_beta_to_iv * dVIX), 1e-4, 5.0)

    C0 = bs_call_price(S0, K, T_years, r=r, q=q, sigma=base_iv)
    C1 = np.array([bs_call_price(S1[i], K, T_years, r=r, q=q, sigma=iv1[i]) for i in range(n_sims)])
    pnl = C1 - C0

    var_95 = np.quantile(pnl, 0.05)
    cvar_95 = pnl[pnl <= var_95].mean() if np.any(pnl <= var_95) else var_95
    out = {
        "C0": C0,
        "base_iv": base_iv,
        "pnl": pnl,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "S1_mean": float(S1.mean()),
        "IV1_mean": float(iv1.mean())
    }
    return out


# ---------------------------
# Main
# ---------------------------
def main(streamlit_mode: bool = False):
    # 1) Intraday data + copulas
    df = get_intraday_data(days=5)
    if df.empty:
        raise RuntimeError("No intraday data pulled; try later.")
    copula_fits = fit_copulas(df)

    # 2) Option chain -> SVI slices -> IV surface (with fallback)
    S0, chain_list = load_spy_options_slices(max_expiries=6)
    if np.isnan(S0) or len(chain_list) == 0:
        raise RuntimeError("No usable option chains returned. Market may be closed or data unavailable.")

    svi_slices: List[SVISlice] = []
    for df_band, exp_ts in chain_list:
        sl = fit_svi_slice(S0, df_band, exp_ts)
        if sl is not None:
            svi_slices.append(sl)

    if len(svi_slices) == 0:
        # Fallback surface from first usable chain
        df0, exp0 = chain_list[0]
        iv_surface = fallback_surface_from_chain(S0, df0, exp0)
        nearest_T = max(1e-6, (to_utc(exp0) - pd.Timestamp.now(tz="UTC")).total_seconds() / (365.0*24*3600.0))
    else:
        iv_surface = interpolate_svi_slices(svi_slices)
        nearest_T = min(svi_slices, key=lambda s: s.T).T

    # Choose a representative ATM contract at nearest expiry
    K_atm = round(S0)
    r = 0.03
    q = 0.0

    # Greeks at baseline
    iv0 = float(iv_surface(S0, K_atm, nearest_T, r=r, q=q))
    delta, gamma, vega, theta, rho = bs_call_greeks(S0, K_atm, nearest_T, r=r, q=q, sigma=iv0)

    # 3) Simulations per copula fit
    results = []
    for fit in copula_fits:
        res = simulate_pnl(
            fit, df[["SPY_ret", "VIX_ret"]],
            S0=S0, K=K_atm, T_years=nearest_T, r=r, q=q, iv_surface=iv_surface,
            n_sims=20000, seed=123
        )
        results.append((fit, res))

    # ---- Presentation ----
    if streamlit_mode and STREAMLIT:
        st.title("Intraday Vol Surface + Copula Risk Engine (SPY/VIX)")
        st.caption("SVI (or fallback) volatility surface; Gaussian/Student-t copulas; MC stress testing.")

        st.subheader("Spot & Chosen Contract")
        c1, c2, c3 = st.columns(3)
        c1.metric("SPY Spot", f"{S0:,.2f}")
        c2.metric("ATM Strike", f"{K_atm}")
        c3.metric("Nearest T (yrs)", f"{nearest_T:.4f}")

        st.subheader("Greeks (Baseline)")
        st.write(pd.DataFrame({
            "Delta":[delta],"Gamma":[gamma],"Vega":[vega],
            "Theta/yr":[theta],"Rho":[rho], "IV (atm)":[iv0]
        }))

        # Vol surface plot (K vs T -> IV)
        if PLOTLY:
            try:
                Ks = np.linspace(S0*0.7, S0*1.3, 40)
                # Build Ts grid:
                if len(svi_slices) > 0:
                    Ts = np.linspace(max(1e-4, min([s.T for s in svi_slices])),
                                     max([s.T for s in svi_slices]), 30)
                else:
                    Ts = np.linspace(max(1e-4, nearest_T*0.25), nearest_T*1.5, 30)

                KK, TT = np.meshgrid(Ks, Ts)
                IV = np.zeros_like(KK)
                for i in range(TT.shape[0]):
                    iv_row = iv_surface(S0, KK[i,:], TT[i,0], r=r, q=q)
                    IV[i,:] = iv_row

                fig = go.Figure(data=[go.Surface(x=KK, y=TT, z=IV)])
                fig.update_layout(
                    title="Implied Volatility Surface (K vs T)",
                    scene=dict(xaxis_title="Strike K", yaxis_title="Maturity T (yrs)", zaxis_title="IV"),
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Surface plot unavailable: {e}")

        st.subheader("Copula Fits & Risk")
        for fit, res in results:
            st.markdown(f"**{fit.family} Copula** — Kendall's τ = {fit.tau:.3f}")
            st.write(pd.DataFrame({
                "C0":[res["C0"]], "Base IV":[res["base_iv"]],
                "Mean S1":[res["S1_mean"]], "Mean IV1":[res["IV1_mean"]],
                "VaR 95%":[res["VaR_95"]], "CVaR 95%":[res["CVaR_95"]]
            }))

            if PLOTLY:
                hist = np.histogram(res["pnl"], bins=80)
                fig2 = go.Figure()
                centers = (hist[1][:-1] + hist[1][1:])/2
                fig2.add_trace(go.Bar(x=centers, y=hist[0]))
                fig2.update_layout(title=f"P&L Distribution — {fit.family}",
                                   xaxis_title="P&L", yaxis_title="Frequency")
                st.plotly_chart(fig2, use_container_width=True)

        st.info("Tip: extend to a multi-leg portfolio, add calendar/butterfly no-arb checks, and estimate IV shock beta from history.")
        return

    # Console fallback
    print("\n=== Intraday Vol Surface + Copula Risk Engine ===")
    print(f"SPY Spot: {S0:.2f} | ATM Strike: {K_atm} | Nearest T: {nearest_T:.5f} yrs")
    print(f"ATM IV: {iv0:.4f}")
    print(f"Greeks: Δ={delta:.4f}, Γ={gamma:.6f}, Vega={vega:.4f}, Θ/yr={theta:.2f}, ρ={rho:.2f}")

    for fit, res in results:
        print(f"\n[{fit.family} Copula] Kendall's tau = {fit.tau:.3f}")
        print(f"  C0={res['C0']:.4f}, Base IV={res['base_iv']:.4f}")
        print(f"  Mean S1={res['S1_mean']:.2f}, Mean IV1={res['IV1_mean']:.4f}")
        print(f"  VaR 95%={res['VaR_95']:.4f}, CVaR 95%={res['CVaR_95']:.4f}")


if __name__ == "__main__":
    # If run via `streamlit run vol_surface_copula_engine.py`, Streamlit sets up its own entry.
    # Here we auto-enable UI when streamlit is available and not in its special context.
    in_streamlit_special = False
    try:
        import __main__ as mainmod
        in_streamlit_special = hasattr(mainmod, "__file__") and "streamlit" in str(mainmod.__file__).lower()
    except Exception:
        pass
    main(streamlit_mode=(STREAMLIT and not in_streamlit_special))
