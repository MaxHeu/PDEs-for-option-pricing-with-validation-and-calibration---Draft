# =============================================================
# Calibration.py  –  BS / Heston / MJD option model calibration
# Inspired by: TheQuantPy – Heston Model Calibration in the Real World
# =============================================================
# pip install yfinance nelson_siegel_svensson scipy numpy pandas
import time
from typing import Union, List, Dict
import math
import warnings
warnings.filterwarnings("ignore")
import Analytical
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

import yfinance as yf
from nelson_siegel_svensson.calibrate import calibrate_nss_ols


# ─────────────────────────────────────────────────────────────
# NSS YIELD CURVE
# ─────────────────────────────────────────────────────────────

def spot_rate(params, maturity):
    """
    Nelson-Siegel-Svensson spot rate.
    params : [beta0, beta1, beta2, beta3, tau1, tau2]
    """
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    t1 = maturity / tau1
    t2 = maturity / tau2
    return (beta0
            + beta1 * (1 - np.exp(-t1)) / t1
            + beta2 * ((1 - np.exp(-t1)) / t1 - np.exp(-t1))
            + beta3 * ((1 - np.exp(-t2)) / t2 - np.exp(-t2)))


def get_risk_free_rate(yield_maturities, yields, maturity):
    curve_fit, _ = calibrate_nss_ols(yield_maturities, yields)
    params = [curve_fit.beta0, curve_fit.beta1, curve_fit.beta2,
              curve_fit.beta3, curve_fit.tau1,  curve_fit.tau2]
    return spot_rate(params, maturity)


# US Treasury benchmark yields (approximate, update as needed)
yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
yields = np.array([0.15, 0.27, 0.50, 0.93, 1.52, 2.13, 2.32,
                   2.34, 2.37, 2.32, 2.65, 2.52]).astype(float) / 100


# ─────────────────────────────────────────────────────────────
# PER-OPTION OBJECTIVE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def Obj_BS(market_price, option_type, S, K, r, sigma, T):
    q = 0
    model_price = Analytical.bs_call(S, K, r, q, sigma, T) if option_type == "call" else Analytical.bs_put(S, K, r, q, sigma, T)
    return (market_price - model_price) ** 2


def Obj_Heston(market_price, option_type, S, K, v0, T, r, kappa, theta, xi, rho):
    model_price = (Analytical.heston_call(S, K, v0, T, r, kappa, theta, xi, rho)
                   if option_type == "call"
                   else Analytical.heston_put(S, K, v0, T, r, kappa, theta, xi, rho))
    return (market_price - model_price) ** 2


def Obj_MJD(market_price, option_type, S, K, r, sigma, T, lam, mu_y, sigma_y):
    q = 0
    model_price = (Analytical.mjd_call(S, K, r, q, sigma, T, lam, mu_y, sigma_y)
                   if option_type == "call"
                   else Analytical.mjd_put(S, K, r, q, sigma, T, lam, mu_y, sigma_y))
    return (market_price - model_price) ** 2


# ─────────────────────────────────────────────────────────────
# OPTION DATA HELPER
# ─────────────────────────────────────────────────────────────

def _fetch_option_chain(ticker: str, maturity: str):
    """
    Download option chain and return a cleaned DataFrame of OTM options
    with columns: strike, mid, type.
    Uses OTM calls (K > S0) and OTM puts (K < S0) to avoid
    in-the-money price distortions from early-exercise premia.
    """
    tk = yf.Ticker(ticker)
    # Validate maturity is listed
    if maturity not in tk.options:
        available = "\n  ".join(tk.options[:10])
        raise ValueError(
            f"Maturity {maturity} not available for {ticker}.\n"
            f"First available expirations:\n  {available}"
        )
    chain = tk.option_chain(maturity)

    # Most recent underlying price
    hist = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
    cols = hist.columns
    # Handle MultiIndex (yfinance >=0.2)
    if isinstance(cols, pd.MultiIndex):
        S0 = float(hist["Close"][ticker].dropna().iloc[-1])
    else:
        S0 = float(hist["Close"].dropna().iloc[-1])

    def clean(df, opt_type, side):
        df = df.copy()
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df = df[(df["bid"] > 0) & (df["ask"] > 0)]
        df = df[df["volume"].fillna(0) > 0]
        df["type"] = opt_type
        if side == "OTM_call":
            df = df[df["strike"] > S0]
        else:
            df = df[df["strike"] < S0]
        return df[["strike", "mid", "type"]]

    otm_calls = clean(chain.calls, "call", "OTM_call")
    otm_puts  = clean(chain.puts,  "put",  "OTM_put")
    options   = pd.concat([otm_puts, otm_calls], ignore_index=True)
    if len(options) < 4:
        raise ValueError(f"Too few liquid OTM options ({len(options)}) for calibration.")

    return S0, options


# ─────────────────────────────────────────────────────────────
# MAIN CALIBRATION FUNCTION
# ─────────────────────────────────────────────────────────────

def calibrate(Model: str, ticker: str, Maturity: str,
                yield_maturities: np.ndarray, yields: np.ndarray) -> dict:
    """
    Calibrate option pricing model to market data.

    Parameters
    ----------
    Model           : 'BS' | 'Heston' | 'MJD'
    ticker          : yfinance ticker symbol (e.g. 'AAPL')
    Maturity        : expiry date string 'YYYY-MM-DD' (must be a listed expiry)
    yield_maturities: array of Treasury maturities in years
    yields          : corresponding yields (decimal)

    Returns
    -------
    parameters : dict
        BS     -> {'r', 'sigma'}
        Heston -> {'r', 'v0', 'kappa', 'theta', 'xi', 'rho'}
        MJD    -> {'r', 'sigma', 'lambda', 'mu_y', 'sigma_y'}
    """
    # ── Fetch data ──────────────────────────────────────────────
    S0, options = _fetch_option_chain(ticker, Maturity)

    today   = dt.today().date()
    mat_date = dt.strptime(Maturity, "%Y-%m-%d").date()
    T = (mat_date - today).days / 252 # maybe should be 252
    if T <= 0:
        raise ValueError(f"Maturity {Maturity} is in the past.")

    r = get_risk_free_rate(yield_maturities, yields, T)
    N = len(options)

    print(f"[{Model:6s}] ticker={ticker}  S0={S0:.2f}  T={T:.4f}y  "
          f"r={r*100:.3f}%  N_opts={N}")

    parameters = {}

    # ─────────────────────────────────────────────────────────
    # Black-Scholes: calibrate sigma (single free parameter)
    # ─────────────────────────────────────────────────────────
    if Model == "BS":

        def loss(params):
            (sigma,) = params
            if sigma <= 1e-6:
                return 1e10
            return sum(
                Obj_BS(row.mid, row.type, S0, row.strike, r, sigma, T)
                for row in options.itertuples()
            ) / N

        res = minimize(loss, x0=[0.20], method="L-BFGS-B",
                       bounds=[(1e-4, 5.0)],
                       options={"maxiter": 2000, "ftol": 1e-14})
        (sigma_cal,) = res.x
        rmse = np.sqrt(res.fun)
        parameters = {"r": r, "sigma": sigma_cal}
        print(f"         sigma={sigma_cal:.6f}   RMSE={rmse:.4f}   success={res.success}")

    # ─────────────────────────────────────────────────────────
    # Heston: calibrate (kappa, theta, xi, rho, v0)
    # ─────────────────────────────────────────────────────────
    elif Model == "Heston":

        def loss(params):
            kappa, theta, xi, rho, v0 = params
            # Hard domain constraints
            if not (kappa > 0 and theta > 0 and xi > 0 and v0 > 0 and -1 < rho < 1):
                return 1e10
            # Feller condition soft penalty: 2κθ ≥ ξ²
            feller_penalty = max(0.0, xi**2 - 2 * kappa * theta) * 1e4
            total = 0.0
            for row in options.itertuples():
                try:
                    total += Obj_Heston(row.mid, row.type, S0, row.strike,
                                        v0, T, r, kappa, theta, xi, rho)
                except Exception:
                    total += 1e4
            return total / N + feller_penalty

        x0     = [2.0,   0.04, 0.5,   -0.6,  0.04]
        bounds = [(0.01, 20.), (1e-4, 1.), (0.01, 5.),
                  (-0.999, 0.999), (1e-4, 1.)]

        # Two-stage: global (Nelder-Mead) then local (L-BFGS-B) refinement
        res1 = minimize(loss, x0,     method="Nelder-Mead",
                        options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-6})
        #res  = minimize(loss, res1.x, method="L-BFGS-B", bounds=bounds,
                        #options={"maxiter": 500, "ftol": 1e-8})

        kappa, theta, xi, rho, v0 = res1.x
        rmse = np.sqrt(res1.fun)
        parameters = {"r": r, "v0": v0, "kappa": kappa,
                      "theta": theta, "xi": xi, "rho": rho}
        print(f"         kappa={kappa:.4f}  theta={theta:.4f}  xi={xi:.4f}  "
              f"rho={rho:.4f}  v0={v0:.4f}  RMSE={rmse:.4f}  success={res1.success}")

    # ─────────────────────────────────────────────────────────
    # Merton Jump Diffusion: calibrate (sigma, lambda, mu_y, sigma_y)
    # ─────────────────────────────────────────────────────────
    elif Model == "MJD":

        def loss(params):
            sigma, lam, mu_y, sigma_y = params
            if not (sigma > 0 and lam >= 0 and sigma_y > 0):
                return 1e10
            total = 0.0
            for row in options.itertuples():
                try:
                    total += Obj_MJD(row.mid, row.type, S0, row.strike,
                                     r, sigma, T, lam, mu_y, sigma_y)
                except Exception:
                    total += 1e4
            return total / N

        x0     = [0.20,  5.0, -0.10,  0.10]
        bounds = [(1e-4, 3.), (0., 30.), (-2., 2.), (1e-4, 2.)]

        res1 = minimize(loss, x0,     method="Nelder-Mead",
                        options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-6})
        #res  = minimize(loss, res1.x, method="L-BFGS-B", bounds=bounds,
        #                options={"maxiter": 500, "ftol": 1e-8})

        sigma, lam, mu_y, sigma_y = res1.x
        rmse = np.sqrt(res1.fun)
        parameters = {"r": r, "sigma": sigma, "lambda": lam,
                      "mu_y": mu_y, "sigma_y": sigma_y}
        print(f"         sigma={sigma:.4f}  lambda={lam:.4f}  "
              f"mu_y={mu_y:.4f}  sigma_y={sigma_y:.4f}  RMSE={rmse:.4f}  success={res1.success}")

    else:
        raise ValueError(f"Unknown model '{Model}'. Choose 'BS', 'Heston', or 'MJD'.")

    return parameters



# ─────────────────────────────────────────────────────────────
# PRIORITY ORDER for portfolio ETF selection
# ─────────────────────────────────────────────────────────────
MAJOR_ETFS = [
    # Broad market
    'SPY', 'QQQ', 'IVV', 'VOO', 'VTI', 'IWM', 'DIA',
    # Tech / Semiconductors
    'XLK', 'VGT', 'SOXX', 'SMH', 'IGV', 'IYW', 'FTEC', 'QTEC', 'PSI',
    # Growth / Innovation / AI
    'ARKK', 'ARKW', 'ARKG', 'ARKQ', 'BOTZ', 'AIQ', 'KOMP', 'ROBO', 'IRBO',
    # Thematic
    'FDN', 'PNQI', 'SKYY', 'WCLD', 'CLOU', 'CIBR', 'HACK', 'BUG',
    # Other sectors
    'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLU',
    'VHT', 'VFH', 'VDE', 'VIS', 'VPU', 'VDC', 'VOX', 'EFA', 'EEM',
]
_PORTFOLIO_PRIORITY = ['SPY', 'QQQ', 'IVV', 'VOO', 'XLK', 'VGT', 'IYW', 'FTEC']
def get_basket_options(
    ticker1: str,
    ticker2: str,
    etf_universe: List[str] = MAJOR_ETFS,
    min_weight: float = 0.0,
) -> Union[List[Dict], bool]:
    """
    Find exchange-traded ETF options whose top holdings contain both tickers.

    ⚠ True OTC basket options are not publicly listed and cannot be queried
    via any free API. This function returns the best publicly-traded proxy:
    ETFs that hold both stocks AND have liquid listed options.

    Parameters
    ----------
    ticker1, ticker2 : str
        Stock tickers, e.g. 'NVDA', 'AAPL'.
    etf_universe : list of str
        ETFs to scan. Defaults to MAJOR_ETFS (~40 major ETFs).
    min_weight : float
        Minimum portfolio weight (0–1) each ticker must have. Default 0.0.

    Returns
    -------
    list of dict  –  each dict contains:
        'etf'                : ETF symbol
        'etf_name'           : Full name
        'weight_{t1}_pct'    : Weight of ticker1 in ETF (%)
        'weight_{t2}_pct'    : Weight of ticker2 in ETF (%)
        'n_expiries'         : Number of available option expiries
        'nearest_expiry'     : Closest expiry date
        'farthest_expiry'    : Furthest expiry date
    False  –  if no qualifying ETF with listed options is found.
    """
    t1 = ticker1.upper().strip()
    t2 = ticker2.upper().strip()

    if t1 == t2:
        raise ValueError("Tickers must be different.")

    found = []

    for etf_sym in etf_universe:
        try:
            etf = yf.Ticker(etf_sym)

            fd = etf.funds_data
            if fd is None:
                continue

            # DataFrame: index=Symbol, cols=['Name', 'Holding Percent']
            holdings = fd.top_holdings
            if holdings is None or holdings.empty:
                continue

            idx = holdings.index.str.upper()
            if t1 not in idx or t2 not in idx:
                continue

            w1 = float(holdings.loc[holdings.index[idx == t1][0], 'Holding Percent'])
            w2 = float(holdings.loc[holdings.index[idx == t2][0], 'Holding Percent'])

            if w1 < min_weight or w2 < min_weight:
                continue

            # Check for listed option expiries
            try:
                option_dates = etf.options    # tuple of 'YYYY-MM-DD' strings
            except Exception:
                option_dates = ()

            if not option_dates:
                continue

            try:
                etf_name = etf.info.get('longName', etf_sym)
            except Exception:
                etf_name = etf_sym

            found.append({
                'etf':                  etf_sym,
                'etf_name':             etf_name,
                f'weight_{t1}_pct':     round(w1 * 100, 3),
                f'weight_{t2}_pct':     round(w2 * 100, 3),
                'n_expiries':           len(option_dates),
                'nearest_expiry':       option_dates[0],
                'farthest_expiry':      option_dates[-1],
            })

            time.sleep(0.05)    # polite rate-limiting

        except Exception:
            continue

    return found if found else False
def _find_portfolio_ticker(ticker1: str, ticker2: str, maturity: str) -> str:
    """
    Return the best liquid ETF proxy for the two-asset basket that:
      (a) holds both tickers in its top holdings, and
      (b) has listed options at the requested maturity.
    Preference: SPY > QQQ > IVV > VOO > XLK > …
    Fallback: ETF with most listed expiries that covers the maturity.
    """
    candidates = get_basket_options(ticker1, ticker2)
    if not candidates:
        raise ValueError(
            f"No ETF with listed options found that holds both "
            f"{ticker1} and {ticker2}. Cannot calibrate rho."
        )

    etf_set = {c['etf']: c for c in candidates}

    # Priority pass
    for preferred in _PORTFOLIO_PRIORITY:
        if preferred in etf_set:
            tk = yf.Ticker(preferred)
            if maturity in tk.options:
                return preferred

    # Fallback: most expiries that cover the maturity
    for c in sorted(candidates, key=lambda x: x['n_expiries'], reverse=True):
        tk = yf.Ticker(c['etf'])
        if maturity in tk.options:
            return c['etf']

    raise ValueError(
        f"None of the candidate ETFs {[c['etf'] for c in candidates]} "
        f"have options listed at maturity {maturity}."
    )


def calibrate_TwoU(
    ticker1: str,
    ticker2: str,
    Maturity: str,
    yield_maturities: np.ndarray,
    yields: np.ndarray,
    portfolio_ticker: str = None,
) -> dict:
    """
    Calibrate the 2-asset Black-Scholes model using the CBOE equicorrelation
    methodology for the implied correlation rho.

    Algorithm
    ---------
    1.  r_1 = r_2 = r  from the NSS yield curve at maturity T  (exact).
    2.  sigma_1, sigma_2  calibrated individually via single-asset BS fits  (exact).
    3.  sigma_P  calibrated via BS fit on the basket ETF.
    4.  For EVERY holding in the ETF's top_holdings:
            - w_i  = ETF holding weight
            - sigma_i = individual BS implied vol at Maturity
        (holdings where calibration fails are dropped; weights are renormalised)
    5.  Implied equicorrelation via CBOE formula [PDF eq. 4.3.4] over ALL holdings:
            rho = (sigma_P^2 - sum_i w_i^2 * sigma_i^2)
                  / (2 * sum_{i<j} w_i * w_j * sigma_i * sigma_j)
        This rho is the equicorrelation of the WHOLE ETF, then reused as
        the pairwise correlation between ticker1 and ticker2.

    Parameters
    ----------
    ticker1 : str         First underlying, e.g. 'NVDA'.
    ticker2 : str         Second underlying, e.g. 'AAPL'.
    Maturity : str        Expiry date 'YYYY-MM-DD'.
    yield_maturities : np.ndarray  Treasury benchmark maturities in years.
    yields : np.ndarray            Corresponding Treasury yields (decimal).
    portfolio_ticker : str, optional
        Override the auto-detected basket ETF (e.g. 'QQQ').
        If None, selected automatically via get_basket_options().

    Returns
    -------
    dict with keys:
        'r_1', 'r_2'            : risk-free rate (r_1 == r_2)
        'sigma_1', 'sigma_2'    : individual implied vols for the two stocks
        'sigma_P'               : ETF implied vol
        'rho'                   : CBOE equicorrelation of the ETF  ∈ (-1, 1)
        'w_1', 'w_2'            : ETF holding weights for the two stocks
        'S1_0', 'S2_0'          : spot prices
        'portfolio_ticker'      : ETF used as portfolio proxy
        'etf_holdings_used'     : dict {symbol: (weight, sigma)} of all
                                  holdings successfully used in rho computation
    """
    t1 = ticker1.upper().strip()
    t2 = ticker2.upper().strip()

    if t1 == t2:
        raise ValueError("ticker1 and ticker2 must be different.")

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Two-Underlying BS Calibration | {t1} + {t2} | T={Maturity}")
    print(sep)

    # ── Steps 1 & 2: individual BS calibrations (r, sigma_1, sigma_2) ──────
    print(f"[1/3] Calibrating single-asset BS for {t1} ")
    params1 = calibrate("BS", t1, Maturity, yield_maturities, yields)

    print(f" [2/3] Calibrating single-asset BS for {t2} ")
    params2 = calibrate("BS", t2, Maturity, yield_maturities, yields)

    r      = params1["r"]    # r_1 = r_2 (same NSS curve, same T)
    sigma1 = params1["sigma"]
    sigma2 = params2["sigma"]

    # ── Step 3: portfolio ETF → sigma_P ──────────────────────────────────────
    if portfolio_ticker is None:
        print(f"[3/3] Auto-detecting basket ETF for {t1} + {t2} ")
        portfolio_ticker = _find_portfolio_ticker(t1, t2, Maturity)

    print(f"\[3/3] Calibrating portfolio BS on ETF {portfolio_ticker} ")
    params_P = calibrate("BS", portfolio_ticker, Maturity, yield_maturities, yields)
    sigma_P  = params_P["sigma"]

    # ── Step 4: calibrate sigma_i for EVERY ETF top holding ──────────────────
    fd       = yf.Ticker(portfolio_ticker).funds_data
    holdings = fd.top_holdings                    # index=Symbol, col='Holding Percent'

    print(f"Calibrating BS implied vol for each of the "
          f"{len(holdings)} ETF top holdings")

    holding_data = {}   # symbol -> (weight, sigma)

    for symbol in holdings.index:
        sym_up = symbol.upper()
        raw_w  = float(holdings.loc[symbol, 'Holding Percent'])

        # If it's one of our two target stocks, reuse already-calibrated sigma
        if sym_up == t1:
            holding_data[sym_up] = (raw_w, sigma1)
            #print(f"   {sym_up:8s}  w={raw_w:.4f}  sigma={sigma1:.6f}  [reused]")
            continue
        if sym_up == t2:
            holding_data[sym_up] = (raw_w, sigma2)
            #print(f"   {sym_up:8s}  w={raw_w:.4f}  sigma={sigma2:.6f}  [reused]")
            continue

        # Otherwise calibrate fresh
        try:
            p     = calibrate("BS", sym_up, Maturity, yield_maturities, yields)
            sigma = p["sigma"]
            holding_data[sym_up] = (raw_w, sigma)
            #print(f"   {sym_up:8s}  w={raw_w:.4f}  sigma={sigma:.6f}")
        except Exception as e:
            continue
            #print(f"   {sym_up:8s}  w={raw_w:.4f}  SKIPPED ({e})")

    if t1 not in holding_data or t2 not in holding_data:
        raise ValueError(
            f"Could not calibrate sigma for {t1} or {t2} within the ETF holdings."
        )

    # ── Step 5: CBOE equicorrelation formula over ALL calibrated holdings ────
    #   Renormalise weights of the calibrated subset so they sum to 1
    symbols = list(holding_data.keys())
    weights = np.array([holding_data[s][0] for s in symbols])
    sigmas  = np.array([holding_data[s][1] for s in symbols])
    weights = weights / weights.sum()   # renormalise to account for dropped holdings

    # numerator  = sigma_P^2 - sum_i w_i^2 * sigma_i^2
    numerator   = sigma_P**2 - np.sum(weights**2 * sigmas**2)

    # denominator = 2 * sum_{i<j} w_i * w_j * sigma_i * sigma_j
    #             = (sum_i w_i * sigma_i)^2 - sum_i (w_i * sigma_i)^2
    ws      = weights * sigmas
    denominator = (ws.sum())**2 - np.sum(ws**2)   # vectorised form of 2*sum_{i<j}

    if abs(denominator) < 1e-14:
        raise ValueError(
            "Denominator ≈ 0: cannot compute rho. "
            "All holdings may have identical implied vols."
        )

    rho = float(np.clip(numerator / denominator, -1.0 + 1e-9, 1.0 - 1e-9))

    # ── Spot prices (for downstream pricing) ─────────────────────────────────
    def _spot(ticker: str) -> float:
        hist = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            return float(hist["Close"][ticker].dropna().iloc[-1])
        return float(hist["Close"].dropna().iloc[-1])

    S1 = _spot(t1)
    S2 = _spot(t2)

    w1 = holding_data[t1][0]
    w2 = holding_data[t2][0]

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  RESULTS")
    print(f"{sep}")
    print(f"  Spots            : {t1}={S1:.4f}  {t2}={S2:.4f}")
    print(f"  ETF weights      : w({t1})={w1:.6f}  w({t2})={w2:.6f}")
    print(f"  Holdings used    : {len(holding_data)} / {len(holdings)}")
    print(f"  r                : {r*100:.4f}%  (r_1 = r_2)")
    print(f"  sigma_1          : {sigma1:.6f}  ({t1})")
    print(f"  sigma_2          : {sigma2:.6f}  ({t2})")
    print(f"  sigma_P          : {sigma_P:.6f}  (ETF: {portfolio_ticker})")
    print(f"  rho (CBOE equicorr.) : {rho:.6f}")
    print(sep)

    return {
        "r_1":               r,
        "r_2":               r,
        "sigma_1":           sigma1,
        "sigma_2":           sigma2,
        "sigma_P":           sigma_P,
        "rho":               rho,
        "w_1":               w1,
        "w_2":               w2,
        "S1_0":              S1,
        "S2_0":              S2,
        "portfolio_ticker":  portfolio_ticker,
        "etf_holdings_used": {s: holding_data[s] for s in holding_data},
    }
