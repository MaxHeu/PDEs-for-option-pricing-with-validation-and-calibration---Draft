"""
AnalyticalSolutions.py
======================
Closed-form / semi-analytical pricers for European (and multi-asset) options.

Models implemented
------------------
1. Black-Scholes (vanilla call/put with continuous dividends)
2. Heston semi-analytical (Gil-Pelaez / Lewis characteristic-function formula)
3. Merton Jump-Diffusion (infinite-series truncation)
4. Exchange option – Margrabe (1978)
5. Spread option – Kirk (1995) approximation
6. Basket option – Lognormal moment-matching approximation
"""
import math
import numpy as np
from scipy import stats, integrate
from typing import Literal
from Models import BlackScholes, Heston, Merton, TwoUnderlyings, PayoffFunction_1u, PayoffFunction_2u

# =============================================================================
# 1.  BLACK-SCHOLES
# =============================================================================

def bs_call(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Black-Scholes price for a European call with continuous dividend yield q.

    C(S,t) = e^{-qτ} S Φ(d1) - e^{-rτ} K Φ(d2)

    Parameters
    ----------
    S     : spot price
    K     : strike
    r     : risk-free rate (continuously compounded)
    q     : continuous dividend yield
    sigma : volatility
    T     : time to maturity τ
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-q * T) * S * stats.norm.cdf(d1) - np.exp(-r * T) * K * stats.norm.cdf(d2)


def bs_put(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Black-Scholes price for a European put via put-call parity:
        P = C - S e^{-qτ} + K e^{-rτ}
    """
    return bs_call(S, K, r, q, sigma, T) - np.exp(-q * T) * S + np.exp(-r * T) * K


def bs_price(
    S: float, K: float, r: float, q: float, sigma: float, T: float,
    option_type: Literal["call", "put"] = "call"
) -> float:
    """Unified Black-Scholes pricer (call or put)."""
    if option_type == "call":
        return bs_call(S, K, r, q, sigma, T)
    return bs_put(S, K, r, q, sigma, T)


# =============================================================================
# 2.  HESTON SEMI-ANALYTICAL (Gil-Pelaez / Lewis form)
# =============================================================================

def _heston_cf(
    phi: complex,
    S0: float,
    K: float,       # present in signature for API compatibility; not used in CF
    v0: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lam: float = 0.0,
) -> complex:
    """
    Heston characteristic function E[e^{iφ ln(S_T)}].

    Parameters
    ----------
    phi   : Fourier argument (real or complex).
    S0    : Initial spot price.
    K     : Strike (unused inside CF; kept for consistent call-site signature).
    v0    : Initial variance ν₀.
    T     : Time to maturity τ.
    r     : Risk-free rate.
    kappa : Mean-reversion speed κ.
    theta : Long-run variance θ.
    xi    : Vol-of-vol ξ.
    rho   : Spot-vol correlation ρ.
    lam   : Market price of volatility risk λ (default 0).

    Returns
    -------
    complex
        Value of the characteristic function at φ.
    """
    i    = 1j
    a    = kappa * theta          # κθ
    b    = kappa + lam            # κ + λ
    rspi = rho * xi * phi * i    # ρ ξ i φ

    d      = np.sqrt((rspi - b) ** 2 + xi ** 2 * (phi * i + phi ** 2))
    g      = (b - rspi + d) / (b - rspi - d)
    exp_dt = np.exp(d * T)

    # ── Exponent components ──────────────────────────────────────────────────
    # C combines the linear-T drift term AND the log term (both are required).
    # Common bug: including only the log part and dropping (a/ξ²)·(b-rspi+d)·T.
    C = (a / xi ** 2) * (
        (b - rspi + d) * T
        - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = (b - rspi + d) / xi ** 2 * (1.0 - exp_dt) / (1.0 - g * exp_dt)

    return np.exp(r * phi * i * T) * (S0 ** (phi * i)) * np.exp(C + v0 * D)


# ─────────────────────────────────────────────────────────────────────────────
# Semi-analytical pricers
# ─────────────────────────────────────────────────────────────────────────────

def heston_call(
    S0: float,
    K: float,
    v0: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lam: float = 0.0,
    n_quad: int = 200,
) -> float:
    """
    Heston semi-analytical European call price.

    Formula:
        C = (S0 - K·e^{-rT})/2
          + (1/π)·∫₀^∞ Re[ (e^{rT}·φ(φ-i) - K·φ(φ)) / (iφ·K^{iφ}) ] dφ

    Parameters
    ----------
    S0     : Initial spot price.
    K      : Strike price.
    v0     : Initial variance ν₀.
    T      : Time to maturity.
    r      : Risk-free rate.
    kappa  : Mean-reversion speed κ.
    theta  : Long-run variance θ.
    xi     : Vol-of-vol ξ.
    rho    : Spot-vol correlation ρ.
    lam    : Market price of vol risk λ (default 0).
    n_quad : Number of quadrature sub-intervals (default 200).

    Returns
    -------
    float
        Call option price.
    """
    def integrand(phi: float) -> float:
        i        = 1j
        cf_shift = _heston_cf(phi - 1j, S0, K, v0, T, r, kappa, theta, xi, rho, lam)
        cf_plain = _heston_cf(phi,       S0, K, v0, T, r, kappa, theta, xi, rho, lam)
        denom    = i * phi * (K ** (i * phi))
        return np.real((np.exp(r * T) * cf_shift - K * cf_plain) / denom)

    result, _ = integrate.quad(integrand, 1e-8, 200, limit=n_quad)
    return (S0 - K * np.exp(-r * T)) / 2.0 + result / np.pi


def heston_put(
    S0: float,
    K: float,
    v0: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lam: float = 0.0,
    n_quad: int = 200,
) -> float:
    """
    Heston European put price via put-call parity.

    Parameters
    ----------
    (same as heston_call)

    Returns
    -------
    float
        Put option price.
    """
    call = heston_call(S0, K, v0, T, r, kappa, theta, xi, rho, lam, n_quad)
    return call - S0 + K * np.exp(-r * T)


def heston_price(
    S0: float,
    K: float,
    v0: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lam: float = 0.0,
    n_quad: int = 200,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Unified Heston pricer dispatching to call or put.

    Parameters
    ----------
    option_type : "call" or "put" (default "call").

    Returns
    -------
    float
        Option price.
    """
    if option_type == "call":
        return heston_call(S0, K, v0, T, r, kappa, theta, xi, rho, lam, n_quad)
    return heston_put(S0, K, v0, T, r, kappa, theta, xi, rho, lam, n_quad)


# =============================================================================
# 3.  MERTON JUMP-DIFFUSION
# =============================================================================

def mjd_call(S: float, K: float, r: float, q: float, sigma: float, T: float,
             lam: float, mu_j: float, delta: float,
             n_max: int = 50) -> float:
    """
    Merton Jump-Diffusion European call price (report eq. 4.1.6–4.1.9).

    C^MJD = Σ_{n=0}^{n_max}  e^{-λτ}(λτ)^n/n!  * C^{(n)}

    where C^{(n)} is a BS-type price with conditional parameters:
        m_n = ln(S) + (r - q - λκ - σ²/2)τ + n μ_J
        v_n = σ²τ + n δ²
        κ   = exp(μ_J + δ²/2) - 1

    Parameters
    ----------
    S     : spot
    K     : strike
    r     : risk-free rate
    q     : dividend yield
    sigma : diffusion volatility
    T     : time to maturity
    lam   : jump intensity λ
    mu_j  : mean log-jump μ_J
    delta : std-dev of log-jump δ
    n_max : truncation of the Poisson sum
    """
    kappa = np.exp(mu_j + 0.5 * delta**2) - 1.0
    price = 0.0
    for n in range(n_max + 1):
        w = np.exp(-lam * T) * (lam * T)**n / math.factorial(n)
        m_n = np.log(S) + (r - q - lam * kappa - 0.5 * sigma**2) * T + n * mu_j
        v_n = sigma**2 * T + n * delta**2
        sqrt_vn = np.sqrt(v_n)
        d1n = (m_n - np.log(K) + v_n) / sqrt_vn
        d2n = d1n - sqrt_vn
        C_n = np.exp(-r * T) * (
            np.exp(m_n + 0.5 * v_n) * stats.norm.cdf(d1n) - K * stats.norm.cdf(d2n)
        )
        price += w * C_n
    return price


def mjd_put(S: float, K: float, r: float, q: float, sigma: float, T: float,
            lam: float, mu_j: float, delta: float, n_max: int = 50) -> float:
    """MJD European put via put-call parity (eq. 4.1.10)."""
    call = mjd_call(S, K, r, q, sigma, T, lam, mu_j, delta, n_max)
    return call - S * np.exp(-q * T) + K * np.exp(-r * T)


def mjd_price(
    S: float, K: float, r: float, q: float, sigma: float, T: float,
    lam: float, mu_j: float, delta: float, n_max: int = 50,
    option_type: Literal["call", "put"] = "call"
) -> float:
    """Unified MJD pricer (call or put)."""
    if option_type == "call":
        return mjd_call(S, K, r, q, sigma, T, lam, mu_j, delta, n_max)
    return mjd_put(S, K, r, q, sigma, T, lam, mu_j, delta, n_max)


# =============================================================================
# 4.  EXCHANGE OPTION – MARGRABE (1978)
# =============================================================================

def margrabe_exchange(S1: float, S2: float, q1: float, q2: float,
                      sigma1: float, sigma2: float, rho12: float,
                      T: float) -> float:
    """
    Margrabe exchange option price: payoff = max(S1_T - S2_T, 0).

    C_ex = S1 e^{-q1τ} Φ(d1) - S2 e^{-q2τ} Φ(d2)

    σ_12 = sqrt(σ1² + σ2² - 2ρ12 σ1 σ2)
    d1   = [ln(S1/S2) + (q2-q1+σ12²/2)τ] / (σ12 √τ)
    d2   = d1 - σ12 √τ

    (report eq. 4.1.12–4.1.14)
    """
    sigma12 = np.sqrt(sigma1**2 + sigma2**2 - 2 * rho12 * sigma1 * sigma2)
    d1 = (np.log(S1 / S2) + (q2 - q1 + 0.5 * sigma12**2) * T) / (sigma12 * np.sqrt(T))
    d2 = d1 - sigma12 * np.sqrt(T)
    return (np.exp(-q1 * T) * S1 * stats.norm.cdf(d1)
            - np.exp(-q2 * T) * S2 * stats.norm.cdf(d2))


# =============================================================================
# 5.  SPREAD OPTION – KIRK (1995) APPROXIMATION
# =============================================================================

def kirk_call_spread(S1: float, S2: float, K: float, r: float,
                q1: float, q2: float, sigma1: float, sigma2: float,
                rho12: float, T: float) -> float:
    """
    Kirk approximation for a spread call: payoff = max(S1_T - S2_T - K, 0).

    F1    = S1 e^{(r-q1)τ}
    F2    = S2 e^{(r-q2)τ}
    F̃2   = F2 + K e^{rτ}          (effective second forward)
    b     = F2 / F̃2
    σ_K  = sqrt(σ1² - 2b ρ12 σ1 σ2 + b² σ2²)
    C_spr = e^{-rτ} [F1 Φ(d̂1) - F̃2 Φ(d̂2)]

    (report eq. 4.1.19–4.1.21)
    """
    F1  = S1 * np.exp((r - q1) * T)
    F2  = S2 * np.exp((r - q2) * T)
    F2t = F2 + K * np.exp(r * T)
    b   = F2 / F2t
    sigma_K = np.sqrt(sigma1**2 - 2 * b * rho12 * sigma1 * sigma2 + b**2 * sigma2**2)
    d1 = (np.log(F1 / F2t) + 0.5 * sigma_K**2 * T) / (sigma_K * np.sqrt(T))
    d2 = d1 - sigma_K * np.sqrt(T)
    return np.exp(-r * T) * (F1 * stats.norm.cdf(d1) - F2t * stats.norm.cdf(d2))

def kirk_put_spread(S1: float, S2: float, K: float, r: float,
                q1: float, q2: float, sigma1: float, sigma2: float,
                rho12: float, T: float) -> float:
    """Kirk spread put via put-call parity."""
    call = kirk_call_spread(S1, S2, K, r, q1, q2, sigma1, sigma2, rho12, T)
    F1  = S1 * np.exp((r - q1) * T)
    F2  = S2 * np.exp((r - q2) * T)
    F2t = F2 + K * np.exp(r * T)
    return call - np.exp(-r * T) * (F1 - F2t)


# =============================================================================
# 6.  BASKET OPTION – LOGNORMAL MOMENT-MATCHING
# =============================================================================

def basket_call_lognormal(
    S: np.ndarray, w: np.ndarray, K: float, r: float,
    q: np.ndarray, sigma: np.ndarray, corr: np.ndarray, T: float
) -> float:
    """
    Basket call option approximation via lognormal moment-matching.

    payoff = max(Σ_i w_i S_i^T - K, 0)

    Step 1 – Compute first two moments of the basket at T:
        m_B = Σ_i w_i S_i e^{(r-qi)τ}
        v_B = Σ_{i,j} w_i w_j S_i S_j e^{(r-qi+r-qj)τ} (e^{ρij σi σj τ} - 1)
              + m_B²   ... keeping variance only

    Step 2 – Fit a lognormal: m_B = exp(μ_B + σ_B²/2),  v_B = (exp(σ_B²)-1)exp(2μ_B+σ_B²)

    Step 3 – Apply Black-Scholes with spot = m_B * e^{-rτ} and vol = σ_B / √τ.

    (report eq. 4.1.22–4.1.24)

    Parameters
    ----------
    S    : array of d initial spot prices
    w    : array of d positive weights summing to any positive value
    K    : strike
    r    : risk-free rate
    q    : array of d dividend yields
    sigma: array of d volatilities
    corr : d×d correlation matrix
    T    : time to maturity
    """
    d = len(S)
    # Forward prices (E[w_i S_i^T] discounted)
    F = w * S * np.exp((r - q) * T)
    m_B = F.sum()  # E[basket at T]

    # Variance of basket at T
    v_B = 0.0
    for i in range(d):
        for j in range(d):
            v_B += (w[i] * S[i] * np.exp((r - q[i]) * T) *
                    w[j] * S[j] * np.exp((r - q[j]) * T) *
                    (np.exp(corr[i, j] * sigma[i] * sigma[j] * T) - 1))

    # Lognormal parameters
    sigma_B_sq = np.log(1.0 + v_B / m_B**2)
    mu_B = np.log(m_B) - 0.5 * sigma_B_sq
    sigma_B = np.sqrt(sigma_B_sq)

    # Equivalent Black-Scholes inputs
    S_tilde = np.exp(mu_B + sigma_B_sq) * np.exp(-r * T)  # = m_B * e^{-rτ} adjusted
    vol_bs  = sigma_B / np.sqrt(T)

    return bs_call(S_tilde, K, r, 0.0, vol_bs, T)


def basket_put_lognormal(
    S: np.ndarray, w: np.ndarray, K: float, r: float,
    q: np.ndarray, sigma: np.ndarray, corr: np.ndarray, T: float
) -> float:
    """Basket put via lognormal moment-matching and put-call parity."""
    call = basket_call_lognormal(S, w, K, r, q, sigma, corr, T)
    # Compute the forward price of the basket for put-call parity
    F = w * S * np.exp((r - q) * T)
    m_B = F.sum()
    return call - m_B * np.exp(-r * T) + K * np.exp(-r * T)


def price(Model, Payoff):
    # no analytical formula for American options, so we raise error if user tries to price them via this function
    if Payoff.ApplicationRule == 'american':
        raise ValueError("Analytical pricing is only available for European options. For American options, please use Monte Carlo or PDE methods.")
    # at the moment, dividend paying stock are not handled, all q are set to zero
    if isinstance(Model, BlackScholes):
        return bs_price(S=Model.S0, K=Payoff.K, r=Model.r, q=0, sigma=Model.sigma, T=Payoff.T, option_type=Payoff.payoff_type)
    elif isinstance(Model, Heston):
        return heston_price(S0=Model.S0, K=Payoff.K, v0=Model.v0, T=Payoff.T, r=Model.r,
                            kappa=Model.kappa, theta=Model.theta, xi=Model.xi, rho=1,
                            option_type=Payoff.payoff_type)
    elif isinstance(Model, Merton):
        return mjd_price(S=Model.S0, K=Payoff.K, r=Model.r, q=0, sigma=Model.sigma, T=Payoff.T,
                         lam=Model.lam, mu_j=Model.muJ, delta=Model.sigmaJ,
                         option_type=Payoff.payoff_type)
    elif isinstance(Model, TwoUnderlyings):
        if Payoff.payoff_type == 'exchange':
            return margrabe_exchange(S1=Model.S0_1, S2=Model.S0_2, q1=0, q2=0,
                                     sigma1=Model.sigma1, sigma2=Model.sigma2, rho12=Model.rho, T=Payoff.T)
        elif Payoff.payoff_type == 'call_spread':
            return kirk_call_spread(S1=Model.S0_1, S2=Model.S0_2, K=Payoff.K, r=Model.r,
                               q1=0, q2=0, sigma1=Model.sigma1, sigma2=Model.sigma2,
                               rho12=Model.rho, T=Payoff.T)
        elif Payoff.payoff_type == 'put_spread':
            return kirk_put_spread(S1=Model.S0_1, S2=Model.S0_2, K=Payoff.K, r=Model.r,
                               q1=0, q2=0, sigma1=Model.sigma1, sigma2=Model.sigma2,
                               rho12=Model.rho, T=Payoff.T)
        elif Payoff.payoff_type == 'call_basket':
            return basket_call_lognormal(S=np.array([Model.S0_1, Model.S0_2]),
                                        w=np.array([Payoff.w1, Payoff.w2]),
                                        K=Payoff.K, r=Model.r,
                                        q=np.array([0, 0]),
                                        sigma=np.array([Model.sigma1, Model.sigma2]),
                                        corr=np.array([[1.0, Model.rho], [Model.rho, 1.0]]),
                                        T=Payoff.T)
        elif Payoff.payoff_type == 'put_basket':
            return basket_put_lognormal(S=np.array([Model.S0_1, Model.S0_2]),
                                       w=np.array([Payoff.w1, Payoff.w2]),
                                       K=Payoff.K, r=Model.r,
                                       q=np.array([0, 0]),
                                       sigma=np.array([Model.sigma1, Model.sigma2]),
                                       corr=np.array([[1.0, Model.rho], [Model.rho, 1.0]]),
                                       T=Payoff.T)
        else:
            raise ValueError("Unsupported payoff type for TwoUnderlyings model. User tried to price a payoff of type '{}', but only 'exchange', 'spread' and 'basket' are supported.".format(Payoff.payoff_type))