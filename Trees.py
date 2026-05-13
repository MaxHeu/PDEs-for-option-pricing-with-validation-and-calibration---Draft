import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Callable, Union
from Models import Tree, PayoffFunction_1u

# ---------------------------------------------------------------------------
# Model parametrisations
# ---------------------------------------------------------------------------

def _params_CRR(r: float, sigma: float, dt: float, S0: float = None, n: int = None) -> dict:
    """Cox, Ross & Rubinstein (1979)."""
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p_u = (np.exp(r * dt) - d) / (u - d)
    p_d = 1 - p_u
    return dict(u=u, d=d, p_u=p_u, p_d=p_d, S0=S0, n=n, dt=dt, r=r)

def _params_RendlemanBartter(r: float, sigma: float, dt: float, S0: float = None, n: int = None) -> dict:
    """Rendleman & Bartter (1979)."""
    u = np.exp((r - sigma**2 / 2) * dt + sigma * np.sqrt(dt))
    d = np.exp((r - sigma**2 / 2) * dt - sigma * np.sqrt(dt))
    return dict(u=u, d=d, p_u=0.5, p_d=0.5, S0=S0, n=n, dt=dt, r=r)

def _params_BinomialTian(r: float, sigma: float, dt: float, S0: float = None, n: int = None) -> dict:
    """Tian (1993) binomial — moment-matching."""
    ert = np.exp(r * dt)
    V = np.exp(sigma**2 * dt)
    sq = np.sqrt(V**2 + 2*V - 3)
    u = 0.5 * ert * V * (V + 1 + sq)
    d = 0.5 * ert * V * (V + 1 - sq)
    p_u = (ert - d) / (u - d)
    p_d = 1 - p_u
    return dict(u=u, d=d, p_u=p_u, p_d=p_d, S0=S0, n=n, dt=dt, r=r)

def _params_TrinomialTian(r: float, sigma: float, dt: float, S0: float = None, n: int = None) -> dict:
    """Tian (1993) trinomial."""
    M = np.exp(r * dt)
    V = np.exp(sigma**2 * dt)
    m = M * V**2
    Kt = (M / 2) * (V**4 + V**3)
    sq = np.sqrt(Kt**2 - m**2)
    u = Kt + sq
    d = Kt - sq
    p_u = (m*d - M*(m+d) + M**2*V) / ((u - d)*(u - m))
    p_d = (u*m - M*(u+m) + M**2*V) / ((u - d)*(m - d))
    p_m = 1 - p_u - p_d
    return dict(u=u, d=d, m=m, p_u=p_u, p_d=p_d, p_m=p_m, S0=S0, n=n, dt=dt, r=r)

def _params_Joshi(r: float, sigma: float, dt: float,
                  S0: float, n: int, L: float) -> dict:
    """Joshi trinomial (adjusted Tian). L : reference level (typically strike K)."""
    M = np.exp(r * dt)
    V = np.exp(sigma**2 * dt)
    m = (L / S0) ** (1 / n)
    Kj = (V / 2) * (M*V + m) + (m / (2*M)) * (m - M)
    sq = np.sqrt(Kj**2 - m**2)
    u = Kj + sq
    d = Kj - sq
    p_u = (m*d - M*(m+d) + M**2*V) / ((u - d)*(u - m))
    p_d = (u*m - M*(u+m) + M**2*V) / ((u - d)*(m - d))
    p_m = 1 - p_u - p_d
    return dict(u=u, d=d, m=m, p_u=p_u, p_d=p_d, p_m=p_m, S0=S0, n=n, dt=dt, r=r)

# ---------------------------------------------------------------------------
# Tree builders
# ---------------------------------------------------------------------------

BINOMIAL_MODELS  = ("CRR", "RendlemanBartter", "BinomialTian", "Joshi")
TRINOMIAL_MODELS = ("TrinomialTian", "Joshi")

def binomial(
    model  : str,
    params : dict,
) -> Tree:
    """
    Build a recombining binomial stock-price tree.

    Parameters
    ----------
    model  : 'CRR' | 'RendlemanBartter' | 'BinomialTian' | 'Joshi'
    params : dict returned by the matching _params_* helper.
             Must contain keys: u, d, p_u, p_d, S0, n, dt, r.
             (Joshi params also contain m but it is unused in the binomial engine.)

    Returns
    -------
    Tree -- stock_tree[i, j] = price at step i with j down-moves. Shape (n+1, n+1).
    """
    u, d   = params["u"],   params["d"]
    p_u    = params["p_u"]
    p_d    = params["p_d"]
    S0     = params["S0"]
    n      = params["n"]
    dt     = params["dt"]
    r      = params["r"]
    T      = dt * n

    if S0 is None or n is None:
        raise ValueError(
            "params must contain 'S0' and 'n'. "
            "Pass them to the _params_* function: e.g. _params_CRR(r, sigma, dt, S0=S0, n=n)."
        )

    S = np.zeros((n + 1, n + 1))
    S[0, 0] = S0
    for i in range(1, n + 1):
        S[i, 0] = S[i - 1, 0] * u
        for j in range(1, i + 1):
            S[i, j] = S[i - 1, j - 1] * d

    return Tree(
        tree_type="binomial", model=model, stock_tree=S,
        r=r, T=T, n=n, dt=dt, p_u=p_u, p_d=p_d,
    )


def trinomial(
    model  : str,
    params : dict,
) -> Tree:
    """
    Build a recombining trinomial stock-price tree.

    Parameters
    ----------
    model  : 'TrinomialTian' | 'Joshi'
    params : dict returned by _params_TrinomialTian or _params_Joshi.
             Must contain keys: u, d, m, p_u, p_d, p_m, S0, n, dt, r.

    Returns
    -------
    Tree -- stock_tree shape (n+1, 2n+1).
    """
    u, d, m       = params["u"],   params["d"],   params["m"]
    p_u, p_d, p_m = params["p_u"], params["p_d"], params["p_m"]
    S0            = params["S0"]
    n             = params["n"]
    dt            = params["dt"]
    r             = params["r"]
    T             = dt * n

    if S0 is None or n is None:
        raise ValueError(
            "params must contain 'S0' and 'n'. "
            "Pass them to _params_TrinomialTian / _params_Joshi."
        )

    S = np.zeros((n + 1, 2 * n + 1))
    S[0, 0] = S0
    for i in range(1, n + 1):
        S[i, 0] = S[i - 1, 0] * u
        for j in range(1, 2 * i):
            S[i, j] = S[i - 1, j - 1] * m
        S[i, 2 * i] = S[i - 1, 2 * (i - 1)] * d

    return Tree(
        tree_type="trinomial", model=model, stock_tree=S,
        r=r, T=T, n=n, dt=dt, p_u=p_u, p_d=p_d, p_m=p_m,
    )

# ---------------------------------------------------------------------------
# Backward-induction pricing engine  (internal)
# ---------------------------------------------------------------------------

def _price_option(
    tree   : Tree,
    payoff : Callable[[np.ndarray], np.ndarray],
    style  : Literal["european", "american"] = "european",
) -> float:
    """Price a European or American option via backward induction."""
    n, dt, r = tree.n, tree.dt, tree.r
    S, disc  = tree.stock_tree, np.exp(-r * dt)

    if tree.tree_type == "binomial":
        p_u, p_d = tree.p_u, tree.p_d
        V = payoff(S[n, :n + 1])
        for i in range(n - 1, -1, -1):
            V_cont = disc * (p_u * V[:i + 1] + p_d * V[1:i + 2])
            V = np.maximum(V_cont, payoff(S[i, :i + 1])) if style == "american" else V_cont

    elif tree.tree_type == "trinomial":
        p_u, p_m, p_d = tree.p_u, tree.p_m, tree.p_d
        V = payoff(S[n, :2 * n + 1])
        for i in range(n - 1, -1, -1):
            k = 2 * i + 1
            V_cont = disc * (p_u * V[:k] + p_m * V[1:k + 1] + p_d * V[2:k + 2])
            V = np.maximum(V_cont, payoff(S[i, :k])) if style == "american" else V_cont
    else:
        raise ValueError(f"Unknown tree_type: {tree.tree_type!r}")

    return float(V[0])

# ---------------------------------------------------------------------------
# Public entry-point  —  matches main.py signature: price(tree, payoff_obj)
# ---------------------------------------------------------------------------

def price(
    tree         : Tree,
    payoff_obj   : PayoffFunction_1u,
) -> float:
    """
    Price an option on a pre-built Tree using a PayoffFunction_1u descriptor.

    Parameters
    ----------
    tree       : Tree built by binomial() or trinomial()
    payoff_obj : PayoffFunction_1u instance (carries payoff_type, K, ApplicationRule)

    Returns
    -------
    float -- option price at t = 0
    """
    K     = payoff_obj.K
    style = payoff_obj.ApplicationRule  # 'european' or 'american'

    if payoff_obj.payoff_type == "call":
        payoff_fn = lambda S: np.maximum(S - K, 0.0)
    elif payoff_obj.payoff_type == "put":
        payoff_fn = lambda S: np.maximum(K - S, 0.0)
    else:
        raise ValueError(f"Unsupported payoff_type: {payoff_obj.payoff_type!r}")

    return _price_option(tree, payoff_fn, style)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import scipy.stats as ss

    S0, K, T, r, sigma, n = 100, 110, 1.0, 0.05, 0.2, 1000
    dt = T / n

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_call = S0 * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)
    bs_put  = K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * ss.norm.cdf(-d1)

    from Models import PayoffFunction_1u
    call_eu = PayoffFunction_1u(payoff_type='call', ApplicationRule='european', K=K, T=T)
    put_eu  = PayoffFunction_1u(payoff_type='put',  ApplicationRule='european', K=K, T=T)
    put_am  = PayoffFunction_1u(payoff_type='put',  ApplicationRule='american', K=K, T=T)

    print(f"{'Model':<22} {'Eur Call':>10} {'Eur Put':>10} {'Am Put':>10}")
    print("-" * 55)

    for ttype, mdl, builder, pfunc in [
        ("binomial",  "CRR",              binomial,  _params_CRR),
        ("binomial",  "RendlemanBartter", binomial,  _params_RendlemanBartter),
        ("binomial",  "BinomialTian",     binomial,  _params_BinomialTian),
        ("trinomial", "TrinomialTian",    trinomial, _params_TrinomialTian),
    ]:
        params = pfunc(r, sigma, dt, S0=S0, n=n)
        tree   = builder(mdl, params)
        print(f"{mdl:<22} {price(tree, call_eu):>10.4f} {price(tree, put_eu):>10.4f} {price(tree, put_am):>10.4f}")

    params_joshi = _params_Joshi(r, sigma, dt, S0, n, K)
    tree_joshi   = binomial("Joshi", params_joshi)
    print(f"{'Joshi':<22} {price(tree_joshi, call_eu):>10.4f} {price(tree_joshi, put_eu):>10.4f} {price(tree_joshi, put_am):>10.4f}")

    print("-" * 52)
    print(f"{'Black-Scholes':<22} {bs_call:>10.4f} {bs_put:>10.4f}")
