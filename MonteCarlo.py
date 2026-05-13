import numpy as np
from scipy import stats
from typing import Callable, Literal, Tuple
from Models import BlackScholes, Heston, Merton, TwoUnderlyings
# =============================================================================
# 1. PATH SIMULATORS
# =============================================================================

def simulate_black_scholes(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    N: int,
    M: int,
    seed: int = 42
) -> np.ndarray:
    """
    Simulate M paths of GBM under risk-neutral measure.

    SDE: dS = r*S*dt + sigma*S*dW
    Exact log-normal discretisation (no Euler bias):
    S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Returns
    -------
    paths : ndarray of shape (M, N+1)
        paths[:, 0] = S0, paths[:, k] = S(k*dt)
    times : ndarray of shape (N+1,)
        Uniform time grid from 0 to T.
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    times = np.linspace(0.0, T, N + 1)
    paths = np.empty((M, N + 1))
    paths[:, 0] = S0
    Z = rng.standard_normal((M, N))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return paths, times


def simulate_heston(
    S0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    N: int,
    M: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate M paths of the Heston stochastic-volatility model (Euler-Maruyama
    on the variance process with full truncation to keep v >= 0).

    SDEs (risk-neutral):
        dS = r*S*dt + sqrt(v)*S*dW1
        dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW2
        corr(dW1, dW2) = rho

    Returns
    -------
    paths : ndarray of shape (M, N+1)  — asset price paths only
    times : ndarray of shape (N+1,)
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    times = np.linspace(0.0, T, N + 1)
    paths = np.empty((M, N + 1))
    v = np.empty((M, N + 1))
    paths[:, 0] = S0
    v[:, 0] = v0
    sqrt_1mrho2 = np.sqrt(1.0 - rho**2)

    for k in range(N):
        Z1 = rng.standard_normal(M)
        Z2 = rho * Z1 + sqrt_1mrho2 * rng.standard_normal(M)
        sqrt_dt = np.sqrt(dt)
        v_pos = np.maximum(v[:, k], 0.0)
        v[:, k + 1] = (v[:, k]
                       + kappa * (theta - v_pos) * dt
                       + xi * np.sqrt(v_pos) * sqrt_dt * Z2)
        paths[:, k + 1] = paths[:, k] * np.exp(
            (r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * Z1
        )

    return paths, times


def simulate_merton_jump_diffusion(
    S0: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    T: float,
    N: int,
    M: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate M paths of the Merton Jump-Diffusion model.

    SDE (risk-neutral):
        dS/S = (r - lambda*k)*dt + sigma*dW + (J-1)*dN
        J = exp(Y),  Y ~ N(mu_j, sigma_j^2)
        k = E[J-1] = exp(mu_j + 0.5*sigma_j^2) - 1  (compensator)

    Returns
    -------
    paths : ndarray of shape (M, N+1)
    times : ndarray of shape (N+1,)
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    times = np.linspace(0.0, T, N + 1)
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    drift_per_step = (r - lam * k - 0.5 * sigma**2) * dt

    paths = np.empty((M, N + 1))
    paths[:, 0] = S0

    for step in range(N):
        Z = rng.standard_normal(M)
        n_jumps = rng.poisson(lam * dt, size=M)
        max_n = int(n_jumps.max()) if n_jumps.max() > 0 else 0
        if max_n > 0:
            Y = rng.normal(mu_j, sigma_j, (M, max_n))
            mask = np.arange(max_n)[None, :] < n_jumps[:, None]
            jump_sum = (Y * mask).sum(axis=1)
        else:
            jump_sum = np.zeros(M)

        log_ret = drift_per_step + sigma * np.sqrt(dt) * Z + jump_sum
        paths[:, step + 1] = paths[:, step] * np.exp(log_ret)

    return paths, times


def simulate_multi_asset(
    S0: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    T: float,
    N: int,
    M: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate M paths of d correlated GBMs under risk-neutral measure.

    SDE for asset i:
        dS_i = r_i * S_i * dt + sigma_i * S_i * dW_i
        corr(dW_i, dW_j) = corr[i,j]

    Returns
    -------
    paths : ndarray of shape (M, d, N+1)
        paths[:, i, k] = price of asset i at time step k for path m
    times : ndarray of shape (N+1,)
    """
    rng = np.random.default_rng(seed)
    d = len(S0)
    dt = T / N
    times = np.linspace(0.0, T, N + 1)
    cov = np.diag(sigma) @ corr @ np.diag(sigma)
    L = np.linalg.cholesky(cov)

    paths = np.empty((M, d, N + 1))
    for i in range(d):
        paths[:, i, 0] = S0[i]

    for step in range(N):
        Z = rng.standard_normal((d, M))
        corr_Z = L @ Z
        for i in range(d):
            log_ret = ((r[i] - 0.5 * sigma[i]**2) * dt
                       + np.sqrt(dt) * corr_Z[i, :])
            paths[:, i, step + 1] = paths[:, i, step] * np.exp(log_ret)

    return paths, times


# =============================================================================
# 2. UNIFIED MONTE CARLO PRICER - Handles both European and American options with confidence intervals
# =============================================================================

def monte_carlo_price(
    paths: np.ndarray,
    style: Literal["european", "american"],
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    r: float,
    T: float,
    times: np.ndarray = None,
    poly_degree: int = 3,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Unified Monte Carlo pricer for European and American options.

    Parameters
    ----------
    paths : ndarray
        Simulated asset paths. Shape (M, N+1) for single-asset models,
        or (M, d, N+1) for multi-asset models.
        For American pricing only single-asset paths (M, N+1) are supported.
    style : {"european", "american"}
        Exercise style.
        - "european" : standard discounted expectation of terminal payoff.
        - "american" : Longstaff-Schwartz regression-based early exercise.
    payoff_fn : callable
        Maps a 1-D array of spot prices (shape (M,)) to a 1-D array of
        intrinsic payoffs (shape (M,)).
        For European multi-asset options, the lambda receives the full paths
        array, so design it accordingly (e.g. lambda paths: np.maximum(...)).
        For American options, the lambda must accept a 1-D spot vector.
        Examples:
            European put  -> lambda x: np.maximum(K - x[:, -1], 0)
            American put  -> lambda x: np.maximum(K - x, 0)
    r : float
        Continuous risk-free rate used for discounting.
    T : float
        Time to maturity (used for European discounting).
    times : ndarray of shape (N+1,), optional
        Time grid — required for "american" style (provides non-uniform dt).
        Automatically inferred as linspace(0, T, N+1) if not provided.
    poly_degree : int
        Degree of polynomial regression for continuation value (American only,
        default 3).
    confidence : float
        Confidence level for the returned interval (default 0.95).

    Returns
    -------
    price : float   — MC option price estimate
    ci_lo : float   — lower bound of the confidence interval
    ci_hi : float   — upper bound of the confidence interval
    """
    style = style.lower().strip()

    if style not in ("european", "american"):
        raise ValueError(f"style must be 'european' or 'american', got '{style}'.")

    # ── European ─────────────────────────────────────────────────────────────
    if style == "european":
        disc = np.exp(-r * T)
        payoffs = payoff_fn(paths)          # callable gets full paths
        disc_pv = disc * payoffs

    # ── American (Longstaff-Schwartz) ────────────────────────────────────────
    else:
        if paths.ndim != 2:
            raise ValueError(
                "American pricing requires single-asset paths of shape (M, N+1)."
            )
        M_paths, Np1 = paths.shape
        N = Np1 - 1

        if times is None:
            times = np.linspace(0.0, T, N + 1)

        # Step 1: initialise cash flows at maturity
        cashflow = payoff_fn(paths[:,-1]).copy().astype(float)

        # Step 2: backward induction
        for i in reversed(range(1, N)):
            dt = times[i + 1] - times[i]
            cashflow *= np.exp(-r * dt)

            x = paths[:, i]
            exercise = payoff_fn(x)
            itm = exercise > 0

            if itm.sum() > poly_degree:
                fitted = np.polynomial.Polynomial.fit(
                    x[itm], cashflow[itm], poly_degree
                )
                continuation = fitted(x)
                ex_idx = itm & (exercise > continuation)
                cashflow[ex_idx] = exercise[ex_idx]

        # Step 3: discount from t_1 to t_0
        dt0 = times[1] - times[0]
        disc_pv = cashflow * np.exp(-r * dt0)

    # ── Confidence interval (shared) ─────────────────────────────────────────
    M_cf = len(disc_pv)
    price = disc_pv.mean()
    stderr = disc_pv.std(ddof=1) / np.sqrt(M_cf)
    z = stats.norm.ppf(0.5 + confidence / 2.0)

    return price, price - z * stderr, price + z * stderr


def price(Model, Payoff, num_paths = 1000):
    # overwrite func in case of american option
    if Payoff.ApplicationRule == 'american':
        if Payoff.payoff_type == 'call':
            Payoff.payoff_func = lambda x: np.maximum(x - Payoff.K, 0)
        elif Payoff.payoff_type == 'put':
            Payoff.payoff_func = lambda x: np.maximum(Payoff.K - x, 0)
        else:
            raise ValueError("Unsupported payoff type for American option")
    if isinstance(Model, BlackScholes):
        paths, times = simulate_black_scholes(
            S0=Model.S0,
            r=Model.r,
            sigma=Model.sigma,
            T=Payoff.T,
            N=100,  # time steps for path simulation
            M=num_paths
        )
        price, ci_lo, ci_hi = monte_carlo_price(
            paths=paths,
            style=Payoff.ApplicationRule,
            payoff_fn= Payoff.payoff_func,  # terminal payoff
            r=Model.r,
            T=Payoff.T,
            times=None  # uniform grid for European; not used for American here
        )
        return price, ci_lo, ci_hi
    elif isinstance(Model, Heston):
        paths, times = simulate_heston(
            S0=Model.S0,
            v0=Model.v0,
            r=Model.r,
            kappa=Model.kappa,
            theta=Model.theta,
            xi=Model.xi,
            rho=0.0,  # Assuming zero correlation for simplicity
            T=Payoff.T,
            N=100,
            M=num_paths
        )
        price, ci_lo, ci_hi = monte_carlo_price(
            paths=paths,
            style=Payoff.ApplicationRule,
            payoff_fn=Payoff.payoff_func,
            r=Model.r,
            T=Payoff.T,
            times=None
        )
        return price, ci_lo, ci_hi
    elif isinstance(Model, Merton):
        paths, times = simulate_merton_jump_diffusion(
            S0=Model.S0,
            r=Model.r,
            sigma=Model.sigma,
            lam=Model.lam,
            mu_j=Model.muJ,
            sigma_j=Model.sigmaJ,
            T=Payoff.T,
            N=100,
            M=num_paths
        )
        price, ci_lo, ci_hi = monte_carlo_price(
            paths=paths,
            style=Payoff.ApplicationRule,
            payoff_fn=Payoff.payoff_func,
            r=Model.r,
            T=Payoff.T,
            times=None
        )
        return price, ci_lo, ci_hi
    elif isinstance(Model, TwoUnderlyings):
        S0 = np.array([Model.S0_1, Model.S0_2])
        r = np.array([Model.r, Model.r])  # Assuming same rate for both assets
        sigma = np.array([Model.sigma1, Model.sigma2]) # Volatilities for each asset
        corr = np.array([[1.0, Model.rho], [Model.rho, 1.0]])
        paths, times = simulate_multi_asset(
            S0=S0,
            r=r,
            sigma=sigma, 
            corr=corr,
            T=Payoff.T,
            N=100, # time steps for path simulation
            M=num_paths
        )
        price, ci_lo, ci_hi = monte_carlo_price(
            paths=paths,
            style=Payoff.ApplicationRule,
            payoff_fn=Payoff.payoff_func,  # full paths for multi-asset payoff
            r=Model.r,
            T=Payoff.T,
            times=None
        )
        return price, ci_lo, ci_hi
    elif isinstance(Model, BlackScholes) and Payoff.ApplicationRule == 'american':
        paths, times = simulate_black_scholes(
            S0=Model.S0,
            r=Model.r,
            sigma=Model.sigma,
            T=Payoff.T,
            N=100,
            M=num_paths
        )
        price, ci_lo, ci_hi = monte_carlo_price(
            paths=paths,
            style='american',
            payoff_fn=Payoff.payoff_func,  # payoff on spot for American
            r=Model.r,
            T=Payoff.T,
            times=times  # non-uniform grid for American exercise
        )
        return price, ci_lo, ci_hi
    
    else:
        raise NotImplementedError("Monte Carlo pricing not implemented for this model.")    
