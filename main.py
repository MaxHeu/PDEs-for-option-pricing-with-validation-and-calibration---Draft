# ==============================================================================
#  Option Pricing Engine — Main Script
#  Models: Black-Scholes, Heston, Merton Jump-Diffusion, Two-Underlyings
#  Calibration: Manual or reverse-engineering observed market data via MSE
#  Methods: Monte Carlo, Analytical, PDE, Trees
# ==============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import time
import typing as tp
from typing import Callable, Literal, Tuple, Union
from datetime import date, timedelta, datetime as dt

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ── Local modules ─────────────────────────────────────────────────────────────
import MonteCarlo as mc
import Analytical as analytical
import PDEs as pdes
import Trees as tr
import Calibration as cal
from Models import *


# ==============================================================================
#  0. Global Configuration
# ==============================================================================

PlotPDEs_Bool = False           # Plot PDE surface for single-underlying european options 

# Treasury yield curve (for risk-free rate calibration)
yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
yields = np.array([0.15, 0.27, 0.50, 0.93, 1.52, 2.13, 2.32,
                   2.34, 2.37, 2.32, 2.65, 2.52], dtype=float) / 100

TRADING_DAYS_PER_YEAR = 252


# ==============================================================================
#  1. Contract Parameters
# ==============================================================================

# ── Single underlying ─────────────────────────────────────────────────────────
Maturity_1u    = "2026-05-15"   # Must be a quoted expiry date
ticker_1u      = "AAPL"
Strike_Call_1u = 115 # Strike for call option on one underlying
Strike_Put_1u  = 85 # Strike for put option on one underlying

# ── Two underlyings ───────────────────────────────────────────────────────────
Maturity_2u          = "2026-05-15"
ticker1              = "AAPL"
ticker2              = "NVDA"
Strike_Call_Spread_2u = 100
Strike_Put_Spread_2u  = 100
Strike_Call_Basket_2u = 100
Strike_Put_Basket_2u  = 100
w1_basket, w2_basket  = 0.5, 0.5


# ==============================================================================
#  2. Model Calibration
# ==============================================================================

Manual_calibration = False # if False -> models calibrated on market data via MSE
Market_calibration = not Manual_calibration

if Manual_calibration:
    # ── Manual (example) parameters ───────────────────────────────────────────
    BS   = BlackScholes(S0=100, r=0.05, sigma=0.2)
    Hes  = Heston(S0=100, r=0.05, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=0.15)
    MJD  = Merton(S0=100, r=0.05, sigma=0.2, lam=0.1, muJ=-0.1, sigmaJ=0.2)
    TwoU = TwoUnderlyings(
        S0_1=100, S0_2=75, r=0.05,
        sigma1=0.2, sigma2=0.25, rho=0.15
    )  # Cholesky decomposition of covariance matrix handled in PDEs.py

else:
    # ── Market calibration (real-life pricing) ────────────────────────────────
    today_d  = date.today()
    tk_check = yf.Ticker(ticker_1u)

    print("=" * 60)
    print(f"  Calibration 1u  |  Ticker: {ticker_1u}  |  Expiry: {Maturity_1u}")
    print("=" * 60)

    params_BS     = cal.calibrate("BS",     ticker_1u, Maturity_1u, yield_maturities, yields)
    params_MJD    = cal.calibrate("MJD",    ticker_1u, Maturity_1u, yield_maturities, yields)
    params_Heston = cal.calibrate("Heston", ticker_1u, Maturity_1u, yield_maturities, yields)

    r_BS, sigma_BS = params_BS["r"], params_BS["sigma"]

    r_Hes, v0_Hes, kappa_Hes, theta_Hes, xi_Hes, rho_Hes = (
        params_Heston["r"],     params_Heston["v0"],    params_Heston["kappa"],
        params_Heston["theta"], params_Heston["xi"],    params_Heston["rho"]
    )

    r_MJD, sigma_MJD, lambda_MJD, mu_y_MJD, sigma_y_MJD = (
        params_MJD["r"],    params_MJD["sigma"],  params_MJD["lambda"],
        params_MJD["mu_y"], params_MJD["sigma_y"]
    )

    last_quote = tk_check.history()["Close"].iloc[-1]

    BS  = BlackScholes(S0=last_quote, r=r_BS, sigma=sigma_BS)
    Hes = Heston(S0=last_quote, r=r_Hes, v0=v0_Hes, kappa=kappa_Hes,
                 theta=theta_Hes, xi=xi_Hes, rho=rho_Hes)
    MJD = Merton(S0=last_quote, r=r_MJD, sigma=sigma_MJD,
                 lam=lambda_MJD, muJ=mu_y_MJD, sigmaJ=sigma_y_MJD)

    # Two-underlying calibration via CBOE methodology
    params_TwoU = cal.calibrate_TwoU(ticker1, ticker2, Maturity_2u, yield_maturities, yields)
    TwoU = TwoUnderlyings(
        S0_1=params_TwoU["S1_0"], S0_2=params_TwoU["S2_0"],
        r=params_TwoU["r_1"],
        sigma1=params_TwoU["sigma_1"], sigma2=params_TwoU["sigma_2"],
        rho=params_TwoU["rho"]
    )

print("Calibration finished")


# ==============================================================================
#  3. Payoff Definitions
# ==============================================================================

today_d = date.today()
MaturityDays_1u = np.busday_count(today_d, Maturity_1u) / TRADING_DAYS_PER_YEAR
MaturityDays_2u = np.busday_count(today_d, Maturity_2u) / TRADING_DAYS_PER_YEAR

# ── Single underlying ─────────────────────────────────────────────────────────
Put_Eu   = PayoffFunction_1u(payoff_type="put",  ApplicationRule="european", K=Strike_Put_1u,  T=MaturityDays_1u)
Call_Eu  = PayoffFunction_1u(payoff_type="call", ApplicationRule="european", K=Strike_Call_1u, T=MaturityDays_1u)
Put_Am   = PayoffFunction_1u(payoff_type="put",  ApplicationRule="american", K=Strike_Put_1u,  T=MaturityDays_1u)
Call_Am  = PayoffFunction_1u(payoff_type="call", ApplicationRule="american", K=Strike_Call_1u, T=MaturityDays_1u)

# ── Two underlyings — European ────────────────────────────────────────────────
CallSpread_Eu  = PayoffFunction_2u(payoff_type="call_spread",  ApplicationRule="european", K=Strike_Call_Spread_2u, T=MaturityDays_2u, w1=1,         w2=1)
PutSpread_Eu   = PayoffFunction_2u(payoff_type="put_spread",   ApplicationRule="european", K=Strike_Put_Spread_2u,  T=MaturityDays_2u, w1=1,         w2=1)
Exchange_Eu    = PayoffFunction_2u(payoff_type="exchange",     ApplicationRule="european", K=0,                     T=MaturityDays_2u, w1=1,         w2=-1)
CallBasket_Eu  = PayoffFunction_2u(payoff_type="call_basket",  ApplicationRule="european", K=Strike_Call_Basket_2u, T=MaturityDays_2u, w1=w1_basket, w2=w2_basket)
PutBasket_Eu   = PayoffFunction_2u(payoff_type="put_basket",   ApplicationRule="european", K=Strike_Put_Basket_2u,  T=MaturityDays_2u, w1=w1_basket, w2=w2_basket)

# ── Two underlyings — American ────────────────────────────────────────────────
CallSpread_Am  = PayoffFunction_2u(payoff_type="call_spread",  ApplicationRule="american", K=Strike_Call_Spread_2u, T=MaturityDays_2u, w1=1,         w2=1)
PutSpread_Am   = PayoffFunction_2u(payoff_type="put_spread",   ApplicationRule="american", K=Strike_Put_Spread_2u,  T=MaturityDays_2u, w1=1,         w2=1)
Exchange_Am    = PayoffFunction_2u(payoff_type="exchange",     ApplicationRule="american", K=0,                     T=MaturityDays_2u, w1=1,         w2=-1)
CallBasket_Am  = PayoffFunction_2u(payoff_type="call_basket",  ApplicationRule="american", K=Strike_Call_Basket_2u, T=MaturityDays_2u, w1=w1_basket, w2=w2_basket)
PutBasket_Am   = PayoffFunction_2u(payoff_type="put_basket",   ApplicationRule="american", K=Strike_Put_Basket_2u,  T=MaturityDays_2u, w1=w1_basket, w2=w2_basket)


# ==============================================================================
#  4. European Option Pricing
# ==============================================================================

num_paths   = 5000
Grid_size1D = [200, 200]        # [S-nodes, time-nodes]
Grid_size2D = [100, 100, 200]   # [S1-nodes, S2/ν-nodes, time-nodes]

# ── 4.1 Monte Carlo ───────────────────────────────────────────────────────────
MC_Price_BS_Put_Eu,          _, _ = mc.price(BS,   Put_Eu,        num_paths)
MC_Price_BS_Call_Eu,         _, _ = mc.price(BS,   Call_Eu,       num_paths)
MC_Price_Hes_Call_Eu,        _, _ = mc.price(Hes,  Call_Eu,       num_paths)
MC_Price_Hes_Put_Eu,         _, _ = mc.price(Hes,  Put_Eu,        num_paths)
MC_Price_MJD_Call_Eu,        _, _ = mc.price(MJD,  Call_Eu,       num_paths)
MC_Price_MJD_Put_Eu,         _, _ = mc.price(MJD,  Put_Eu,        num_paths)
MC_Price_TwoU_CallSpread_Eu, _, _ = mc.price(TwoU, CallSpread_Eu, num_paths)
MC_Price_TwoU_PutSpread_Eu,  _, _ = mc.price(TwoU, PutSpread_Eu,  num_paths)
MC_Price_TwoU_Exchange_Eu,   _, _ = mc.price(TwoU, Exchange_Eu,   num_paths)
MC_Price_TwoU_CallBasket_Eu, _, _ = mc.price(TwoU, CallBasket_Eu, num_paths)
MC_Price_TwoU_PutBasket_Eu,  _, _ = mc.price(TwoU, PutBasket_Eu,  num_paths)
print("European Monte Carlo prices: OK")

# ── 4.2 Analytical ────────────────────────────────────────────────────────────
Analytical_Price_BS_Put_Eu          = analytical.price(BS,   Put_Eu)
Analytical_Price_BS_Call_Eu         = analytical.price(BS,   Call_Eu)
Analytical_Price_Hes_Call_Eu        = analytical.price(Hes,  Call_Eu)
Analytical_Price_Hes_Put_Eu         = analytical.price(Hes,  Put_Eu)
Analytical_Price_MJD_Call_Eu        = analytical.price(MJD,  Call_Eu)
Analytical_Price_MJD_Put_Eu         = analytical.price(MJD,  Put_Eu)
print("European analytical prices (1D): OK")

Analytical_Price_TwoU_CallSpread_Eu = analytical.price(TwoU, CallSpread_Eu)
Analytical_Price_TwoU_PutSpread_Eu  = analytical.price(TwoU, PutSpread_Eu)
Analytical_Price_TwoU_Exchange_Eu   = analytical.price(TwoU, Exchange_Eu)
Analytical_Price_TwoU_CallBasket_Eu = analytical.price(TwoU, CallBasket_Eu)
Analytical_Price_TwoU_PutBasket_Eu  = analytical.price(TwoU, PutBasket_Eu)
print("European analytical prices (2D): OK")

# ── 4.3 PDEs ──────────────────────────────────────────────────────────────────
PDE_Price_BS_Put_Eu   = pdes.solve_pde(BS,  Put_Eu,  Grid_size1D, PlotPDEs_Bool)
PDE_Price_BS_Call_Eu  = pdes.solve_pde(BS,  Call_Eu, Grid_size1D, PlotPDEs_Bool)
PDE_Price_Hes_Call_Eu = pdes.solve_pde(Hes, Call_Eu, Grid_size2D, PlotPDEs_Bool)
PDE_Price_Hes_Put_Eu  = pdes.solve_pde(Hes, Put_Eu,  Grid_size2D, PlotPDEs_Bool)
PDE_Price_MJD_Call_Eu = pdes.solve_pde(MJD, Call_Eu, Grid_size1D, PlotPDEs_Bool)
PDE_Price_MJD_Put_Eu  = pdes.solve_pde(MJD, Put_Eu,  Grid_size1D, PlotPDEs_Bool)
print("European PDE prices (1D): OK")

PDE_Price_TwoU_CallSpread_Eu = pdes.solve_pde(TwoU, CallSpread_Eu, Grid_size2D)
PDE_Price_TwoU_PutSpread_Eu  = pdes.solve_pde(TwoU, PutSpread_Eu,  Grid_size2D)
PDE_Price_TwoU_Exchange_Eu   = pdes.solve_pde(TwoU, Exchange_Eu,   Grid_size2D)
PDE_Price_TwoU_CallBasket_Eu = pdes.solve_pde(TwoU, CallBasket_Eu, Grid_size2D)
PDE_Price_TwoU_PutBasket_Eu  = pdes.solve_pde(TwoU, PutBasket_Eu,  Grid_size2D)
print("European PDE prices (2D): OK")

# ── 4.4 Trees (Black-Scholes dynamics only) ───────────────────────────────────
n = 1000
dt_tree = MaturityDays_1u / n

params_CRR              = tr._params_CRR(BS.r, BS.sigma, dt_tree, S0=BS.S0, n=n)
params_RendlemanBartter = tr._params_RendlemanBartter(BS.r, BS.sigma, dt_tree, S0=BS.S0, n=n)
params_BinomialTian     = tr._params_BinomialTian(BS.r, BS.sigma, dt_tree, S0=BS.S0, n=n)
params_TrinomialTian    = tr._params_TrinomialTian(BS.r, BS.sigma, dt_tree, S0=BS.S0, n=n)
params_Joshi            = lambda K: tr._params_Joshi(BS.r, BS.sigma, dt_tree, S0=BS.S0, n=n, L=K)

bin_CRR              = tr.binomial("CRR",              params_CRR)
bin_RendlemanBartter = tr.binomial("RendlemanBartter",  params_RendlemanBartter)
bin_BinomialTian     = tr.binomial("BinomialTian",      params_BinomialTian)
trin_Tian            = tr.trinomial("TrinomialTian",    params_TrinomialTian)
trin_joshi           = lambda K: tr.trinomial("Joshi",  params_Joshi(K))

# European put — trees
bin_price_CRR_put_eu              = tr.price(bin_CRR,              Put_Eu)
bin_price_RendlemanBartter_put_eu = tr.price(bin_RendlemanBartter, Put_Eu)
bin_price_BinomialTian_put_eu     = tr.price(bin_BinomialTian,     Put_Eu)
trin_price_Tian_put_eu            = tr.price(trin_Tian,            Put_Eu)
trin_price_joshi_put_eu           = tr.price(trin_joshi(Put_Eu.K), Put_Eu)

# European call — trees
bin_price_CRR_call_eu              = tr.price(bin_CRR,               Call_Eu)
bin_price_RendlemanBartter_call_eu = tr.price(bin_RendlemanBartter,  Call_Eu)
bin_price_BinomialTian_call_eu     = tr.price(bin_BinomialTian,      Call_Eu)
trin_price_Tian_call_eu            = tr.price(trin_Tian,             Call_Eu)
trin_price_joshi_call_eu           = tr.price(trin_joshi(Call_Eu.K), Call_Eu)

# ── 4.5 Results ───────────────────────────────────────────────────────────────
print("Price Comparison - European Options")
print("  Note: prices are comparable within a model, not across models (different calibrations)")
print("-" * 20, "One Underlying", "-" * 20)
print(f"Black-Scholes Put:  MC={MC_Price_BS_Put_Eu:.4f},  Analytical={Analytical_Price_BS_Put_Eu:.4f},  PDE={PDE_Price_BS_Put_Eu:.4f}")
print(f"Black-Scholes Call: MC={MC_Price_BS_Call_Eu:.4f},  Analytical={Analytical_Price_BS_Call_Eu:.4f},  PDE={PDE_Price_BS_Call_Eu:.4f}")
print(f"Trees Put:  CRR={bin_price_CRR_put_eu:.4f},  RB={bin_price_RendlemanBartter_put_eu:.4f},  BinTian={bin_price_BinomialTian_put_eu:.4f},  TrinTian={trin_price_Tian_put_eu:.4f},  Joshi={trin_price_joshi_put_eu:.4f}")
print(f"Trees Call: CRR={bin_price_CRR_call_eu:.4f},  RB={bin_price_RendlemanBartter_call_eu:.4f},  BinTian={bin_price_BinomialTian_call_eu:.4f},  TrinTian={trin_price_Tian_call_eu:.4f},  Joshi={trin_price_joshi_call_eu:.4f}")
print(f"Heston Call: MC={MC_Price_Hes_Call_Eu:.4f},  Analytical={Analytical_Price_Hes_Call_Eu:.4f},  PDE={PDE_Price_Hes_Call_Eu:.4f}")
print(f"Heston Put:  MC={MC_Price_Hes_Put_Eu:.4f},  Analytical={Analytical_Price_Hes_Put_Eu:.4f},  PDE={PDE_Price_Hes_Put_Eu:.4f}")
print(f"Merton Call: MC={MC_Price_MJD_Call_Eu:.4f},  Analytical={Analytical_Price_MJD_Call_Eu:.4f},  PDE={PDE_Price_MJD_Call_Eu:.4f}")
print(f"Merton Put:  MC={MC_Price_MJD_Put_Eu:.4f},  Analytical={Analytical_Price_MJD_Put_Eu:.4f},  PDE={PDE_Price_MJD_Put_Eu:.4f}")
print("-" * 20, "Two Underlyings", "-" * 20)
print(f"Call Spread:     MC={MC_Price_TwoU_CallSpread_Eu:.4f},  Analytical={Analytical_Price_TwoU_CallSpread_Eu:.4f},  PDE={PDE_Price_TwoU_CallSpread_Eu:.4f}")
print(f"Put Spread:      MC={MC_Price_TwoU_PutSpread_Eu:.4f},  Analytical={Analytical_Price_TwoU_PutSpread_Eu:.4f},  PDE={PDE_Price_TwoU_PutSpread_Eu:.4f}")
print(f"Exchange Option: MC={MC_Price_TwoU_Exchange_Eu:.4f},  Analytical={Analytical_Price_TwoU_Exchange_Eu:.4f},  PDE={PDE_Price_TwoU_Exchange_Eu:.4f}")
print(f"Call Basket:     MC={MC_Price_TwoU_CallBasket_Eu:.4f},  Analytical={Analytical_Price_TwoU_CallBasket_Eu:.4f},  PDE={PDE_Price_TwoU_CallBasket_Eu:.4f}")
print(f"Put Basket:      MC={MC_Price_TwoU_PutBasket_Eu:.4f},  Analytical={Analytical_Price_TwoU_PutBasket_Eu:.4f},  PDE={PDE_Price_TwoU_PutBasket_Eu:.4f}")


# ==============================================================================
#  5. American Option Pricing
# ==============================================================================

print("=" * 52)
print("  American option prices")
print("=" * 52)

# ── 5.1 Monte Carlo ───────────────────────────────────────────────────────────
MC_Price_BS_Put_Am,   _, _ = mc.price(BS,  Put_Am,  num_paths)
MC_Price_BS_Call_Am,  _, _ = mc.price(BS,  Call_Am, num_paths)
MC_Price_Hes_Call_Am, _, _ = mc.price(Hes, Call_Am, num_paths)
MC_Price_Hes_Put_Am,  _, _ = mc.price(Hes, Put_Am,  num_paths)
MC_Price_MJD_Call_Am, _, _ = mc.price(MJD, Call_Am, num_paths)
MC_Price_MJD_Put_Am,  _, _ = mc.price(MJD, Put_Am,  num_paths)
print("American Monte Carlo prices: OK")

# ── 5.2 PDEs ──────────────────────────────────────────────────────────────────
PDE_Price_BS_Put_Am   = pdes.solve_pde(BS,  Put_Am,  Grid_size1D)
PDE_Price_BS_Call_Am  = pdes.solve_pde(BS,  Call_Am, Grid_size1D)
PDE_Price_Hes_Call_Am = pdes.solve_pde(Hes, Call_Am, Grid_size2D)
PDE_Price_Hes_Put_Am  = pdes.solve_pde(Hes, Put_Am,  Grid_size2D)
PDE_Price_MJD_Call_Am = pdes.solve_pde(MJD, Call_Am, Grid_size1D)
PDE_Price_MJD_Put_Am  = pdes.solve_pde(MJD, Put_Am,  Grid_size1D)
print("American PDE prices (1D): OK")

PDE_Price_TwoU_CallBasket_Am = pdes.solve_pde(TwoU, CallBasket_Am, Grid_size2D)
PDE_Price_TwoU_PutBasket_Am  = pdes.solve_pde(TwoU, PutBasket_Am,  Grid_size2D)
PDE_Price_TwoU_Exchange_Am   = pdes.solve_pde(TwoU, Exchange_Am,   Grid_size2D)
PDE_Price_TwoU_CallSpread_Am = pdes.solve_pde(TwoU, CallSpread_Am, Grid_size2D)
PDE_Price_TwoU_PutSpread_Am  = pdes.solve_pde(TwoU, PutSpread_Am,  Grid_size2D)
print("American PDE prices (2D): OK")

# ── 5.3 Trees (Black-Scholes dynamics only) ───────────────────────────────────
# American put — trees
bin_price_CRR_put_Am              = tr.price(bin_CRR,              Put_Am)
bin_price_RendlemanBartter_put_Am = tr.price(bin_RendlemanBartter, Put_Am)
bin_price_BinomialTian_put_Am     = tr.price(bin_BinomialTian,     Put_Am)
trin_price_Tian_put_Am            = tr.price(trin_Tian,            Put_Am)
trin_price_joshi_put_Am           = tr.price(trin_joshi(Put_Am.K), Put_Am)

# American call — trees
bin_price_CRR_call_Am              = tr.price(bin_CRR,               Call_Am)
bin_price_RendlemanBartter_call_Am = tr.price(bin_RendlemanBartter,  Call_Am)
bin_price_BinomialTian_call_Am     = tr.price(bin_BinomialTian,      Call_Am)
trin_price_Tian_call_Am            = tr.price(trin_Tian,             Call_Am)
trin_price_joshi_call_Am           = tr.price(trin_joshi(Call_Am.K), Call_Am)

# ── 5.4 Results ───────────────────────────────────────────────────────────────
print("Price Comparison - American Options")
print("  Note: LSMC not implemented for 2D payoffs")
print("-" * 20, "One Underlying", "-" * 20)
print(f"Black-Scholes Put:  MC={MC_Price_BS_Put_Am:.4f},  PDE={PDE_Price_BS_Put_Am:.4f}")
print(f"Black-Scholes Call: MC={MC_Price_BS_Call_Am:.4f},  PDE={PDE_Price_BS_Call_Am:.4f}")
print(f"Trees Put:  CRR={bin_price_CRR_put_Am:.4f},  RB={bin_price_RendlemanBartter_put_Am:.4f},  BinTian={bin_price_BinomialTian_put_Am:.4f},  TrinTian={trin_price_Tian_put_Am:.4f},  Joshi={trin_price_joshi_put_Am:.4f}")
print(f"Trees Call: CRR={bin_price_CRR_call_Am:.4f},  RB={bin_price_RendlemanBartter_call_Am:.4f},  BinTian={bin_price_BinomialTian_call_Am:.4f},  TrinTian={trin_price_Tian_call_Am:.4f},  Joshi={trin_price_joshi_call_Am:.4f}")
print(f"Heston Call: MC={MC_Price_Hes_Call_Am:.4f},  PDE={PDE_Price_Hes_Call_Am:.4f}")
print(f"Heston Put:  MC={MC_Price_Hes_Put_Am:.4f},  PDE={PDE_Price_Hes_Put_Am:.4f}")
print(f"Merton Call: MC={MC_Price_MJD_Call_Am:.4f},  PDE={PDE_Price_MJD_Call_Am:.4f}")
print(f"Merton Put:  MC={MC_Price_MJD_Put_Am:.4f},  PDE={PDE_Price_MJD_Put_Am:.4f}")
print("-" * 20, "Two Underlyings", "-" * 20)
print(f"Call Spread:     PDE={PDE_Price_TwoU_CallSpread_Am:.4f}")
print(f"Put Spread:      PDE={PDE_Price_TwoU_PutSpread_Am:.4f}")
print(f"Exchange Option: PDE={PDE_Price_TwoU_Exchange_Am:.4f}")
print(f"Call Basket:     PDE={PDE_Price_TwoU_CallBasket_Am:.4f}")
print(f"Put Basket:      PDE={PDE_Price_TwoU_PutBasket_Am:.4f}")
