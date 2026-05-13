# PDEs-for-option-pricing-with-validation-and-calibration
Numerical methods to price european and american options for one or two underlyings. Includes code for market calibration via yfinance

# Option Pricing via PDEs: Theory to Market Implementation

A comprehensive end-to-end framework for derivative pricing using Partial Differential Equations (PDEs), bridging rigorous mathematical theory with practical market calibration.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📚 Overview

This repository implements numerical methods for option pricing across multiple stochastic models, with full derivations, validation, and market calibration capabilities. The framework covers:

- **Models**: Black-Scholes (GBM), Heston (stochastic volatility), Multi-Asset (correlated underlyings), Merton Jump-Diffusion
- **Products**: European/American options (call/put), Exchange options, Spread options, Basket options
- **Methods**: Finite Difference Methods, ADI schemes, Operator Splitting, Monte Carlo, Tree methods
- **Calibration**: Real-world market data integration (implied volatility, MSE minimization, CBOE equicorrelation)

The accompanying [technical report](main.pdf) provides complete mathematical derivations, numerical analysis, and validation studies.

## 🎯 Key Features

- **End-to-end pipeline**: From stochastic differential equations → PDE derivation → numerical solution → market calibration → pricing
- **Multiple validation methods**: Analytical formulas, Monte Carlo (including Longstaff-Schwartz), Binomial/Trinomial trees
- **Computational efficiency**: Non-uniform grids, ADI time-stepping, convolution-based integral evaluation
- **Real market data**: Calibration on live option chains (AAPL, NVDA, SPY, TTE)
- **American options**: Linear Complementarity Problem (LCP) formulation with early exercise constraints

## 📂 Repository Structure

```
.
├── main.py              # Main execution script with examples
├── PDEs.py              # PDE/PIDE solvers (Black-Scholes, Heston, Multi-Asset, Merton)
├── MonteCarlo.py        # Monte Carlo and Longstaff-Schwartz implementations
├── Trees.py             # Binomial/Trinomial tree methods (CRR, Tian, Joshi, etc.)
├── Analytical.py        # Closed-form solutions (Black-Scholes, Heston CF, Margrabe, Kirk)
├── Calibration.py       # Market calibration routines (IV fitting, MSE optimization)
├── Models.py            # Model parameter dataclasses and asset dynamics
└── main.pdf             # Complete technical report (95 pages)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/option-pricing-pdes.git
cd option-pricing-pdes

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from Models import BlackScholesModel, HestonModel
import PDEs as pdes
import Analytical as analytical

# Example 1: Black-Scholes European Call
bs_model = BlackScholesModel(S0=100, r=0.05, sigma=0.2)
pde_price = pdes.black_scholes_pde(
    model=bs_model, K=100, T=1.0, 
    option_type='call', style='european'
)
analytical_price = analytical.black_scholes_call(bs_model, K=100, T=1.0)

print(f"PDE Price: {pde_price:.4f}")
print(f"Analytical Price: {analytical_price:.4f}")

# Example 2: Heston Model with ADI solver
heston_model = HestonModel(
    S0=100, r=0.05, v0=0.04, 
    kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7
)
price = pdes.heston_adi(
    model=heston_model, K=100, T=1.0, 
    option_type='call', style='european'
)

# Example 3: Market Calibration
from Calibration import calibrate_heston_market
import yfinance as yf

# Fetch market data and calibrate
ticker = yf.Ticker("AAPL")
calibrated_model = calibrate_heston_market(
    ticker=ticker, 
    maturity_days=30
)
```

## 🧮 Implemented Models

### 1. Black-Scholes (Geometric Brownian Motion)
```
dS_t = r S_t dt + σ S_t dW_t
```
- **Solver**: Fully implicit finite difference scheme
- **Products**: European/American calls and puts
- **Validation**: Analytical formulas

### 2. Heston (Stochastic Volatility)
```
dS_t = r S_t dt + √ν_t S_t dW₁_t
dν_t = κ(θ - ν_t)dt + σ_ν √ν_t dW₂_t
```
- **Solver**: ADI (Alternating Direction Implicit) on non-uniform grids
- **Products**: European/American options
- **Validation**: Characteristic function inversion, Monte Carlo

### 3. Multi-Asset (Correlated Black-Scholes)
```
dS^i_t = r S^i_t dt + σ_i S^i_t Σ_ij dW^j_t
```
- **Solver**: 2D ADI finite differences
- **Products**: Exchange options, spread options, basket options
- **Validation**: Margrabe formula, Kirk approximation

### 4. Merton Jump-Diffusion
```
dS_t = r S_t dt + σ S_t dW_t + S_t dJ_t
```
- **Solver**: Operator splitting (diffusion + jumps)
- **Products**: European options
- **Validation**: Merton infinite series, Monte Carlo

## 📊 Validation & Calibration

All numerical methods are validated against:
- ✅ **Analytical solutions** (where available)
- ✅ **Monte Carlo simulations** (standard + Longstaff-Schwartz for American)
- ✅ **Tree methods** (CRR, Rendleman-Bartter, Binomial Tian, Trinomial Tian, Joshi)

Market calibration supports:
- 🎯 **Implied volatility** surface fitting (Black-Scholes)
- 🎯 **MSE minimization** on option chains (Heston, Merton)
- 🎯 **CBOE equicorrelation** methodology (Multi-Asset)

## 📖 Mathematical Background

The repository implements the complete PDE framework:

1. **Derivation**: Start from risk-neutral SDEs, apply Itô's lemma and martingale pricing
2. **PDE formulation**: Derive Black-Scholes-type PDEs/PIDEs for each model
3. **Numerical methods**: Finite differences, ADI, operator splitting
4. **Boundary conditions**: No-arbitrage constraints and payoff functions
5. **Validation**: Multi-method comparison for accuracy verification

See [`main.pdf`](main.pdf) for complete derivations, proofs, and numerical analysis.

## 📈 Example Results

### Validation Accuracy
Comparison of PDE solver vs analytical solution for Black-Scholes European call:
- Mean relative error: < 0.01%
- Maximum absolute error: < $0.05

### Market Calibration
AAPL Heston calibration (liquid market):
- Successfully fitted to 50+ option strikes
- Implied volatility surface RMSE: 2.3%

TTE calibration (illiquid market):
- Sparse data led to parameter non-identifiability
- **Key insight**: Data quality critically impacts calibration success

## ⚙️ Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
yfinance>=0.1.70
pandas>=1.3.0
```

Create `requirements.txt`:
```bash
numpy
scipy
matplotlib
yfinance
pandas
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@techreport{heuse2025option,
  title={Option Pricing via PDEs with Validation and Calibration},
  author={Heuse, Maxime},
  year={2025},
  institution={Louvain School of Management},
  note={Available at: https://github.com/yourusername/option-pricing-pdes}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional models (local-stochastic volatility, rough Heston)
- Exotic payoffs (barriers, Asian, lookback options)
- GPU acceleration for high-dimensional problems
- Machine learning integration for calibration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Maxime Heuse**
- Email: maximeheuse@gmail.com
- Specialization: Applied Mathematics, Quantitative Finance, Actuarial Engineering

## 🙏 Acknowledgments

- LSM Investment Club for fostering passion in quantitative finance
- Anonymous reviewers and professionals who provided valuable feedback
- Open-source community for numerical libraries

---

**⚠️ Disclaimer**: This code is for educational and research purposes only. It has not been peer-reviewed and should not be used for actual trading or financial decision-making without proper validation and risk management.

(This ReadMe was written by AI)
