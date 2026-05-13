import numpy as np
from typing import Literal, Callable
from dataclasses import dataclass, field


@dataclass


class BlackScholes():
    def __init__(self, S0, r, sigma):
        self.S0 = S0
        self.r = r
        self.sigma = sigma

class Heston():
    def __init__(self, S0, r, v0, kappa, theta, xi, rho = 0):
        self.S0 = S0
        self.r = r
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

class Merton():
    def __init__(self, S0, r, sigma, lam, muJ, sigmaJ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.lam = lam
        self.muJ = muJ
        self.sigmaJ = sigmaJ

class TwoUnderlyings():
    def __init__(self, S0_1, S0_2, r, sigma1, sigma2, rho):
        self.S0_1 = S0_1
        self.S0_2 = S0_2
        self.r = r
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho

class CalibrationParameters():
    '''
    Inputs : 
    model_name : str : 'BlackScholes', 'Heston', 'Merton', 'TwoUnderlyings'
    parameters : dict : dictionary of parameters for the model. The keys depend on the model_name.
     - For 'BlackScholes' : {'r': float, 'sigma': float}
     - For 'Heston' : {'r': float, 'v0': float, 'kappa': float, 'theta': float, 'xi': float, 'rho': float}
     - For 'Merton' : {'r': float, 'sigma': float, 'lam': float, 'muJ': float, 'sigmaJ': float}
     - For 'TwoUnderlyings' : {'r': float, 'sigma1': float, 'sigma2': float, 'rho': float}
    
    Outputs : 
    structured object containing the parameters for the model. The attributes depend on the model_name.
    
    '''
    def __init__(self, model_name, parameters):
        if model_name == 'BlackScholes':
            self.r = parameters['r']
            self.sigma = parameters['sigma']
        elif model_name == 'Heston':
            self.r = parameters['r']
            self.v0 = parameters['v0']
            self.kappa = parameters['kappa']
            self.theta = parameters['theta']
            self.xi = parameters['xi']
            self.rho = parameters['rho']
        elif model_name == 'Merton':
            self.r = parameters['r']
            self.sigma = parameters['sigma']
            self.lam = parameters['lam']
            self.muJ = parameters['muJ']
            self.sigmaJ = parameters['sigmaJ']
        elif model_name == 'TwoUnderlyings':
            self.r = parameters['r']
            self.sigma1 = parameters['sigma1']
            self.sigma2 = parameters['sigma2']
            self.rho = parameters['rho']
        else :
            raise ValueError("Unsupported model name")
@dataclass
class Tree:
    """
    Self-describing container for a recombining binomial or trinomial price tree.

    Attributes
    ----------
    tree_type : 'binomial' | 'trinomial'
    model     : 'CRR' | 'RendlemanBartter' | 'BinomialTian' |
                'TrinomialTian' | 'Joshi'
    stock_tree: np.ndarray
        binomial  -> shape (n+1, n+1)   [i,j] = step i, j down-moves
        trinomial -> shape (n+1, 2n+1)  [i,j] j=0 full-up, j=2i full-down
    r, T, n   : rate, maturity, steps
    dt        : T / n
    p_u, p_d  : risk-neutral up/down probabilities
    p_m       : middle probability (trinomial only)
    """
    tree_type : Literal["binomial", "trinomial"]
    model     : str
    stock_tree: np.ndarray
    r  : float
    T  : float
    n  : int
    dt : float
    p_u: float
    p_d: float
    p_m: float = field(default=None)



class PayoffFunction_1u():
    def __init__(self, payoff_type, ApplicationRule, K, T):
        self.payoff_type = payoff_type
        self.ApplicationRule = ApplicationRule
        self.payoff_func = self.get_payoff_function()
        self.K = K
        self.T = T
    def get_payoff_function(self): # This is a poor design choice. This is designed for MC only. Analytical and PDEs build the payoff function themselves. 
        # pay attention that what you do here needs to be done in "analytical" and "PDEs" as well.
        if self.payoff_type == 'call':
            return lambda paths: np.maximum(paths[:,-1] - self.K, 0)
        elif self.payoff_type == 'put':
            def function(paths):
                return np.maximum(self.K - paths[:,-1], 0)
            return function
        else:
            raise ValueError("Unsupported payoff type")

class PayoffFunction_2u():
    def __init__(self, payoff_type, ApplicationRule, K, T, w1 = 0.5, w2 = 0.5):
        self.payoff_type = payoff_type
        self.ApplicationRule = ApplicationRule
        self.payoff_func = self.get_payoff_function()
        self.K = K # only one stike
        self.T = T
        self.w1 = w1
        self.w2 = w2
    def get_payoff_function(self):

        if self.payoff_type == 'call_spread':
            return lambda paths: np.maximum(paths[:, 0, -1] +  paths[:, 1, -1] - self.K, 0) 
        elif self.payoff_type == 'put_spread':
            return lambda paths: np.maximum(self.K - (paths[:, 0, -1] + paths[:, 1, -1]), 0) 
        elif self.payoff_type == 'exchange':
            return lambda paths: np.maximum(paths[:, 0, -1] - paths[:, 1, -1], 0)
        elif self.payoff_type == 'call_basket':
            return lambda paths: np.maximum(self.w1 * paths[:, 0, -1] + self.w2 * paths[:, 1, -1] - self.K, 0)
        elif self.payoff_type == 'put_basket':
            return lambda paths: np.maximum(self.K - (self.w1 * paths[:, 0, -1] + self.w2 * paths[:, 1, -1]), 0)
        else:
            raise ValueError("Unsupported payoff type")
