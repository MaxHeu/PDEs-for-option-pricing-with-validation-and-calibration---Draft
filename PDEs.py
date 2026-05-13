import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy import signal
import scipy.stats as ss
from Models import Heston, Merton, BlackScholes, TwoUnderlyings, PayoffFunction_1u, PayoffFunction_2u

class TwoUnderlyingsPDESolver:
    """
    Two-underlyings PDE solver using finite differences.
    
    The two-underlyings PDE:
    ∂V/∂t + (1/2)σ₁²S₁²∂²V/∂S₁² + (1/2)σ₂²S₂²∂²V/∂S₂² + ρσ₁σ₂S₁S₂∂²V/∂S₁∂S₂ 
          + rS₁∂V/∂S₁ + rS₂∂V/∂S₂ - rV = 0
    
    Semi-discretization in space yields:
    dU/dt = A·U + b(t)
    
    where U is the vector of option values at grid points.
    """
    

    def __init__(self,S1_0,S2_0, S1_max, S2_max, N_S1, N_S2, r, SigmaMatrix, rho, solver):
        def compute_S_grid(self,m_1, S_max, S0, c):
            """
            Compute the non-uniform grid S_i defined by:
                S_i = K + c * sinh(zeta_i)
            
            Parameters
            ----------
            m_1 : int
                Number of intervals (returns m_1 + 1 points)
            S_max : float
                Maximum value of S
            S_0 : float
                Initial Underlying Value (or shift parameter)
            c : float
                Stretching parameter (must be > 0)

            Returns
            -------
            S : numpy.ndarray
                Array of S_i values of length m_1 + 1
            """

            # Compute zeta endpoints
            zeta_min = np.arcsinh(-S0 / c)
            zeta_max = np.arcsinh((S_max - S0) / c)

            # Step size
            delta_zeta = (zeta_max - zeta_min) / m_1

            # zeta grid
            zeta = zeta_min + delta_zeta * np.arange(m_1 + 1)

            # S grid
            S = S0 + c * np.sinh(zeta)

            return S,zeta



        """
        Initialize the Heston PDE solver.
        
        Parameters:
        -----------
        S1_max : float
            Maximum asset price in grid
        S2_max : float
            Maximum asset price in grid
        N_S1 : int
            Number of grid points in S1 direction
        N_S2 : int
            Number of grid points in S2 direction
        r : float
            Risk-free rate
        SigmaMatrix : 2x2 numpy array
            Volatility matrix (covariance of Brownian motions)
        solver : str
            'Do' for Douglas scheme, 'CS' for Craig-Sneyd scheme
        """
        self.S1_0 = S1_0
        self.S2_0 = S2_0
        self.S1_max = S1_max
        self.S2_max = S2_max
        self.N_S1 = N_S1
        self.N_S2 = N_S2
        self.r = r
        self.sigma11 = SigmaMatrix[0, 0]
        self.sigma12 = SigmaMatrix[0, 1]
        self.sigma21 = SigmaMatrix[1, 0]
        self.sigma22 = SigmaMatrix[1, 1]
        self.rho = rho
        self.solver = solver
        
        # Create spatial grids
        self.S1 = compute_S_grid(self,m_1=(N_S1-1), S_max = S1_max, S0=S1_0, c=S1_0/5)[0]
        self.S2 = compute_S_grid(self,m_1=(N_S2-1), S_max=S2_max, S0=S2_0, c=S2_0/5)[0]
        self.dS1 = np.diff(self.S1)  # non-constant grid spacing in S1 direction (will be computed from self.S1)
        self.dS2 = np.diff(self.S2)  # non-constant grid spacing in S2 direction (will be computed from self.S2)
        
        # Create mesh
        self.S1_mesh, self.S2_mesh = np.meshgrid(self.S1, self.S2, indexing='ij')
        
        # Total number of interior points
        self.N_total = N_S1 * N_S2

    def delta(self, d_grid, i, j):
        '''
        Compute delta_i,j parameter for non-uniform grid for second derivatives.
        
        For grid point i, we need spacings:
        - d_grid[i-1] = spacing between points i-1 and i (used for backward difference)
        - d_grid[i] = spacing between points i and i+1 (used for forward difference)
        
        The parameter j indicates which coefficient:
        j = -1: coefficient for point i-1
        j = 0: coefficient for point i  
        j = 1: coefficient for point i+1
        '''
        if j == -1:
            # For point i-1: uses d_grid[i-1] and d_grid[i]
            delta = 2 / (d_grid[i-1] * (d_grid[i-1] + d_grid[i]))
        elif j == 0:
            # For point i (diagonal): uses d_grid[i-1] and d_grid[i]
            delta = -2 / (d_grid[i-1] * d_grid[i])
        elif j == 1:
            # For point i+1: uses d_grid[i-1] and d_grid[i]
            delta = 2 / (d_grid[i] * (d_grid[i-1] + d_grid[i]))
        return delta
    
    def beta(self, d_grid, i, j):
        '''
        Compute beta_i,j parameter for non-uniform grid for first derivatives.
        Similar indexing as delta.
        '''
        if j == -1:
            # For point i-1
            beta = -d_grid[i] / (d_grid[i-1] * (d_grid[i-1] + d_grid[i]))
        elif j == 0:
            # For point i (diagonal)
            beta = (d_grid[i] - d_grid[i-1]) / (d_grid[i-1] * d_grid[i])
        elif j == 1:
            # For point i+1
            beta = d_grid[i-1] / (d_grid[i] * (d_grid[i-1] + d_grid[i]))
        return beta
    
    def gamma(self, d_grid, i, j):
        """
        Compute gamma_i,j parameter for non-uniform grid
        d_grid is the grid spacing (dS or dnu) at point (i,j)
        """
        if j == 0 : 
            gamma = (-2*d_grid[i+1] - d_grid[i+2])/(d_grid[i+1] * (d_grid[i+1] + d_grid[i+2]))
        elif j == 1 : 
            gamma = (d_grid[i+1] + d_grid[i+2])/(d_grid[i+1] * d_grid[i+2])
        elif j == 2 : 
            gamma = -d_grid[i+1]/(d_grid[i+2] * (d_grid[i+1] + d_grid[i+2]))
        return gamma
        
    
    def build_matrix_A(self):
        '''
        Build the matrix A = A0 + A1 + A2 for the semi-discrete system dU/dt = A·U + b.
        '''
        r = self.r
        N_S1, N_S2 = self.N_S1, self.N_S2
        # Be careful : input is covariance matrix, but we need cholesky decomposition terms !!!
        sigma_11 = np.sqrt(self.sigma11)
        sigma_12 = 0
        sigma_21 = self.rho * np.sqrt(self.sigma22)
        sigma_22 = np.sqrt(self.sigma22) * np.sqrt(1 - self.rho**2)

        
        n = N_S1 * N_S2
        
        # Initialize sparse matrices
        A0 = sp.lil_matrix((n, n))  # Mixed derivative term
        A1 = sp.lil_matrix((n, n))  # S1-direction derivatives
        A2 = sp.lil_matrix((n, n))  # S2-direction derivatives
        
        def idx(i, j):
            return i * N_S2 + j
        
        # Loop over ALL grid points (including boundaries)
        for i in range(N_S1):
            for j in range(N_S2):
                k = idx(i, j)
                
                # BOUNDARY POINTS: Set identity (will be handled by b vector)
                if i == 0 or i == N_S1 - 1 or j == 0 or j == N_S2 - 1:
                    # Don't set anything in A matrices for boundary points
                    # They'll be handled through the b vector
                    continue
                
                # INTERIOR POINTS: Apply finite difference stencil
                # Get local values
                S1_i = self.S1[i]
                S2_j = self.S2[j]
                
                # PDE coefficients
                coef_S1 = r*S1_i            # ∂V/∂S1
                coef_S1_sd = ((sigma_11**2 + sigma_12**2) * S1_i**2) / 2  # (1/2)νS²∂²V/∂S²    
                coef_S2 = r*S2_j            # ∂V/∂S2
                coef_S2_sd = ((sigma_21**2 + sigma_22**2) * S2_j**2) / 2  # (1/2)σ²ν∂²V/∂ν²
                coef_S1S2 = (sigma_11 * sigma_21 + sigma_12 * sigma_22)* S1_i * S2_j # ∂²V/∂S1∂S2
                coef_0 = -self.r                           # -rV
                
                # ============ SECOND DERIVATIVES ============
                
                # Second derivative in S1: goes to A1
                if coef_S1_sd != 0:
                    A1[k, idx(i-1, j)] += coef_S1_sd * self.delta(self.dS1, i, -1)
                    A1[k, idx(i, j)]   += coef_S1_sd * self.delta(self.dS1, i, 0)
                    A1[k, idx(i+1, j)] += coef_S1_sd * self.delta(self.dS1, i, 1)
                
                # Second derivative in S2: goes to A2
                if coef_S2_sd != 0:
                    A2[k, idx(i, j-1)] += coef_S2_sd * self.delta(self.dS2, j, -1)
                    A2[k, idx(i, j)]   += coef_S2_sd * self.delta(self.dS2, j, 0)
                    A2[k, idx(i, j+1)] += coef_S2_sd * self.delta(self.dS2, j, 1)
                
                # ============ MIXED DERIVATIVE ============
                
                # Mixed derivative: ∂²V/∂S1∂S2 goes to A0
                # Uses product of beta coefficients
                if coef_S1S2 != 0:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            beta_S1 = self.beta(self.dS1, i, di)
                            beta_S2 = self.beta(self.dS2, j, dj)
                            A0[k, idx(i+di, j+dj)] += coef_S1S2 * beta_S1 * beta_S2
                
                # ============ FIRST DERIVATIVES ============
                
                # First derivative in S1: goes to A1
                if coef_S1 != 0:
                    A1[k, idx(i-1, j)] += coef_S1 * self.beta(self.dS1, i, -1)
                    A1[k, idx(i, j)]   += coef_S1 * self.beta(self.dS1, i, 0)
                    A1[k, idx(i+1, j)] += coef_S1 * self.beta(self.dS1, i, 1)
                
                # First derivative in S2: goes to A2
                if coef_S2 != 0:
                    A2[k, idx(i, j-1)] += coef_S2 * self.beta(self.dS2, j, -1)
                    A2[k, idx(i, j)]   += coef_S2 * self.beta(self.dS2, j, 0)
                    A2[k, idx(i, j+1)] += coef_S2 * self.beta(self.dS2, j, 1)
                
                # ============ ZERO-ORDER TERM ============
                
                # Discount rate: split between A1 and A2
                A1[k, k] += coef_0 / 2
                A2[k, k] += coef_0 / 2
        
        # Store matrices separately for ADI schemes
        self.A0 = A0.tocsr()
        self.A1 = A1.tocsr()
        self.A2 = A2.tocsr()
        
        # Return full matrix A = A0 + A1 + A2
        return A0.tocsr(), A1.tocsr(), A2.tocsr()

    def build_vector_b(self, t, T, K, option_type='call_spread', w1=0.5, w2=0.5):
        N_S1, N_S2 = self.N_S1, self.N_S2
        n   = N_S1 * N_S2
        b0  = np.zeros(n);  b1 = np.zeros(n);  b2 = np.zeros(n)
        tau = T - t
        df  = np.exp(-self.r * tau)

        def idx(i, j): return i * N_S2 + j

        def bc_value(S1_i, S2_j, otype):
            if otype == 'exchange':
                if   S1_i == self.S1[0]:  return 0.0
                elif S1_i == self.S1[-1]: return max(S1_i - S2_j, 0.0)
                elif S2_j == self.S2[0]:  return S1_i
                elif S2_j == self.S2[-1]: return max(S1_i - S2_j, 0.0)
            elif otype == 'call_spread':
                if   S1_i == self.S1[0]:  return max(-K*df - S2_j, 0.0)
                elif S2_j == self.S2[0]:  return max(S1_i - K*df, 0.0)
                elif S1_i == self.S1[-1]: return max(S1_i - S2_j - K*df, 0.0)
                elif S2_j == self.S2[-1]: return max(S1_i - S2_j - K*df, 0.0)
            elif otype == 'put_spread':
                if   S1_i == self.S1[0]:  return max(K*df + S2_j, 0.0)
                elif S2_j == self.S2[0]:  return max(K*df - S1_i, 0.0)
                elif S1_i == self.S1[-1]: return max(K*df - (S1_i - S2_j), 0.0)
                elif S2_j == self.S2[-1]: return max(K*df - (S1_i - S2_j), 0.0)
            elif otype == 'call_basket':
                if   S1_i == self.S1[0]:  return max(w2*S2_j - K*df, 0.0)
                elif S2_j == self.S2[0]:  return max(w1*S1_i - K*df, 0.0)
                elif S1_i == self.S1[-1]: return max(w1*S1_i + w2*S2_j - K*df, 0.0)
                elif S2_j == self.S2[-1]: return max(w1*S1_i + w2*S2_j - K*df, 0.0)
            elif otype == 'put_basket':
                if   S1_i == self.S1[0]:  return max(K*df - w2*S2_j, 0.0)
                elif S2_j == self.S2[0]:  return max(K*df - w1*S1_i, 0.0)
                elif S1_i == self.S1[-1]: return max(K*df - (w1*S1_i + w2*S2_j), 0.0)
                elif S2_j == self.S2[-1]: return max(K*df - (w1*S1_i + w2*S2_j), 0.0)
            return 0.0

        for i in range(N_S1):
            for j in range(N_S2):
                if 0 < i < N_S1-1 and 0 < j < N_S2-1:
                    continue
                k    = idx(i, j)
                val  = bc_value(self.S1[i], self.S2[j], option_type)
                b1[k] = val / 2.0
                b2[k] = val / 2.0

        return b0, b1, b2
                   
                                 
    def solve(self, K, T, payoff_func, theta=0.5,
              option_type='call_spread', w1=0.5, w2=0.5,
              solver='Do', exercise_style='european',
              plot_sparsity=False):
        """
        Solve the two-underlyings PDE with Douglas or Craig-Sneyd ADI.

        Parameters
        ----------
        K              : float    – strike
        T              : float    – maturity
        payoff_func    : callable – payoff at maturity; signature depends on option_type:
                           exchange     -> payoff_func(S1, S2)
                           call/put_spread -> payoff_func(S1, S2, K)
                           call/put_basket -> payoff_func(w1, w2, S1, S2, K)
        dt             : float    – time step
        theta          : float    – ADI parameter (0.5 recommended)
        option_type    : str      – 'exchange','call_spread','put_spread',
                                    'call_basket','put_basket'
        w1, w2         : float    – basket weights
        solver         : str      – 'Do' or 'CS'
        exercise_style : str      – 'european' or 'american'
                          For 'american', the IT-type projection is applied
                          after every ADI time step (report §3.5):
                              U^{n+1}_{ij} <- max(U^{n+1}_cont,ij , phi(S1_i, S2_j))
        plot_sparsity  : bool

        Returns
        -------
        U_grid : ndarray (N_S1, N_S2)
        A      : sparse matrix (A0+A1+A2)
        """
        dt = T / 200  # time step for ADI (can be adjusted)
        def F_j(A_j, b_j, w): return A_j.dot(w) + b_j

        A0, A1, A2 = self.build_matrix_A()

        if plot_sparsity:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, A, title in zip(axes, [A0, A1, A2],
                                    ['A0 (mixed)', 'A1 (S1)', 'A2 (S2)']):
                ax.spy(A[45:105, 45:105], markersize=1)
                ax.set_title(f'Sparsity of {title}')
            plt.tight_layout()
            #plt.savefig('Sparsities_TwoAsset.pdf', dpi=300)
            plt.show()

        # --- terminal payoff ---
        if option_type == 'exchange':
            payoff = payoff_func(self.S1_mesh, self.S2_mesh)
        elif option_type in ('call_spread', 'put_spread'):
            payoff = payoff_func(self.S1_mesh, self.S2_mesh, K)
        elif option_type in ('call_basket', 'put_basket'):
            payoff = payoff_func(w1, w2, self.S1_mesh, self.S2_mesh, K)
        else:
            raise ValueError(f"Unknown option_type: {option_type}")

        U           = payoff.flatten()
        is_american = (exercise_style.lower() == 'american')
        payoff_flat = payoff.flatten()   # phi(S1_i, S2_j) for projection

        N_t    = int(T / dt)
        dt_act = T / N_t
        I      = sp.identity(self.N_total, format='csr')
        _solver = solver if isinstance(solver, str) else self.solver

        for n in range(N_t):
            tau_n = n * dt_act
            b0, b1, b2 = self.build_vector_b(t=tau_n, T=T, K=K,
                                              option_type=option_type,
                                              w1=w1, w2=w2)
            if _solver == 'Do':
                F      = F_j(A0, b0, U) + F_j(A1, b1, U) + F_j(A2, b2, U)
                Y0     = U + dt_act * F
                Y1     = spsolve(I - theta * dt_act * A1, Y0 - theta * dt_act * A1.dot(U))
                Y2     = spsolve(I - theta * dt_act * A2, Y1 - theta * dt_act * A2.dot(U))
                U_cont = Y2

            elif _solver == 'CS':
                Y0     = U + dt_act * (A0.dot(U)+b0 + A1.dot(U)+b1 + A2.dot(U)+b2)
                Y1     = spsolve(I - theta * dt_act * A1, Y0 - theta * dt_act * A1.dot(U))
                Y2     = spsolve(I - theta * dt_act * A2, Y1 - theta * dt_act * A2.dot(U))
                Y3     = Y0 + 0.5 * dt_act * (A0.dot(Y2) - A0.dot(U))
                Y4     = spsolve(I - theta * dt_act * A1, Y3 - theta * dt_act * A1.dot(U))
                Y5     = spsolve(I - theta * dt_act * A2, Y4 - theta * dt_act * A2.dot(U))
                U_cont = Y5

            else:
                raise ValueError("solver must be 'Do' or 'CS'")

            # American early-exercise projection (IT-type, report §3.5):
            # U^{n+1}_{ij} <- max(U^{n+1}_cont,ij , phi(S1_i, S2_j))
            if is_american:
                U = np.maximum(U_cont, payoff_flat)
            else:
                U = U_cont

        U_grid = U.reshape((self.N_S1, self.N_S2))
        return U_grid
   
    def get_option_price(self, U_grid, S10, S20):
        """
        Interpolate option price at initial spot and variance.
        
        Parameters:
        -----------
        U_grid : numpy array (N_S1, N_S2)
            Option values on grid
        S10 : float
            Initial first asset price
        S20 : float
            Initial second asset price
            
        Returns:
        --------
        price : float
            Interpolated option price
        """
        interp = RectBivariateSpline(self.S1, self.S2, U_grid)
        return float(interp(S10, S20))

class MertonJumpDiffusionPDESolver:
    """
    Merton Jump-Diffusion PIDE solver using finite differences
    with FFT-based convolution for the jump integral term.

    The Merton PIDE (in log-price x = log(S), forward tau = T - t):
    ∂V/∂tau = (1/2)σ²∂²V/∂x² + (r - m - 1/2 σ²)∂V/∂x - (r + λ)V
              + λ ∫ V(x + y) f_Y(y) dy

    where m = λ(E[e^Y] - 1), Y ~ N(μ_J, σ_J²) (Merton log-normal jumps).

    Time-marching is forward in tau (tau = 0 is maturity, tau = T is present).
    The diffusion part is treated implicitly; the jump integral is treated explicitly
    via fftconvolve.
    """

    def __init__(self, S0, S_max, S_min, Nspace, Ntime, r, sigma, lam, muJ, sigJ):
        """
        Initialize the Merton Jump-Diffusion PIDE solver.

        Parameters:
        -----------
        S0 : float
            Initial asset price
        S_max : float
            Maximum asset price for spatial grid boundary
        S_min : float
            Minimum asset price for spatial grid boundary
        Nspace : int
            Number of spatial grid points (in log-price)
        Ntime : int
            Number of time steps
        r : float
            Risk-free rate
        sigma : float
            Diffusion volatility of the underlying
        lam : float
            Jump intensity (average number of jumps per year)
        muJ : float
            Mean of the log-jump size Y ~ N(muJ, sigJ²)
        sigJ : float
            Standard deviation of the log-jump size
        """
        self.S0     = S0
        self.X0     = np.log(S0)
        self.S_max  = S_max
        self.S_min  = S_min
        self.Nspace = Nspace
        self.Ntime  = Ntime
        self.r      = r
        self.sigma  = sigma
        self.lam    = lam
        self.muJ    = muJ
        self.sigJ   = sigJ

        # Log-price grid boundaries
        self.x_max = np.log(S_max)
        self.x_min = np.log(S_min)

        # Grid spacing in log-price
        self.dx = (self.x_max - self.x_min) / (Nspace - 1)

        # Extra padding points to capture jump tails (3 std dev of jump component)
        dev_X = np.sqrt(lam * sigJ**2 + lam * muJ**2)
        self.extraP = int(np.floor(3 * dev_X / self.dx))

        # Extended log-price grid (with padding)
        self.x = np.linspace(
            self.x_min - self.extraP * self.dx,
            self.x_max + self.extraP * self.dx,
            Nspace + 2 * self.extraP
        )

    # ------------------------------------------------------------------
    # Jump measure discretization
    # ------------------------------------------------------------------

    def build_jump_measure(self):
        """
        Discretize the Merton log-normal jump measure ν(dy) = λ · f_Y(y) dy,
        where Y ~ N(muJ, sigJ²).

        Uses CDF differences on a grid of width dx centered at each node,
        extended over ±(extraP+1) nodes around 0.

        Returns:
        --------
        nu : numpy.ndarray
            Discretized jump measure vector of length 2*(extraP+1)
        m : float
            Compensator m = λ(E[e^Y] - 1)
        """
        dx      = self.dx
        extraP  = self.extraP
        muJ     = self.muJ
        sigJ    = self.sigJ
        lam     = self.lam

        cdf = ss.norm.cdf(
            np.linspace(
                -(extraP + 1 + 0.5) * dx,
                 (extraP + 1 + 0.5) * dx,
                 2 * (extraP + 2)
            ),
            loc=muJ,
            scale=sigJ
        )
        nu = lam * (cdf[1:] - cdf[:-1])
        m  = lam * (np.exp(muJ + 0.5 * sigJ**2) - 1.0)
        return nu, m

    # ------------------------------------------------------------------
    # Build implicit tridiagonal operator
    # ------------------------------------------------------------------

    def build_tridiagonal(self, dt, m):
        """
        Assemble the implicit tridiagonal sparse matrix D for the diffusion
        and discount/jump-decay part of the PIDE.

        The interior-node stencil (forward Euler in tau, implicit diffusion):
            a · V_{i-1} + b · V_i + c · V_{i+1}

        where:
            a =  (dt/2) · [(r - m - σ²/2)/dx  - σ²/dx²]
            b =  1 + dt · [σ²/dx² + r + λ]
            c = -(dt/2) · [(r - m - σ²/2)/dx  + σ²/dx²]

        Parameters:
        -----------
        dt : float
            Time step size
        m : float
            Jump compensator m = λ(E[e^Y] - 1)

        Returns:
        --------
        DD : SuperLU factorization
            Factored tridiagonal system ready for DD.solve(rhs)
        a, c : float
            Sub- and super-diagonal coefficients (needed for boundary offsets)
        """
        sig2 = self.sigma ** 2
        dxx  = self.dx ** 2
        r    = self.r
        lam  = self.lam

        a =  (dt / 2.0) * ((r - m - 0.5 * sig2) / self.dx - sig2 / dxx)
        b =  1.0 + dt   * (sig2 / dxx + r + lam)
        c = -(dt / 2.0) * ((r - m - 0.5 * sig2) / self.dx + sig2 / dxx)

        Nint = self.Nspace - 2
        D  = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nint, Nint)).tocsc()
        DD = splu(D)
        return DD, a, c

    # ------------------------------------------------------------------
    # Build boundary conditions
    # ------------------------------------------------------------------

    def build_boundary_conditions(self, K, T, option_type='call'):
        """
        Initialize the solution array V and set boundary conditions
        for all time levels (tau = 0 .. T).

        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Maturity (in years)
        option_type : str
            'call' or 'put'

        Returns:
        --------
        V : numpy.ndarray, shape (Nspace + 2*extraP, Ntime)
            Solution array with terminal payoff and boundaries pre-filled
        Tau : numpy.ndarray
            Time grid from 0 to T (forward tau)
        dt : float
            Time step size
        """
        x      = self.x
        extraP = self.extraP
        Nspace = self.Nspace
        Ntime  = self.Ntime
        r      = self.r

        Tau, dt = np.linspace(0.0, T, Ntime, retstep=True)

        V      = np.zeros((Nspace + 2 * extraP, Ntime))
        offset = np.zeros(Nspace - 2)

        if option_type == 'call':
            Payoff = np.maximum(np.exp(x) - K, 0.0)
            V[:, 0] = Payoff
            # S -> +inf: call ~ S - K·e^{-r(T-tau)}
            V[-extraP - 1:, :] = (
                np.exp(x[-extraP - 1:]).reshape(extraP + 1, 1)
                - K * np.exp(-r * (T - Tau)).reshape(1, Ntime)
            )
            # S -> 0: call -> 0
            V[:extraP + 1, :] = 0.0

        elif option_type == 'put':
            Payoff = np.maximum(K - np.exp(x), 0.0)
            V[:, 0] = Payoff
            # S -> 0: put ~ K·e^{-r(T-tau)}
            V[:extraP + 1, :] = K * np.exp(-r * (T - Tau)).reshape(1, Ntime)
            # S -> +inf: put -> 0
            V[-extraP - 1:, :] = 0.0
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

        return V, Tau, dt

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def plotPDE(self, Grid):
        """
        Visualize the Merton Jump-Diffusion PIDE solution.

        Parameters
        ----------
        Grid : ndarray, shape (Nspace + 2*extraP, Ntime)
            Full space-time solution returned by solve().
            Grid[:, 0]  = terminal payoff (tau = 0)
            Grid[:, -1] = present value   (tau = T)

        Extended domain
        ---------------
        Grid contains extraP padding nodes on each side (jump-tail buffer).
        Only the physical slice Grid[extraP : extraP+Nspace, :] is plotted,
        converted to price space via S = exp(x).
        """
        from mpl_toolkits.mplot3d import Axes3D     # noqa: F401
        import matplotlib.gridspec as gridspec

        # ── Strip the padded domain ───────────────────────────────────────────
        ep  = self.extraP
        Ns  = self.Nspace
        x_p = self.x[ep : ep + Ns]                  # physical log-price grid
        S_p = np.exp(x_p)                            # convert to price
        V_p = Grid[ep : ep + Ns, :]                  # shape (Nspace, Ntime)
        Tau = np.linspace(0.0, self.T, self.Ntime)

        # ── Downsample for 3-D surface (performance) ──────────────────────────
        step_S = max(1, Ns         // 60)
        step_t = max(1, self.Ntime // 60)
        S_d    = S_p[::step_S]
        tau_d  = Tau[::step_t]
        V_d    = V_p[::step_S, ::step_t]
        T_m, S_m = np.meshgrid(tau_d, S_d)

        # ── Figure ────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            f"Merton Jump-Diffusion PIDE Solution  —  "
            f"S₀={self.S0}, K={self.K}, T={self.T}y, "
            f"σ={self.sigma}, λ={self.lam}, μⱼ={self.muJ}, σⱼ={self.sigJ}, r={self.r}",
            fontsize=13, fontweight='bold', y=0.98,
        )
        gs  = gridspec.GridSpec(1, 1, figure=fig,
                                top=0.93, bottom=0.07, left=0.07, right=0.97)
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        surf = ax1.plot_surface(S_m, T_m, V_d, cmap='viridis', alpha=0.88, linewidth=0)
        fig.colorbar(surf, ax=ax1, shrink=0.45, pad=0.04, label='V(S, τ)')
        ax1.set_xlabel('Stock Price S',       labelpad=8)
        ax1.set_ylabel('Time to Maturity τ',  labelpad=8)
        ax1.set_zlabel('Option Price V',      labelpad=8)
        ax1.set_title(
            f'Option Price Surface V(S, τ)  [padding: ±{self.extraP} nodes stripped]',
            pad=10)
        ax1.view_init(elev=25, azim=-55)

        plt.tight_layout()
        plt.show()

    def solve(self, K, T, payoff_func=None, option_type='call',
              exercise_style='european', plotPDE_Bool = False):
        """
        Solve the Merton PIDE by forward iteration in tau.

        Parameters
        ----------
        K              : float    – strike
        T              : float    – maturity
        payoff_func    : callable or None – optional custom payoff(x, K)
        option_type    : str      – 'call' or 'put'
        exercise_style : str      – 'european' or 'american'
                          For 'american', after the full IMEX step
                          (implicit diffusion + explicit FFT jump integral)
                          the IT-type projection is applied at interior nodes:
                              u^{n+1}_k  <-  max(u^{n+1}_cont,k , phi(S_k))
                          (report §3.5, Boen & in 't Hout operator-splitting PIDCP).

        Returns
        -------
        V      : ndarray (Nspace + 2*extraP, Ntime)
        x_grid : ndarray – extended log-price grid
        """
        extraP  = self.extraP
        Nspace  = self.Nspace
        x       = self.x
        self.T = T
        self.K = K
        nu, m               = self.build_jump_measure()
        V, Tau, dt          = self.build_boundary_conditions(K, T, option_type)

        if payoff_func is not None:
            V[:, 0] = payoff_func(x, K)

        DD, a_coef, c_coef  = self.build_tridiagonal(dt, m)
        is_american         = (exercise_style.lower() == 'american')

        # Payoff on the full extended grid for American projection
        # phi depends only on S = exp(x), not on tau
        if is_american:
            if option_type == 'call':
                payoff_grid = np.maximum(np.exp(x) - K, 0.0)
            else:
                payoff_grid = np.maximum(K - np.exp(x), 0.0)
            payoff_interior = payoff_grid[extraP + 1 : -extraP - 1]

        offset = np.zeros(Nspace - 2)

        for i in range(self.Ntime - 1):
            # Boundary offsets
            offset[0]  = a_coef * V[extraP, i]
            offset[-1] = c_coef * V[-1 - extraP, i]

            # Explicit FFT jump integral
            V_jump = (V[extraP + 1 : -extraP - 1, i]
                      + dt * signal.fftconvolve(V[:, i], nu, mode="valid"))

            # Implicit diffusion step -> continuation value
            V_cont = DD.solve(V_jump - offset)

            # American projection: enforce early-exercise constraint
            # u^{n+1}_k <- max(u^{n+1}_cont,k , phi(S_k))  (report §3.5)
            if is_american:
                V_cont = np.maximum(V_cont, payoff_interior)

            V[extraP + 1 : -extraP - 1, i + 1] = V_cont

        if plotPDE_Bool:
            self.plotPDE(V)
        return V

    # ------------------------------------------------------------------
    # get_option_price
    # ------------------------------------------------------------------

    def get_option_price(self, V, S0=None):
        """
        Interpolate the option price at initial spot S0 from the solution grid.

        The price corresponds to the last column of V (tau = T, i.e. t = 0).

        Parameters:
        -----------
        V : numpy.ndarray, shape (Nspace + 2*extraP, Ntime)
            Full solution grid returned by solve()
        S0 : float or None
            Initial asset price. If None, uses self.S0.

        Returns:
        --------
        price : float
            Interpolated option price at t=0 and S=S0
        """
        if S0 is None:
            S0 = self.S0
        X0 = np.log(S0)
        return float(np.interp(X0, self.x, V[:, -1]))

class HestonPDESolver:
    """
    Heston PDE solver using finite differences.
    
    The Heston PDE:
    ∂V/∂t + (1/2)νS²∂²V/∂S² + ρσνS∂²V/∂S∂ν + (1/2)σ²ν∂²V/∂ν² 
          + rS∂V/∂S + κ(θ-ν)∂V/∂ν - rV = 0
    
    Semi-discretization in space yields:
    dU/dt = A·U + b(t)
    
    where U is the vector of option values at grid points.
    """
    

    def __init__(self, S_max, nu_max, N_S, N_nu, r, kappa, theta, sigma, rho, solver):
        def compute_S_grid(self,m_1, S_max, K, c):
            """
            Compute the non-uniform grid S_i defined by:
                S_i = K + c * sinh(zeta_i)
            
            Parameters
            ----------
            m_1 : int
                Number of intervals (returns m_1 + 1 points)
            S_max : float
                Maximum value of S
            K : float
                Strike (or shift parameter)
            c : float
                Stretching parameter (must be > 0)

            Returns
            -------
            S : numpy.ndarray
                Array of S_i values of length m_1 + 1
            """

            # Compute zeta endpoints
            zeta_min = np.arcsinh(-K / c)
            zeta_max = np.arcsinh((S_max - K) / c)

            # Step size
            delta_zeta = (zeta_max - zeta_min) / m_1

            # zeta grid
            zeta = zeta_min + delta_zeta * np.arange(m_1 + 1)

            # S grid
            S = K + c * np.sinh(zeta)

            return S,zeta

        def compute_nu_grid(self, m_2, V, K, d):
            """
            Compute the non-uniform grid nu_i defined by:
                nu_i = d*arcsinh(eta_j)
            
            Parameters
            ----------
            m_2 : int
                Number of intervals (returns m_2 + 1 points)
            V : float
                Maximum value of nu
            K : float
                Strike (or shift parameter)
            d : float
                Stretching parameter (must be > 0)

            Returns
            -------
            nu : numpy.ndarray
                Array of nu_i values of length m_2 + 1
            """


            # Step size
            delta_eta = np.arcsinh(V/d) / m_2

            # Compute zeta endpoints
            eta_min = 0
            eta_max = V

            # zeta grid
            eta = eta_min + delta_eta * np.arange(m_2 + 1)

            # nu grid
            nu = d * np.sinh(eta)

            return nu,eta


        """
        Initialize the Heston PDE solver.
        
        Parameters:
        -----------
        S_max : float
            Maximum asset price in grid
        nu_max : float
            Maximum variance in grid
        N_S : int
            Number of grid points in S direction
        N_nu : int
            Number of grid points in nu direction
        r : float
            Risk-free rate
        kappa : float
            Mean reversion speed
        theta : float
            Long-term variance
        sigma : float
            Volatility of variance
        rho : float
            Correlation between Brownian motions
        """
        self.S_max = S_max
        self.nu_max = nu_max
        self.N_S = N_S
        self.N_nu = N_nu
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.solver = solver
        
        # Create spatial grids
        self.S = compute_S_grid(self,m_1=(N_S-1), S_max=S_max, K=100, c=20)[0]
        self.nu = compute_nu_grid(self,m_2=(N_nu-1), V=5, K=100, d=1/100)[0]
        self.dS = np.diff(self.S)  # non-constant grid spacing in S direction (will be computed from self.S)
        self.dnu = np.diff(self.nu)  # non-constant grid spacing in nu direction (will be computed from self.nu)
        
        # Create mesh
        self.S_mesh, self.nu_mesh = np.meshgrid(self.S, self.nu, indexing='ij')
        
        # Total number of interior points
        self.N_total = N_S * N_nu

    def delta(self, d_grid, i, j):
        '''
        Compute delta_i,j parameter for non-uniform grid for second derivatives.
        
        For grid point i, we need spacings:
        - d_grid[i-1] = spacing between points i-1 and i (used for backward difference)
        - d_grid[i] = spacing between points i and i+1 (used for forward difference)
        
        The parameter j indicates which coefficient:
        j = -1: coefficient for point i-1
        j = 0: coefficient for point i  
        j = 1: coefficient for point i+1
        '''
        if j == -1:
            # For point i-1: uses d_grid[i-1] and d_grid[i]
            delta = 2 / (d_grid[i-1] * (d_grid[i-1] + d_grid[i]))
        elif j == 0:
            # For point i (diagonal): uses d_grid[i-1] and d_grid[i]
            delta = -2 / (d_grid[i-1] * d_grid[i])
        elif j == 1:
            # For point i+1: uses d_grid[i-1] and d_grid[i]
            delta = 2 / (d_grid[i] * (d_grid[i-1] + d_grid[i]))
        return delta
    
    def beta(self, d_grid, i, j):
        '''
        Compute beta_i,j parameter for non-uniform grid for first derivatives.
        Similar indexing as delta.
        '''
        if j == -1:
            # For point i-1
            beta = -d_grid[i] / (d_grid[i-1] * (d_grid[i-1] + d_grid[i]))
        elif j == 0:
            # For point i (diagonal)
            beta = (d_grid[i] - d_grid[i-1]) / (d_grid[i-1] * d_grid[i])
        elif j == 1:
            # For point i+1
            beta = d_grid[i-1] / (d_grid[i] * (d_grid[i-1] + d_grid[i]))
        return beta
    
    def gamma(self, d_grid, i, j):
        """
        Compute gamma_i,j parameter for non-uniform grid
        d_grid is the grid spacing (dS or dnu) at point (i,j)
        """
        if j == 0 : 
            gamma = (-2*d_grid[i+1] - d_grid[i+2])/(d_grid[i+1] * (d_grid[i+1] + d_grid[i+2]))
        elif j == 1 : 
            gamma = (d_grid[i+1] + d_grid[i+2])/(d_grid[i+1] * d_grid[i+2])
        elif j == 2 : 
            gamma = -d_grid[i+1]/(d_grid[i+2] * (d_grid[i+1] + d_grid[i+2]))
        return gamma
        
    
    def build_matrix_A(self):
        '''
        Build the matrix A = A0 + A1 + A2 for the semi-discrete system dU/dt = A·U + b.
        '''
        N_S, N_nu = self.N_S, self.N_nu
        dS, dnu = self.dS, self.dnu
        
        n = N_S * N_nu
        
        # Initialize sparse matrices
        A0 = sp.lil_matrix((n, n))  # Mixed derivative term
        A1 = sp.lil_matrix((n, n))  # S-direction derivatives
        A2 = sp.lil_matrix((n, n))  # nu-direction derivatives
        
        def idx(i, j):
            return i * N_nu + j
        
        # Loop over ALL grid points (including boundaries)
        for i in range(N_S):
            for j in range(N_nu):
                k = idx(i, j)
                
                # BOUNDARY POINTS: Set identity (will be handled by b vector)
                if i == 0 or i == N_S - 1 or j == 0 or j == N_nu - 1:
                    # Don't set anything in A matrices for boundary points
                    # They'll be handled through the b vector
                    continue
                
                # INTERIOR POINTS: Apply finite difference stencil
                # Get local values
                S_i = self.S[i]
                nu_j = self.nu[j]
                
                # PDE coefficients
                coef_SS = 0.5 * nu_j * S_i**2              # ∂²V/∂S²
                coef_nunu = 0.5 * self.sigma**2 * nu_j     # ∂²V/∂ν²
                coef_Snu = self.rho * self.sigma * nu_j * S_i  # ∂²V/∂S∂ν
                coef_S = self.r * S_i                      # ∂V/∂S
                coef_nu = self.kappa * (self.theta - nu_j) # ∂V/∂ν
                coef_0 = -self.r                           # -rV
                
                # ============ SECOND DERIVATIVES ============
                
                # Second derivative in S: goes to A1
                if coef_SS != 0:
                    A1[k, idx(i-1, j)] += coef_SS * self.delta(dS, i, -1)
                    A1[k, idx(i, j)]   += coef_SS * self.delta(dS, i, 0)
                    A1[k, idx(i+1, j)] += coef_SS * self.delta(dS, i, 1)
                
                # Second derivative in nu: goes to A2
                if coef_nunu != 0:
                    A2[k, idx(i, j-1)] += coef_nunu * self.delta(dnu, j, -1)
                    A2[k, idx(i, j)]   += coef_nunu * self.delta(dnu, j, 0)
                    A2[k, idx(i, j+1)] += coef_nunu * self.delta(dnu, j, 1)
                
                # ============ MIXED DERIVATIVE ============
                
                # Mixed derivative: ∂²V/∂S∂ν goes to A0
                # Uses product of beta coefficients
                if coef_Snu != 0:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            beta_S = self.beta(dS, i, di)
                            beta_nu = self.beta(dnu, j, dj)
                            A0[k, idx(i+di, j+dj)] += coef_Snu * beta_S * beta_nu
                
                # ============ FIRST DERIVATIVES ============
                
                # First derivative in S: goes to A1
                if coef_S != 0:
                    A1[k, idx(i-1, j)] += coef_S * self.beta(dS, i, -1)
                    A1[k, idx(i, j)]   += coef_S * self.beta(dS, i, 0)
                    A1[k, idx(i+1, j)] += coef_S * self.beta(dS, i, 1)
                
                # First derivative in nu: goes to A2
                if coef_nu != 0:
                    A2[k, idx(i, j-1)] += coef_nu * self.beta(dnu, j, -1)
                    A2[k, idx(i, j)]   += coef_nu * self.beta(dnu, j, 0)
                    A2[k, idx(i, j+1)] += coef_nu * self.beta(dnu, j, 1)
                
                # ============ ZERO-ORDER TERM ============
                
                # Discount rate: split between A1 and A2
                A1[k, k] += coef_0 / 2
                A2[k, k] += coef_0 / 2
        
        # Store matrices separately for ADI schemes
        self.A0 = A0.tocsr()
        self.A1 = A1.tocsr()
        self.A2 = A2.tocsr()
        
        # Return full matrix A = A0 + A1 + A2
        return A0.tocsr(), A1.tocsr(), A2.tocsr()

    def build_vector_b(self, t, T, K, option_type='call'):
        '''
        Build the boundary condition vectors b0(t), b1(t), b2(t).
        
        According to the ADI paper (In 't Hout & Foulon 2010):
        - b0: boundary contributions from mixed derivative term
        - b1: boundary contributions from S-direction terms
        - b2: boundary contributions from nu-direction terms
        
        Parameters:
        -----------
        t : float
            Current time
        T : float
            Maturity time
        K : float
            Strike price
        option_type : str
            'call' or 'put'
        '''
        N_S, N_nu = self.N_S, self.N_nu
        n = N_S * N_nu
        
        b0 = np.zeros(n)
        b1 = np.zeros(n)
        b2 = np.zeros(n)
        
        tau = T - t  # Time to maturity
        df = np.exp(-self.r * tau)  # Discount factor
        
        def idx(i, j):
            return i * N_nu + j
        
        # =============== BOUNDARY CONDITIONS ===============
        
        for i in range(N_S):
            for j in range(N_nu):
                k = idx(i, j)
                S_i = self.S[i]
                nu_j = self.nu[j]
                
                # Skip interior points
                if 0 < i < N_S-1 and 0 < j < N_nu-1:
                    continue
                
                # -------- BOUNDARY: S = 0 (i = 0) --------
                if i == 0:
                    if option_type == 'call':
                        # Call worth 0 at S=0
                        b1[k] = 0.0
                        b2[k] = 0.0
                    else:  # put
                        # Put worth K*exp(-r*tau) at S=0
                        b1[k] = K * df / 2
                        b2[k] = K * df / 2
                
                # -------- BOUNDARY: S = S_max (i = N_S-1) --------
                elif i == N_S - 1:
                    if option_type == 'call':
                        # Call: use Neumann condition ∂V/∂S = exp(-r_f*tau)
                        # This affects A1 through the derivative approximation
                        # The boundary value itself is: V ≈ S - K*exp(-r*tau)
                        value = S_i * np.exp(-0.0 * tau) - K * df  # assuming r_f = 0
                        b1[k] = value / 2
                        b2[k] = value / 2
                    else:  # put
                        # Put worth 0 at large S
                        b1[k] = 0.0
                        b2[k] = 0.0
                
                # -------- BOUNDARY: nu = 0 (j = 0) --------
                elif j == 0:
                    # At nu=0, volatility is zero - option approaches intrinsic value
                    # For small tau, V ≈ max(S - K, 0)*exp(-r*tau) for call
                    if option_type == 'call':
                        value = max(S_i - K * df, 0)
                    else:
                        value = max(K * df - S_i, 0)
                    b1[k] = value / 2
                    b2[k] = value / 2
                
                # -------- BOUNDARY: nu = nu_max (j = N_nu-1) --------
                elif j == N_nu - 1:
                    # At large variance: for call, V ≈ S*exp(-r_f*tau)
                    if option_type == 'call':
                        value = S_i * np.exp(-0.0 * tau)  # assuming r_f = 0
                    else:
                        value = K * df
                    b1[k] = value / 2
                    b2[k] = value / 2
        
        return b0, b1, b2                 

    def plotPDE(self, snapshots):
        """
        Visualize the Heston PDE solution as 3-D surfaces at four τ snapshots.

        Parameters
        ----------
        snapshots : dict  {label (str): U_grid (ndarray, shape (N_S, N_nu))}
            Produced by solve(..., return_snapshots=True).
            Each U_grid[i, j] = V(S_i, ν_j) at the corresponding τ.
            Expected keys (in order): 'T/4', 'T/2', '3T/4', 'T'

        Notes
        -----
        A shared colorscale is used across all four panels so that
        the evolution of V across τ is visually comparable.
        """
        from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

        # ── Downsample for 3-D rendering performance ──────────────────────────
        step_S  = max(1, self.N_S  // 40)
        step_nu = max(1, self.N_nu // 40)
        S_d  = self.S[::step_S]
        nu_d = self.nu[::step_nu]
        S_m, nu_m = np.meshgrid(S_d, nu_d, indexing='ij')

        # ── Global z-range for a consistent colorscale across panels ─────────
        z_min = min(v.min() for v in snapshots.values())
        z_max = max(v.max() for v in snapshots.values())

        labels_ordered = ['T/4', 'T/2', '3T/4', 'T']
        subtitles = {
            'T/4':  'τ = T/4   (close to maturity)',
            'T/2':  'τ = T/2',
            '3T/4': 'τ = 3T/4',
            'T':    'τ = T   (present value, t = 0)',
        }

        fig = plt.figure(figsize=(15, 11))
        fig.suptitle(
            "Heston PDE Solution  —  V(S, ν, τ) at Four Time-to-Maturity Snapshots\n"
            f"K={self.K if hasattr(self, 'K') else '—'}, "
            f"T={self.T if hasattr(self, 'T') else '—'}y, "
            f"κ={self.kappa}, θ={self.theta}, σ={self.sigma}, ρ={self.rho}, r={self.r}",
            fontsize=12, fontweight='bold', y=0.99,
        )

        for idx, lbl in enumerate(labels_ordered):
            ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
            Z  = snapshots[lbl][::step_S, ::step_nu]
            ax.plot_surface(
                S_m, nu_m, Z,
                cmap='plasma', alpha=0.88, linewidth=0,
                vmin=z_min, vmax=z_max,
            )
            ax.set_xlabel('Stock Price S', labelpad=7, fontsize=9)
            ax.set_ylabel('Variance ν',    labelpad=7, fontsize=9)
            ax.set_zlabel('V(S, ν)',        labelpad=7, fontsize=9)
            ax.set_title(subtitles[lbl], fontsize=10, pad=10)
            ax.view_init(elev=25, azim=-55)
            ax.tick_params(labelsize=7)

        # ── Shared colorbar ───────────────────────────────────────────────────
        sm = plt.cm.ScalarMappable(
            cmap='plasma',
            norm=plt.Normalize(vmin=z_min, vmax=z_max)
        )
        sm.set_array([])
        fig.colorbar(
            sm, ax=fig.axes,
            shrink=0.35, pad=0.04, aspect=25,
            label='Option Price V(S, ν)',
        )

        plt.subplots_adjust(top=0.93, hspace=0.08, wspace=0.02)
        plt.tight_layout()
        plt.show()
                                                
    def solve(self, K, T, payoff_func, theta=0.5,
              option_type='call', solver='Do',
              exercise_style='european',
              plot_sparsity=False,
              return_snapshots=False):
        """
        Solve the Heston PDE with Douglas or Craig-Sneyd ADI.

        Parameters
        ----------
        K              : float    – strike
        T              : float    – maturity
        payoff_func    : callable – payoff_func(S_mesh, K)
        dt             : float    – time step
        theta          : float    – ADI parameter (0.5 = Crank-Nicolson weight)
        option_type    : str      – 'call' or 'put'
        solver         : str      – 'Do' (Douglas) or 'CS' (Craig-Sneyd)
        exercise_style : str      – 'european' or 'american'
                          For 'american', after every ADI sub-step the IT-type
                          projection is applied on the full 2-D grid:
                              U^{n+1}_{ij} <- max(U^{n+1}_cont,ij , phi(S_i))
                          (Haentjens & in 't Hout, report §3.5).
        plot_sparsity  : bool     – plot sparsity of A0, A1, A2
        """
        dt=T/200
        def F_j(A_j, b_j, w):
            return A_j.dot(w) + b_j

        A0, A1, A2 = self.build_matrix_A()

        if plot_sparsity:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, A, title in zip(axes, [A0, A1, A2],
                                    ['A0 (mixed)', 'A1 (S)', 'A2 (nu)']):
                ax.spy(A[45:105, 45:105], markersize=1)
                ax.set_title(f'Sparsity of {title}')
            plt.tight_layout()
            #plt.savefig('Sparsities_A0A1A2.pdf', dpi=300)
            plt.show()
        # overwrite payoff_func for debug purposes
        if option_type == 'call':
            payoff_func = lambda S, K: np.maximum(S - K, 0)
        elif option_type == 'put':
            payoff_func = lambda S, K: np.maximum(K - S, 0)
        payoff   = payoff_func(self.S_mesh, K)
        U        = payoff.flatten()

        # Payoff on flat grid for American projection
        is_american  = (exercise_style.lower() == 'american')
        payoff_flat  = payoff.flatten()   # phi(S_i, nu_j) – but payoff depends only on S

        b0, b1, b2 = self.build_vector_b(t=0, T=T, K=K, option_type=option_type)

        N_t      = int(T / dt)
        dt_act   = T / N_t
        I        = sp.identity(self.N_total, format='csr')

        # ── NEW: define snapshot target steps ─────────────────────────────────────
        snap_steps = {
            'T/4':  max(1, N_t // 4) - 1,
            'T/2':  max(1, N_t // 2) - 1,
            '3T/4': max(1, 3 * N_t // 4) - 1,
            'T':    N_t - 1,
        }
        snapshots = {}
        # ──────────────────────────────────────────────────────────────────────────

        adi      = solver if isinstance(solver, str) else self.solver

        for n in range(N_t):
            if adi == 'Do':
                F  = F_j(A0, b0, U) + F_j(A1, b1, U) + F_j(A2, b2, U)
                Y0 = U + dt_act * F
                Y1 = spsolve(I - theta * dt_act * A1, Y0 - theta * dt_act * A1.dot(U))
                Y2 = spsolve(I - theta * dt_act * A2, Y1 - theta * dt_act * A2.dot(U))
                U_cont = Y2

            elif adi == 'CS':
                Y0 = U + dt_act * (A0.dot(U) + b0 + A1.dot(U) + b1 + A2.dot(U) + b2)
                Y1 = spsolve(I - theta * dt_act * A1, Y0 - theta * dt_act * A1.dot(U))
                Y2 = spsolve(I - theta * dt_act * A2, Y1 - theta * dt_act * A2.dot(U))
                Y3 = Y0 + 0.5 * dt_act * (A0.dot(Y2) - A0.dot(U))
                Y4 = spsolve(I - theta * dt_act * A1, Y3 - theta * dt_act * A1.dot(U))
                Y5 = spsolve(I - theta * dt_act * A2, Y4 - theta * dt_act * A2.dot(U))
                U_cont = Y5

            else:
                raise ValueError("solver must be 'Do' or 'CS'")

            # American early-exercise projection (IT-type, report §3.5):
            #   U^{n+1}_{ij} <- max(U^{n+1}_cont,ij , phi(S_i))
            if is_american:
                U = np.maximum(U_cont, payoff_flat)
            else:
                U = U_cont
            # ── NEW: save snapshot if this step matches a target ──────────────────
            if return_snapshots:
                for label, target in snap_steps.items():
                    if n == target:
                        snapshots[label] = U.reshape((self.N_S, self.N_nu)).copy()
            # ──────────────────────────────────────────────────────────────────────


        U_grid = U.reshape((self.N_S, self.N_nu))

        if return_snapshots:                        # ← NEW
            self.plotPDE(snapshots)

        return U_grid
 
    def get_option_price(self, U_grid, S0, nu0):
        """
        Interpolate option price at initial spot and variance.
        
        Parameters:
        -----------
        U_grid : numpy array (N_S, N_nu)
            Option values on grid
        S0 : float
            Initial asset price
        nu0 : float
            Initial variance
            
        Returns:
        --------
        price : float
            Interpolated option price
        """
        interp = RectBivariateSpline(self.S, self.nu, U_grid)
        return float(interp(S0, nu0))

class BlackScholesPDESolver:
    """
    Black-Scholes PDE solver using finite differences (implicit scheme).

    European pricing: standard implicit backward time-marching.
    American pricing: same time-stepping with an IT-type pointwise projection
                      at each time level enforcing the early-exercise constraint
                          V^n  <-  max(V^n_cont, payoff(S))
                      This is equivalent to solving the discrete LCP under
                      standard M-matrix monotonicity conditions (see report §3.5).

    The Black-Scholes PDE:
        dV/dt + (1/2)*sigma^2*S^2 * d2V/dS2 + r*S*dV/dS - r*V = 0
    """

    def __init__(self, S0, S_max, N, M, r, sigma, solver='implicit'):
        """
        Parameters
        ----------
        S0    : float  – initial asset price
        S_max : float  – upper bound of the spatial grid
        N     : int    – number of stock-price steps  (N+1 nodes)
        M     : int    – number of time steps          (M+1 nodes)
        r     : float  – risk-free rate
        sigma : float  – volatility
        solver: str    – reserved for future scheme variants ('implicit')
        """
        self.S0    = S0
        self.S_max = S_max
        self.N     = N
        self.M     = M
        self.r     = r
        self.sigma = sigma
        self.solver = solver

        self.S      = np.linspace(0.0, S_max, N + 1)
        self.DeltaS = S_max / N
        self.DeltaT = None

    # ------------------------------------------------------------------
    # Tridiagonal coefficients
    # ------------------------------------------------------------------

    def d_n(self, n, alpha, beta):
        return 1.0 + alpha * n**2 + beta

    def u_n(self, n, alpha, beta):
        return -0.5 * (alpha * (n - 1)**2 + beta * (n - 1))

    def l_n(self, n, alpha, beta):
        return -0.5 * (alpha * (n + 1)**2 - beta * (n + 1))

    # ------------------------------------------------------------------
    # Thomas algorithm
    # ------------------------------------------------------------------

    def thomas(self, a, b, c, d):
        n  = len(b)
        cp = np.zeros(n)
        dp = np.zeros(n)
        X  = np.zeros(n)

        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, n):
            dnum  = b[i] - a[i] * cp[i - 1]
            cp[i] = c[i] / dnum
            dp[i] = (d[i] - a[i] * dp[i - 1]) / dnum

        X[n - 1] = dp[n - 1]
        for i in range(n - 2, -1, -1):
            X[i] = dp[i] - cp[i] * X[i + 1]

        return X

    # ------------------------------------------------------------------
    # Build tridiagonal system
    # ------------------------------------------------------------------

    def build_tridiagonal(self, alpha, beta):
        N = self.N
        a = np.zeros(N - 1)
        b = np.zeros(N - 1)
        c = np.zeros(N - 1)

        for i in range(N - 1):
            n    = i + 1
            b[i] = self.d_n(n, alpha, beta)
            if i > 0:
                a[i] = self.l_n(n - 1, alpha, beta)
            if i < N - 2:
                c[i] = self.u_n(n + 1, alpha, beta)

        return a, b, c

    # ------------------------------------------------------------------
    # Solve  (European or American)
    # ------------------------------------------------------------------
    def plotPDE(self, Grid):
        """
        Visualize the Black-Scholes PDE solution on a 3-panel figure.

        Parameters
        ----------
        Grid : ndarray, shape (N+1, M+1)
            Full space-time solution returned by solve().
            Grid[i, m] = V(S_i, t_m), where m=0 is t=0 (present)
            and m=M is t=T (maturity).

        Panels
        ------
        1. 3-D surface  : V(S, t) over the full grid
        """
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        import matplotlib.gridspec as gridspec

        S      = self.S
        T      = self.T
        K      = self.K
        t_grid = np.linspace(0.0, T, self.M + 1)

        # ── Downsample for 3-D surface (performance) ──────────────────────────
        step = max(1, len(S) // 60)
        S_p  = S[::step]
        t_p  = t_grid[::step]
        C_p  = Grid[::step, ::step]
        T_m, S_m = np.meshgrid(t_p, S_p)

        # ── Greeks via central finite differences at t = 0 ────────────────────
        Delta = np.gradient(Grid[:, 0], S)
        Gamma = np.gradient(Delta, S)

        # ── Layout ────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            f"Black-Scholes PDE Solution  —  "
            f"S₀={self.S0}, K={K}, T={T}y, σ={self.sigma}, r={self.r}",
            fontsize=13, fontweight='bold', y=0.98,
        )
        gs = gridspec.GridSpec(
            1, 1, figure=fig,
            hspace=0.38, wspace=0.32,
            top=0.93, bottom=0.07, left=0.07, right=0.97,
        )

        # ── Panel 1 : 3-D surface ─────────────────────────────────────────────
        ax1  = fig.add_subplot(gs[0, :], projection='3d')
        surf = ax1.plot_surface(S_m, T_m, C_p, cmap='viridis', alpha=0.88, linewidth=0)
        fig.colorbar(surf, ax=ax1, shrink=0.45, pad=0.04, label='V(S, t)')
        ax1.set_xlabel('Stock Price S',  labelpad=8)
        ax1.set_ylabel('Time t',         labelpad=8)
        ax1.set_zlabel('Option Price V', labelpad=8)
        ax1.set_title('Option Price Surface V(S, t)', pad=10)
        ax1.view_init(elev=25, azim=-55)


        plt.tight_layout()
        plt.show()

    def solve(self, K, T, payoff_func, option_type='call',
              exercise_style='european', plotPDE = False):
        """
        Solve the Black-Scholes PDE.

        Parameters
        ----------
        K              : float    – strike price
        T              : float    – maturity (years)
        payoff_func    : callable – payoff_func(S, K) -> terminal payoff array
        option_type    : str      – 'call' or 'put'
        exercise_style : str      – 'european' or 'american'
                          For American options the early-exercise constraint
                          V^n <- max(V^n_cont, payoff(S)) is applied at every
                          time step after solving the linear system (IT projection,
                          report §3.5).  For a non-dividend-paying stock the
                          American call coincides with the European call, so the
                          projection has no effect in that case.

        Returns
        -------
        C : ndarray, shape (N+1, M+1)
            Option values on the full space-time grid.
        """
        N, M   = self.N, self.M
        S      = self.S
        r      = self.r
        S_max  = self.S_max

        DeltaT      = T / M
        self.DeltaT = DeltaT
        self.T      = T
        self.K      = K

        t     = np.linspace(0.0, T, M + 1)
        alpha = self.sigma**2 * DeltaT
        beta  = r * DeltaT

        is_american = (exercise_style.lower() == 'american')

        # ---- terminal payoff ----
        C          = np.zeros((N + 1, M + 1))
        C[:, -1]   = payoff_func(S, K)

        # ---- boundary conditions ----
        if option_type == 'call':
            C[0, :]  = 0.0
            C[-1, :] = S_max - K * np.exp(-r * (T - t))
        else:
            C[0, :]  = K * np.exp(-r * (T - t))
            C[-1, :] = 0.0

        # ---- payoff on full grid (for American projection) ----
        payoff_grid = payoff_func(S, K)   # shape (N+1,)

        # ---- tridiagonal system (constant across time steps) ----
        a, b, c = self.build_tridiagonal(alpha, beta)

        # ---- backward time-marching ----
        for m in range(M - 1, -1, -1):
            z      = np.zeros(N - 1)
            z[0]   = self.l_n(0, alpha, beta) * C[0, m]
            z[-1]  = self.u_n(N, alpha, beta) * C[N, m]
            rhs    = C[1:N, m + 1] - z

            # continuation value (European solve)
            C[1:N, m] = self.thomas(a, b, c, rhs)

            # American projection: enforce early-exercise constraint
            # V^n <- max(V^n_cont, phi(S))  at interior nodes
            if is_american:
                C[1:N, m] = np.maximum(C[1:N, m], payoff_grid[1:N])

        if plotPDE:
            self.plotPDE(C)

        return C

    # ------------------------------------------------------------------
    # get_option_price
    # ------------------------------------------------------------------

    def get_option_price(self, C, S0):
        """
        Interpolate the option price at spot S0 from column t=0 of the grid.
        """
        interp = interp1d(self.S, C[:, 0], kind='linear')
        return float(interp(S0))

def solve_pde(Model, Payoff, Grid_size, PlotPDEs_Bool = False):
    # 1) Check to which class the model belongs
    if isinstance(Model, BlackScholes):
        # 2) Create a PDE solver instance for Black-Scholes
        pde_solver = BlackScholesPDESolver(
            S0=Model.S0,
            S_max=5*Payoff.K,
            N=Grid_size[0],
            M=Grid_size[1],
            r=Model.r,
            sigma=Model.sigma
        )
        # 3) Solve the PDE
        if Payoff.payoff_type == 'call' :
            def payoff_func(S, K):
                return np.maximum(S - K, 0)
        elif Payoff.payoff_type == 'put' :
            def payoff_func(S, K):
                return np.maximum(K - S, 0)
        
        V = pde_solver.solve(
            K=Payoff.K,
            T=Payoff.T,
            option_type=Payoff.payoff_type,
            payoff_func=payoff_func,
            exercise_style=Payoff.ApplicationRule,
            plotPDE=PlotPDEs_Bool)
        # 4) Get the option price at S0
        price = pde_solver.get_option_price(V, S0=Model.S0)
        return price
    elif isinstance(Model, Merton):
        # 2) Create a PDE solver instance for Merton
        pde_solver = MertonJumpDiffusionPDESolver(
            S0=Model.S0,
            S_min = Payoff.K / 10,
            S_max=Payoff.K * 5,
            Nspace=Grid_size[0],
            Ntime=Grid_size[1],
            r=Model.r,
            sigma=Model.sigma,
            lam=Model.lam,
            muJ=Model.muJ,
            sigJ=Model.sigmaJ
        )
        # 3) Solve the PDE
        if Payoff.payoff_type == 'call' :
            def payoff_func(x, K): return np.maximum(np.exp(x) - K, 0.0)
        elif Payoff.payoff_type == 'put' : 
            def payoff_func(x, K):  return np.maximum(K - np.exp(x), 0.0)

        V = pde_solver.solve(
            K=Payoff.K,
            T=Payoff.T,
            payoff_func=payoff_func,
            option_type=Payoff.payoff_type,
            exercise_style=Payoff.ApplicationRule,
            plotPDE_Bool = PlotPDEs_Bool        )
        # 4) Get the option price at S0
        price = pde_solver.get_option_price(V, S0=Model.S0)
        return price
    elif isinstance(Model, Heston):
        # 2) Create a PDE solver instance for Heston
        pde_solver = HestonPDESolver(
            S_max=Payoff.K * 5,
            nu_max=5,
            N_S=Grid_size[0],
            N_nu=Grid_size[1],
            r=Model.r,
            kappa=Model.kappa,
            theta=Model.theta,
            sigma=Model.xi,
            rho=0,
            solver='Do'  # or 'CS'
        )
        # 3) Solve the PDE
        if Payoff.payoff_type == 'call' :
            def payoff_func(S, K):
                return np.maximum(S - K, 0)
        elif Payoff.payoff_type == 'put' :
            def payoff_func(S, K):
                return np.maximum(K - S, 0)
        V = pde_solver.solve(
            K=Payoff.K,
            T=Payoff.T,
            payoff_func=Payoff.payoff_func,
            option_type=Payoff.payoff_type,
            exercise_style=Payoff.ApplicationRule,
            plot_sparsity=False,
            return_snapshots = PlotPDEs_Bool
        )
        # 4) Get the option price at S0 and nu0
        price = pde_solver.get_option_price(V, S0=Model.S0, nu0=Model.v0)
        return price
    elif isinstance(Model, TwoUnderlyings):
        # 2) Create a PDE solver instance for Two Underlyings
        pde_solver = TwoUnderlyingsPDESolver(
            S1_0=Model.S0_1,
            S2_0=Model.S0_2,
            S1_max=Model.S0_1 * 10,
            S2_max=Model.S0_2 * 10,
            N_S1=Grid_size[0],
            N_S2=Grid_size[1],
            r=Model.r,
            SigmaMatrix=np.array([[Model.sigma1**2, Model.rho * Model.sigma1 * Model.sigma2],
                                  [Model.rho * Model.sigma1 * Model.sigma2, Model.sigma2**2]]), # this is covariance matrix
            rho=Model.rho,
            solver='Do'  # or 'CS',
        )
        # overwrite payoff_func for debug purposes
        if Payoff.payoff_type == 'call_spread':
            payoff_func = lambda S1, S2, K: np.maximum(S1 - S2 - K, 0)
        elif Payoff.payoff_type == 'put_spread':
            payoff_func = lambda S1, S2, K: np.maximum(K - (S1 - S2), 0)
        elif Payoff.payoff_type == 'call_basket':
            payoff_func = lambda w1, w2, S1, S2, K: np.maximum(w1 * S1 + w2 * S2 - K, 0)
        elif Payoff.payoff_type == 'put_basket':
            payoff_func = lambda w1, w2, S1, S2, K: np.maximum(K - (w1 * S1 + w2 * S2), 0)
        elif Payoff.payoff_type == 'exchange':
            payoff_func = lambda S1, S2: np.maximum(S1 - S2, 0)
        # 3) Solve the PDE
        V = pde_solver.solve(
            K=Payoff.K,
            T=Payoff.T,
            payoff_func=payoff_func,
            option_type=Payoff.payoff_type,
            exercise_style=Payoff.ApplicationRule
        )
        # 4) Get the option price at S1_0 and S2_0
        price = pde_solver.get_option_price(V, S10=Model.S0_1, S20=Model.S0_2)
        return price
    else:
        raise ValueError("Unsupported model type")