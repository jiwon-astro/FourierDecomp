import numpy as np

# --- mode specific defaults ----
mode_default = 'gaia'
DATA_CONFIGS = {
    'ogle': {
        'filters': np.array(['V', 'I']),
        'prefixs': np.array([0, 1]),
        'lc_colors': ['yellowgreen', 'orange'],
        'lc_markers': ['o', 's'],
        'activated_bands': [1],  # indices in prefixs
        'wesenheit_factor': 1.55
    },
    'gaia': {
        'filters': np.array(['g', 'bp', 'rp']),
        'prefixs': np.array([0, 1, 2]),
        'lc_colors': ['#000000', '#0343DF', '#E50000'],
        'lc_markers': ['o', 's', 'D'],
        'activated_bands': [0, 1, 2],  # indices in prefixs
        'wesenheit_factor':1.90
    },
    'ztf': {
        'filters': np.array(['zg', 'zr', 'zi']),
        'prefixs': np.array([0, 1, 2]),
        'lc_colors': ['#4D96FF', '#87C55F', '#FF6B6B'],
        'lc_markers': ['o', 's', 'D'],
        'activated_bands': [0, 1, 2],  # indices in prefixs
        'wesenheit_factor':[2.274, 2.905, 4.051] # (1) gi, (2) gr, (3) ri (Ngeow+2021) / Ag=3.518E,Ar=2.617E, andAi=1.971E (Green+2019)
    }
}

# --- period search ---
n0 = 2 # oversampling
harmonics = 1 # maximum order of harmonics [P0, P0/n, n*P0]
K = 5 # peak search for aliasing cleaning
snr = 5 # relative peak strength for aliased frequency detection

# --- light curve quality features ---
n_grid = 50

# --- fourier series ---
M_MIN = 3   # minimum Fourier series order
M_MAX = 15  # Max fourier order
M_PAD = 2 # scanning range above M_trunc
ERR_FLOOR = 0.001

# --- adaptive fitting ---
THRESHOLD = 0.1    # threshold
#ORDER_BIC_TOL = 10.0 # BIC tolerence 
BIC_TOL_WEIGHT = 0.1 # tol = WEIGHT *sqrt(2νeff)max(1,χ2_red_best)
RES_CORR_WEIGHT = 3 # residual unit-lag autocorrelation weight (for order selection)
tie_breaker = "minimum" # selection rule for M_fits with similar BIC values

# --- fitting range ---
# Peak-to-peak amplitude range for each band [mag]
Amin, Amax = 0.05, 2

# Harmonic coefficient bounds A_j [shape coefficients]
# Scalar -> same bound for every harmonic
# Array   -> order-dependent bounds (A1, A2, ...)
Amin_harmonic = 0.0 # always
Amax_harmonic = 1.5

pmin, pmax = 0.2, 300 # [days]
delta_P_tol = 0.0001 # [days]

coef_mode = 'ab' # ab (alpha, beta) or AQ (A, Q)
quality_weight = True # photometric quality weight based on residual

opt_method = 'lasso'
lam0 = 1e-2 # regularization coeff
lam_min = 1e-4
lam_max = 5e-2

lam_spike = 1e-1 # spike penalty weight
lam_h     = 10 # harmonic penalty weight
lam_sl    = 3e-2 # slope residual penalty weight
n_grid    = 50 # grid evaluation (for spike penalty)
power     = 4.0 # harmonic penalty: ~ j^power
branch_err_frac   = 0.03

N_BRANCH_BINS     = 5 
MIN_BRANCH_POINTS = 5
HUBER_DELTA       = 2.5

period_fit   = False
use_optim    = True
adaptive_lam = True
use_refit    = True

REFIT_SIGMA = 3.0
REFIT_MAXITER = 3

init = 'rrfit' # lsq or rrfit
