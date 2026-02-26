import numpy as np

# --- LC photometric bands ----
filters = np.array(['V','I'])
prefixs = np.array([0, 1])
lc_colors = ['yellowgreen','orange']
lc_markers = ['o','s']

activated_bands = [1] # in prefixs
n_bands = len(activated_bands)

# --- fourier series ---
M_MIN = 3   # minimum Fourier series order
M_MAX = 15  # Max fourier order
ERR_FLOOR = 0.01

# --- adaptive fitting ---
THRESHOLD = 0.1    # threshold   

# --- fitting range ---
Amin, Amax = 0.05, 2 # [mag]
pmin, pmax = 0.2, 300 # [days]
delta_P_tol = 0.0001 # [days] 

opt_method = 'lasso'
lam = 1e-4 # regularization coeff

n0 = 2 # oversampling
harmonics = 1 # maximum order of harmonics [P0, P0/n, n*P0]
K = 5 # peak search for aliasing cleaning
snr = 5 # relative peak strength for aliased frequency detection

period_fit = False