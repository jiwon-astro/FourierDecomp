import numpy as np

# --- LC photometric bands ----
activated_bands = [1] # in prefixs
n_bands = len(activated_bands)

# --- fourier series ---
M_MIN = 3   # minimum Fourier series order
M_MAX = 15  # Max fourier order
ORDERS_MAX = 1 + np.arange(M_MAX).reshape(-1, 1) # Fourier series order
ERR_FLOOR = 0.01

# --- adaptive fitting ---
THRESHOLD = 0.05    # threshold   

# --- fitting range ---
Amin, Amax = 0, 2 # [mag]
pmin, pmax = 0.2, 300 # [days]'



