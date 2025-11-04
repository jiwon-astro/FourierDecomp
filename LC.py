import numpy as np
from .params import M_MAX

# --- LC identifier ---
ident_names = ['ID','pulsation','RA','Dec','OGLE-IV ID','OGLE-III ID','OGLE-II ID','other']
phot_names = ['t','mag','emag']

# --- photometric bands ---
filters = np.array(['V','I'])
prefixs = np.array([0, 1])
colors = ['yellowgreen','orange']
markers = ['o','s']

def compute_phase(t, P):
    return (t / P) % 1.0

def phase_gap_exceeds(t, P, threshold=0.05, M_fit=M_MAX): 
    if len(t) < (M_fit + 2): return True # M_fit 사용
    phi = compute_phase(t, P)
    phi_sorted = np.sort(phi)
    gaps = np.diff(np.r_[phi_sorted, phi_sorted[0] + 1.0])
    return gaps.max() > threshold