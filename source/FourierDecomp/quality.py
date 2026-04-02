from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

from . import params
from .IO import epoch_arrays, get_data_config
from .LSQ import F

# ========================================
# Data containers
# ========================================
@dataclass
class QualityRecord:
    sid: str
    band: str
    n_epoch: int
    fval_grid: float
    occupied_fraction: float
    phase_mask: float
    binned_residual: float
    gmax: float
    coverage_entropy: float

    def to_row(self):
        return asdict(self)
    
# ========================================
# Phase coverage utils
# ========================================
def wrap_phase(phase):
    phase = np.asarray(phase, dtype=float)
    return np.mod(phase, 1.0)

def sort_phase(phase):
    return np.sort(wrap_phase(phase))

def circular_phase_gaps(phase, sorted=False):
    """Return sorted circular phase gaps."""
    if not sorted: p = sort_phase(phase)
    if p.size == 0: return np.array([1.0], dtype=float)
    if p.size == 1: return np.array([1.0], dtype=float)

    diffs = np.diff(p)
    wrap_gap = 1.0 - p[-1] + p[0]
    gaps = np.concatenate([diffs, [wrap_gap]])
    return gaps

def gap_max(phase, sorted=False):
    return float(np.max(circular_phase_gaps(phase, sorted=sorted)))

def phase_bin_counts(phase, n_grid = 50):
    p = wrap_phase(phase)
    if p.size == 0: return np.zeros(n_grid, dtype=int)
    bins = np.floor(p * n_grid).astype(int)
    bins = np.clip(bins, 0, n_grid - 1)
    counts = np.bincount(bins, minlength=n_grid)
    return counts.astype(int)

def occupied_fraction(phase, n_grid = 50):
    counts = phase_bin_counts(phase, n_grid=n_grid).astype(float)
    return np.count_nonzero(counts) / n_grid

def coverage_entropy(phase, n_grid = 50):
    # degree of uniformity of phase gap distributes
    counts = phase_bin_counts(phase, n_grid=n_grid).astype(float)
    total = counts.sum()
    if total <= 0: return 0.0
    p = counts / total
    p = p[p > 0]
    h = -(p * np.log(p)).sum()
    return float(h / np.log(n_grid))

# ====================================
# binning
# ====================================
def phase_gap_mask(phase, n_grid = 50):
    """
    Boolean mask of length n_grid:
      1 -> this phase bin contains at least one observation
      0 -> no observation in this bin
    """
    return (phase_bin_counts(phase, n_grid=n_grid) > 0).astype(np.int8)

def _window_coverage(mask, center_bin, width = 8):
    # circular padding
    n = len(mask); wh = width//2
    idx = [(center_bin + k) % n for k in range(-wh, wh + 1)]
    return np.mean(mask[idx])

def binned_residual_function(phase, residual, n_grid = 50, statistic = "mean"):
    """
    Phase: phi_i = ((t_i-E)/P)%1
    Residual: res_i = m_i - m_model_i (best-fit Fourier Series, evaluated on t_i)
    Unsampled bins are NaN by default and should be paired with phase_mask.
    """
    phase = wrap_phase(phase)
    bins = np.floor(phase * n_grid).astype(int)
    bins = np.clip(bins, 0, n_grid - 1)

    out = np.full(n_grid, np.nan, dtype=float)
    for i in range(n_grid):
        sel = (bins == i)
        if not np.any(sel): continue
        vals = residual[sel]
        if statistic == "mean": out[i] = np.mean(vals)
        elif statistic == "median": out[i] = np.median(vals)
        elif statistic == "rms": out[i] = np.sqrt(np.mean(vals ** 2))
        else: raise ValueError(f"Unsupported statistic: {statistic}")
    return out

# ==================================
# main function
# ==================================
def lightcurve_quality(sid, df_FD, n_grid=50, mode=None, bands=None,
                       statistic='median', period_key='P', epoch_key='E'):
    if mode is None: mode = get_data_config().mode
    cfg = get_data_config(mode)
    filters = cfg.filters
    if bands is None: bands = filters # all-band

    M_MAX = params.M_MAX
    # -------------------
    # load data
    # ------------------
    # 1) Fourer Decomposition
    fd_row = df_FD[df_FD['ID'] == sid]
    m0s  = fd_row[[f"m0_{f}" for f in filters]].to_numpy(dtype=float)[0] # mean magnitude
    amps = fd_row[[f"amp_{f}" for f in filters]].to_numpy(dtype=float)[0] # amplitude
    P = float(fd_row[period_key]) # period
    E = float(fd_row[epoch_key])  # epoch
    coef_A_names = [f"A{i}" for i in range(1, M_MAX + 1)]
    coef_Q_names = [f"Q{i}" for i in range(1, M_MAX + 1)]
    As = fd_row[coef_A_names].to_numpy(dtype=float)[0] # fourier coefficients
    Qs = fd_row[coef_Q_names].to_numpy(dtype=float)[0] 

    # 2) Lightcurve
    t, mag, emag, bands = epoch_arrays(ls_data, sid, mode=mode)
    bmask = [(bands == band) for band in filters]

    # ------------------
    # grid evaluation
    # ------------------
    phi_grid = np.arange(0, 1, 1/n_grid)
    quality_dict = {}
    for i, bm in enumerate(bmask):
        flt = filters[i]
        if not flt in bands: continue
        t_ft, mag_ft, emag_ft = t[bm], mag[bm], emag[bm]
        n_epoch = len(t_ft)

        # calculate residual
        phi_ft = ((t_ft - E)/P)%1
        theta_ft = np.array([m0s[i], amps[i], As, Qs, 1, 0])
        fval = F(theta_ft, phi_ft, M_MAX, coef_mode='AQ')
        res_ft = mag_ft - fval # residual

        # statistics
        fval_grid_ft  = F(theta_ft, phi_grid, M_MAX, coef_mode='AQ')
        occupation_ft = occupied_fraction(phi_ft, n_grid=n_grid)
        phase_mask_ft = phase_gap_mask(phi_ft, n_grid=n_grid)
        bin_res_ft    = binned_residual_function(phi_ft, res_ft, 
                                                n_grid=n_grid, statistic=statistic)
        gmax_ft       = gap_max(phi_ft)
        H_coverage_ft = coverage_entropy(phi_ft, n_grid=n_grid)
        
        quality_ft = QualityRecord(sid, band=flt, n_epoch=n_epoch,
                                   fval_grid=fval_grid_ft, occupied_fraction=occupation_ft, phase_mask=phase_mask_ft,
                                   binned_residual=bin_res_ft, gmax=gmax_ft, coverage_entropy=H_coverage_ft)
        quality_dict[flt] = quality_ft
    return quality_dict

def build_quality_table(ids, df_FD, mode=None, bands=None, n_grid=50,
                        statistic='median', output_fpath=None, overwrite=True):
    if mode is None: mode = get_data_config().mode

    tabs = []
    for sid in tqdm(ids):
        try:
            tab_i = lightcurve_quality(
                sid=sid,
                df_FD=df_FD,
                n_grid=n_grid,
                mode=mode,
                bands=bands,
                statistic=statistic,
            )
            if len(tab_i) > 0:
                tabs.append(tab_i)
        except Exception as e:
            print(f"[quality] skip ID={sid}: {e}")

