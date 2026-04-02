from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm.notebook import tqdm
from astropy.table import Table

import numpy as np
import pandas as pd

from . import params
from .IO import epoch_arrays, get_data_config
from .LSQ import F
    
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
def _calc_quality_single(sid, args_ft, theta_ft, flt, n_grid=50, statistic='median'):

    M_MAX = params.M_MAX
    t_ft, mag_ft, emag_ft = args_ft
    n_epoch = len(t_ft)
    E = theta_ft[-1]
    P = theta_ft[-2]

    phi_grid = np.arange(0, 1, 1/n_grid) # set phase grid
    # calculate residual
    phi_ft = ((t_ft - E)/P)%1
    fval = F(theta_ft, t_ft, M_MAX, coef_mode='AQ')
    res_ft = mag_ft - fval # residual

    # evaluation at phase grid
    theta_ft_phi = theta_ft.copy()
    theta_ft_phi[-2] = 1; theta_ft_phi[-1] = 0
    phi_grid = np.arange(0, 1, 1/n_grid) # set phase grid
    fval_grid_ft  = F(theta_ft_phi, phi_grid, M_MAX, coef_mode='AQ')

    # statistics
    occupation_ft = occupied_fraction(phi_ft, n_grid=n_grid)
    phase_mask_ft = phase_gap_mask(phi_ft, n_grid=n_grid)
    bin_res_ft    = binned_residual_function(phi_ft, res_ft, 
                                            n_grid=n_grid, statistic=statistic)
    gmax_ft       = gap_max(phi_ft)
    H_coverage_ft = coverage_entropy(phi_ft, n_grid=n_grid)
    
    return {'ID':sid, 'band':flt, 'N':n_epoch,'occupied_fraction':occupation_ft, 
            'gmax':gmax_ft, 'coverage_entropy':H_coverage_ft,
            'fval_grid':fval_grid_ft,'residual_grid':bin_res_ft, 'phase_mask':phase_mask_ft}


def lightcurve_quality(sid, df_FD, n_grid=50, mode=None, selected_bands=None,
                       statistic='median', period_key='P', epoch_key='E'):
    if mode is None: mode = get_data_config().mode
    cfg = get_data_config(mode)
    filters = cfg.filters
    if selected_bands is None: selected_bands = filters # all-band
    M_MAX = params.M_MAX

    # -------------------
    # load data
    # ------------------
    # 1) Fourer Decomposition result
    fd_row = df_FD[df_FD['ID'] == sid]
    if len(fd_row) == 0:
        raise ValueError(f"ID={sid} not found in df_FD")
    
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
    # evaluation
    # ------------------
    quality_list = []
    for i, bm in enumerate(bmask):
        flt = filters[i]
        if not flt in selected_bands: continue
        args_ft = (t[bm], mag[bm], emag[bm])
        theta_ft = np.array([m0s[i], amps[i], As, Qs, P, E], dtype=object)
        q = _calc_quality_single(sid, args_ft, theta_ft, flt, n_grid=50, statistic='median')
        if q is not None: quality_list.append(q)
    return quality_list

def build_quality_table(ids, df_FD, mode=None, selected_bands=None, n_grid=50,
                        statistic='median', output_fpath=None, overwrite=True):
    if mode is None: mode = get_data_config().mode

    rows = []
    for sid in tqdm(ids):
        try:
            row_i = lightcurve_quality(
                sid=sid,
                df_FD=df_FD,
                n_grid=n_grid,
                mode=mode,
                selected_bands=selected_bands,
                statistic=statistic,
            ) # -> list
            if len(row_i) > 0: rows += row_i
        except Exception as e:
            print(f"[quality] skip ID={sid}: {e}")

    out = Table(rows=rows)
    if output_fpath is not None:
        output_fpath.parent.mkdir(parents=True, exist_ok=True)
        out.write(output_fpath, format='ascii.ecsv', overwrite=overwrite)
    return out

