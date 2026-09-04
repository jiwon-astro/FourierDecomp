import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Sequence

from gatspy import periodic # multiband lomb-scargle
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import median_filter
from scipy.optimize import minimize_scalar # brent method
from astropy.stats import sigma_clip

from .params import pmin, pmax, n0, delta_P_tol
from .IO import get_data_config

from scipy.ndimage import median_filter, gaussian_filter1d

import warnings
warnings.filterwarnings('ignore')

# ==================================
fmin = 1/pmax # expected minimum frequency [days^-1] (<100d)
fmax = 1/pmin # expected maximum frequency [days^-1](anomalous cepheids - 0.4d)
Nterms = 1 # truncated fourier series for Lomb-Scargle 
#f_alias = np.array([1.0]) # list of aliased frequency [days^-1]
#f_alias_tol = 0.005
# =====================================

# Define Lomb-Scargle model (default Nbase = 1, Nband = 1) + Fast
model = periodic.LombScargleMultibandFast(fit_period = True, Nterms=Nterms,
                                          optimizer_kwds={'quiet':True}) #MultibandFast -> silence_warnings=False option N/A
model.optimizer.period_range = (pmin,pmax)

def calc_fgrid(t, n0=5, period_min=None, period_max=None):
    # n0 : oversampling ratio
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        raise ValueError("period_search_requires_at_least_two_epochs")
    t0, T = np.min(t), np.ptp(t) #initial epoch & length of time window
    if not np.isfinite(T) or T <= 0:
        raise ValueError("period_search_requires_positive_baseline")
    period_min = pmin if period_min is None else float(period_min)
    period_max = pmax if period_max is None else float(period_max)
    if period_max <= period_min:
        raise ValueError(
            f"period_search_baseline_too_short:{T:.6g}d")
    tn = np.expand_dims(t-t0, axis=0).T
    
    # Frequency Grid
    delta_f = 1/(n0*T) # grid spacing (VanderPlas 2018)
    #delta_f = (len(t)/2)/(n0*T) # grid spacing (Frescura 2008)
    f_lo = 1.0 / period_max
    f_hi = 1.0 / period_min
    f = np.arange(f_lo, f_hi + delta_f, delta_f)
    period = 1/f
    
    return f, period, delta_f

def local_background(y, win=201):
    bg = median_filter(y, size=win, mode='nearest')
    resid = y - bg
    mad = median_filter(np.abs(resid), size=win, mode='nearest')
    sig = 1.4826 * mad
    sig = np.maximum(sig, 1e-6)
    return bg, sig

# ======= window function =======
def window_function(t, bmask, f):
    t0, T = t[0], t[-1]-t[0] #initial epoch & length of time window    
    # calculate window function power spectrum
    P_W = np.zeros_like(f)
    for bm in bmask :
        t_ft = t[bm]
        tn_ft = np.expand_dims(t_ft-t0, axis=0).T
        # window power spectrum of current band
        phase = np.exp(-2*np.pi*1j*f*tn_ft)
        P_W += np.abs(np.sum(phase,axis=0))**2
    
    # Normalize (maximum = 1)
    P_W /= max(P_W) 
    return P_W

def window_alias_mask(freqs, f0, f_alias, f_alias_tol=0.01, n_alias = 2):
    # f_alias (scalar)
    # masking ±f_alias [d^-1] aliases: any strong power where |f' - (f ± f_alias)| is tiny
    mask = np.zeros_like(freqs, dtype = 'bool')
    for n in range(1, n_alias+1):
        mask|=(np.abs(freqs - (f0 + n*f_alias)) < f_alias_tol) | (np.abs(freqs - (f0 - n*f_alias)) < f_alias_tol)
    return mask

# ======= aliased / harmonic solutions ========
def harmonic_periods(P0, harmonics=2, period_min=None, period_max=None):
    period_min = pmin if period_min is None else float(period_min)
    period_max = pmax if period_max is None else float(period_max)
    Ps = [P0]
    for n in range(2, harmonics + 1):
        if n * P0 <= period_max: Ps.append(n * P0) # subharmonic in frequency (overtone in period)
        if P0 / n >= period_min: Ps.append(P0 / n) # harmonic in frequency (subharmonic in period)
    return Ps

def aliased_periods(P, alias_freqs, n=1, m=1):
    alias_freqs = np.asarray(alias_freqs).reshape(-1,1)
    if len(P)==0 or len(alias_freqs) ==0:
        return np.array([])
    P2s = []
    for sign in [-1, 1]:
        Ps = np.vstack([1/np.abs(n/P+m*sign*alias_freqs)])
        P2s.append(Ps)
    return np.vstack(P2s)

def cluster_periods(periods, logP_tol=0.05, min_gap=0.0, max_width=None,
                    return_boundary=True):
    periods = np.asarray(periods, dtype=float)
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if len(periods) == 0: return []

    logPs = np.sort(np.log10(periods))
    intervals = [(x - logP_tol, x + logP_tol, [x]) for x in logPs]

    merged = []
    cur_lb, cur_ub, cur_members = intervals[0]
    for lb, ub, members in intervals[1:]:
        overlap = (lb <= cur_ub + min_gap)
        new_width = max(cur_ub, ub) - cur_lb
        if overlap and (max_width is None or new_width <= max_width):
            cur_ub = max(cur_ub, ub)
            cur_members.extend(members)
        else:
            merged.append((cur_lb, cur_ub, cur_members))
            cur_lb, cur_ub, cur_members = lb, ub, members

    merged.append((cur_lb, cur_ub, cur_members))
    if return_boundary:
        boundaries = [(lb, ub) for lb, ub, _ in merged]
        return boundaries
    return [np.array(m) for _, _, m in merged] # members

# ========= Period Search with Lomb-Scargle algorithm ==========
def robust_period_search(t, mag, emag, bands, 
                         n0 = 5, K = 8, snr = 3, harmonics = 2,
                         plot = False, mode=None):
    t = np.asarray(t, dtype=float)
    mag = np.asarray(mag, dtype=float)
    emag = np.asarray(emag, dtype=float)
    bands = np.asarray(bands).astype(str)

    cfg = get_data_config(mode)
    filters = cfg.filters
    activated_bands = cfg.activated_bands
    active_names = [str(filters[ib]) for ib in activated_bands]

    # Use only scientifically activated bands.  In particular, a single
    # inactive-band point must not make the multiband normal matrix singular.
    observed_names = [
        band for band in active_names if np.count_nonzero(bands == band) >= 2
    ]
    if not observed_names:
        raise ValueError("missing_observed_active_band")
    use = np.isin(bands, observed_names)
    use &= np.isfinite(t) & np.isfinite(mag) & np.isfinite(emag) & (emag > 0)
    t_fit, mag_fit, emag_fit, bands_fit = (
        t[use], mag[use], emag[use], bands[use])
    if t_fit.size < 3:
        raise ValueError("period_search_insufficient_active_epochs")

    baseline = float(np.ptp(t_fit))
    # gatspy explicitly disallows periods longer than the time baseline.
    period_max = min(float(pmax), baseline * (1.0 - 1e-8))
    freqs, periods, delta_f = calc_fgrid(
        t_fit, n0=n0, period_min=pmin, period_max=period_max)
    fmin_local, fmax_local = 1.0 / period_max, 1.0 / pmin

    local_model = periodic.LombScargleMultibandFast(
        fit_period=True, Nterms=Nterms, optimizer_kwds={'quiet': True})
    local_model.optimizer.period_range = (pmin, period_max)
    bmask = [(bands_fit == band) for band in observed_names]

    # 1) evaluate Lomb-Scargle power
    local_model.fit(t_fit, mag_fit, emag_fit, bands_fit)
    Pf_LS = np.asarray(local_model.periodogram(periods), dtype=float)
    if not np.any(np.isfinite(Pf_LS)):
        raise ValueError("period_search_nonfinite_periodogram")

    # 2) Finding peak (coarse search)
    #sep = int(sep_frac*fmin/delta_f)
    sigma_Pf_LS = np.nanstd(Pf_LS)
    pidx = find_peaks(Pf_LS, height = snr * sigma_Pf_LS)[0]
    # select peaks having large contrast (prominence)
    if pidx.size:
        prom = peak_prominences(Pf_LS, pidx)[0]
        prom_thres = np.median(prom)
        pidx = find_peaks(
            Pf_LS, height=snr * sigma_Pf_LS,
            prominence=prom_thres)[0]
    if pidx.size == 0:
        # A low-S/N source still needs a deterministic candidate.  Prefer
        # separated local maxima; fall back to the global maximum only.
        pidx = find_peaks(Pf_LS)[0]
    if pidx.size == 0:
        pidx = np.array([int(np.nanargmax(Pf_LS))])
    # select K peaks
    pidx = pidx[np.argsort(Pf_LS[pidx])[::-1]][:K]
    P_coarse = periods[pidx]
    
    # zoom-in search
    def objective_func(P):
        if P <= pmin or P >= period_max: return np.inf
        return -local_model.score([P])[0]
   
    P_refined = []
    for P0 in P_coarse:
        # grid refinement to reach to the desired accuracy
        if delta_f * P0 > delta_P_tol:
            f0 = 1.0 / P0
            # setting bounds at nearby f0
            f_low = max(fmin_local, f0 - delta_f)
            f_high = min(fmax_local, f0 + delta_f)
            # minimization
            res = minimize_scalar(objective_func, 
                                  bounds=(1.0/f_high, 1.0/f_low), 
                                  method='bounded')
            if res.success: 
                #print(f'{P0} / {res.x}')
                P_refined.append(res.x)
        else: P_refined.append(P0)

    # 3) period candidates
    Ps = []
    for P0 in P_refined: 
         Ps += harmonic_periods(
             P0, harmonics, period_min=pmin, period_max=period_max)
    Ps = np.unique(np.asarray(Ps, dtype=float))
    if Ps.size == 0:
        raise ValueError("period_search_no_candidate")
    Zs = local_model.score(Ps)
    mask = (Zs > 3*sigma_Pf_LS) # if peaks are significant
    Ps, Zs = Ps[mask], Zs[mask]

    if Ps.size == 0:
        best = int(np.nanargmax(Pf_LS))
        Ps = np.array([periods[best]], dtype=float)
        Zs = np.array([Pf_LS[best]], dtype=float)

    if plot: plot_LS(freqs, periods, Pf_LS, peaks = [Ps, Zs],
                     thresh = [sigma_Pf_LS, snr * sigma_Pf_LS])
        
    return Ps, Zs

def period_fit_boundary_search(t, mag, emag, bands, n0 = 5, K = 5, Kw = 10, 
                               snr_LS = 3, snr_window = 5, harmonics = 2,
                               logP_tol = 0.1, max_width=1.0):
    freqs, periods, delta_f = calc_fgrid(t, n0 = n0)
    
    cfg = get_data_config()
    filters = cfg.filters
    activated_bands = cfg.activated_bands
    bmask = [(bands==filters[ib]) for ib in activated_bands] # original definition: include all passbands?

    # 1) evaluate Lomb-Scargle power
    model.fit(t, mag, emag, bands)
    Pf_LS = model.periodogram(periods)
    sigma_Pf_LS = np.std(Pf_LS)
    pidx = find_peaks(Pf_LS, height = snr_LS * sigma_Pf_LS)[0] 
    # select peaks having large contrast (prominence)
    prom = peak_prominences(Pf_LS, pidx)[0]
    prom_thres = np.median(prom)
    pidx = find_peaks(Pf_LS, height = snr_LS * sigma_Pf_LS, prominence = prom_thres)[0] 
    # select K peaks
    pidx = pidx[np.argsort(Pf_LS[pidx])[::-1]][:K]
    P_coarse = periods[pidx]; Z_coarse = Pf_LS[pidx]

    # 3) window function
    Pw = window_function(t, bmask, freqs)
    distance = max(10, int(5*n0))
    height_thr = np.percentile(Pw, 95)
    prom_thr = 5.0 * np.median(np.abs(Pw - np.median(Pw)))
    pidx_w, props_w = find_peaks(Pw, height=height_thr, prominence=prom_thr,
                          distance=distance)
    """
    bkg_Pw, sig_Pw = local_background(Pw, win = n0*100+1) # calculate local background level by moving-mdeian filter
    Pw_flat = Pw - bkg_Pw #/ sig_P_W
    pidx_w, props_w = find_peaks(Pw_flat, height = snr_window * sig_Pw, 
                    prominence=max(5, 2*snr_window)*sig_Pw, distance=distance)
    """
    prom_alias = props_w['prominences']
    pidx_w_sorted = pidx_w[np.argsort(prom_alias)][::-1] # sorting with respect to prominences 
    # Pw_alias = Pw[pidx_sorted][:Kw]
    alias_freqs = freqs[pidx_w_sorted][:Kw].tolist()
    
    # 4) calculate aliased periods
    P_alias = []
    for n in range(1, 1+harmonics):
        P2s = aliased_periods(P_coarse, alias_freqs, n=n, m=1)
        if (P2s is None) or len(P2s)==0: 
            continue
        P_alias.append(np.hstack(P2s))
    P_alias = np.hstack(P_alias)
    P_alias = P_alias[(P_alias >= pmin) & (P_alias <= pmax)] # ensure range

    # 5) clustering 
    logP_cluster = cluster_periods(P_alias, logP_tol=logP_tol, max_width=max_width,
                                   return_boundary=True)
    return P_coarse, Z_coarse, alias_freqs, logP_cluster
    
# ======== plot lomb-scargle ===========
def plot_LS(freqs, periods, Pf_LS, peaks = None, thresh = None):
    fig, ax = plt.subplots(2,1, figsize = (15,8))
    # freq-domain
    ax[0].plot(freqs, Pf_LS, color = 'k')
    ax[0].hlines(thresh, fmin-0.05, fmax, ls = 'dotted', color='darkred')
    ax[0].set_xlim(fmin-0.05, fmax)
    ax[0].set_xlabel('Frequencies [$\\rm day^{-1}$]')
    ax[0].set_ylabel('$P_{LS}$')

    # period domain (lograithmic)
    ax[1].plot(periods, Pf_LS, color='k')
    ax[1].hlines(thresh, pmin, pmax, ls = 'dotted', color='darkred')
    ax[1].set_xlim(pmin, pmax)
    ax[1].set_xlabel('Period [day]')
    ax[1].set_ylabel('$P_{LS}$')
    ax[1].set_xscale('log')
    
    if peaks:
        Ps, Zs = peaks
        ax[0].scatter(1/Ps, Zs,marker='+',color='r')
        ax[1].scatter(Ps, Zs,marker='+',color='r')
        
    plt.tight_layout()


# -----------------------------------------------------------------------------
# Period-candidate review helpers
# -----------------------------------------------------------------------------

def build_period_candidate_bank(
    source_id: Any,
    *,
    base_period: float,
    gaia_period: float | None = None,
    rrfit_periods: Sequence[float] = (),
    ls_periods: Sequence[float] = (),
    manual_periods: Sequence[float] = (),
    include_half_double: bool = True,
    log_tolerance: float = 1e-4,
) -> pd.DataFrame:
    """Collect and deduplicate named period candidates for one source."""

    records: list[tuple[str, float]] = [("base", base_period)]
    if gaia_period is not None:
        records.append(("gaia", gaia_period))
    records.extend((f"rrfit_{i + 1}", value) for i, value in enumerate(rrfit_periods))
    records.extend((f"ls_{i + 1}", value) for i, value in enumerate(ls_periods))
    records.extend((f"manual_{i + 1}", value) for i, value in enumerate(manual_periods))
    if include_half_double:
        records.extend([
            ("base_half", 0.5 * float(base_period)),
            ("base_double", 2.0 * float(base_period)),
        ])
    combined: list[dict[str, Any]] = []
    for label, value in records:
        try:
            period = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(period) or period <= 0:
            continue
        match = next((
            row for row in combined
            if abs(np.log10(period / row["period"])) <= log_tolerance
        ), None)
        if match is None:
            combined.append({
                "ID": str(source_id), "candidate": label,
                "period": period, "sources": label,
            })
        else:
            match["sources"] += ";" + label
    out = pd.DataFrame(combined)
    if len(out):
        out["ratio_to_base"] = out["period"] / float(base_period)
        out.insert(1, "candidate_id", [f"P{i + 1}" for i in range(len(out))])
    return out


def blocked_time_cv_period_score(
    epoch_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    period: float,
    *,
    order: int = 3,
    n_splits: int = 5,
) -> dict[str, float | int]:
    """Score a period with low-order fits held out in contiguous time blocks."""

    t, mag, emag, bands = [np.asarray(value) for value in epoch_data]
    bands = bands.astype(str)
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be positive")
    residuals: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    output: dict[str, float | int] = {}
    folds_used = 0
    for band in np.unique(bands):
        mask = (
            (bands == band) & np.isfinite(t) & np.isfinite(mag)
            & np.isfinite(emag) & (emag > 0))
        tb, yb, eb = t[mask], mag[mask], emag[mask]
        if len(tb) < max(8, 2 * int(order) + 3):
            output[f"cv_rmse_{band}"] = np.nan
            continue
        sorter = np.argsort(tb)
        tb, yb, eb = tb[sorter], yb[sorter], eb[sorter]
        band_residuals, band_weights = [], []
        for test_index in np.array_split(
            np.arange(len(tb)), min(int(n_splits), len(tb))
        ):
            if not len(test_index):
                continue
            train_mask = np.ones(len(tb), dtype=bool)
            train_mask[test_index] = False
            supported_order = min(
                int(order), int((train_mask.sum() - 1) // 2))
            if supported_order < 1:
                continue
            harmonic = 1 + np.arange(supported_order)

            def design(values):
                phase = (
                    2.0 * np.pi * (values / float(period))[:, None] * harmonic)
                return np.column_stack([
                    np.ones(len(values)), np.cos(phase), np.sin(phase)])

            x_train = design(tb[train_mask])
            weight_train = 1.0 / np.maximum(eb[train_mask], 1e-3) ** 2
            root_weight = np.sqrt(weight_train)
            solution = np.linalg.lstsq(
                x_train * root_weight[:, None],
                yb[train_mask] * root_weight, rcond=None)[0]
            prediction = design(tb[test_index]) @ solution
            band_residuals.append(yb[test_index] - prediction)
            band_weights.append(1.0 / np.maximum(eb[test_index], 1e-3) ** 2)
            folds_used += 1
        if band_residuals:
            band_residual = np.concatenate(band_residuals)
            band_weight = np.concatenate(band_weights)
            output[f"cv_rmse_{band}"] = float(
                np.sqrt(np.mean(band_residual**2)))
            residuals.append(band_residual)
            weights.append(band_weight)
        else:
            output[f"cv_rmse_{band}"] = np.nan
    if residuals:
        residual = np.concatenate(residuals)
        weight = np.concatenate(weights)
        output["cv_rmse"] = float(np.sqrt(np.mean(residual**2)))
        output["cv_weighted_rmse"] = float(
            np.sqrt(np.average(residual**2, weights=weight)))
        output["cv_n"] = int(len(residual))
    else:
        output.update({"cv_rmse": np.nan, "cv_weighted_rmse": np.nan, "cv_n": 0})
    output["cv_folds_used"] = int(folds_used)
    return output
