from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from . import params
from .IO import phot_names, epoch_arrays, get_data_config
from .uncertainty import (
    conditional_curve_uncertainty, conditional_shared_curve_uncertainty,
)

def compute_phase(t, P, E=0.0):
    return ((t - E) / P) % 1.0

def phase_gap_score(t, P, M_fit=None): 
    if M_fit is None: M_fit = params.M_MAX
    if len(t) < (M_fit + 2): return True # M_fit 사용
    phi = compute_phase(t, P)
    phi_sorted = np.sort(phi)
    gaps = np.diff(np.r_[phi_sorted, phi_sorted[0] + 1.0])
    gmax = gaps.max()
    # g90  = np.percentile(gaps, 90)
    return gmax
    #return gaps.max() > threshold

# Expand phased light curve (arbitary phase range)
def expand_light_curve(t, flux, flux_err,  period, phase_range = (0,1), E=0.0):
    original_phase = compute_phase(t, period, E=E)
    
    #phase range [0,1) -> [n,n+1)
    phase_i, phase_f = phase_range
    n_i, n_f = int(np.floor(phase_i)), int(np.ceil(phase_f))-1 # determine duplication range
    
    # expanding data
    phase_list, flux_list, ferr_list = [], [], []
    for n in range(n_i, n_f+1):
        shifted_phase = original_phase + n
        mask = (shifted_phase >= phase_i) & (shifted_phase <= phase_f)
        if np.any(mask):
            phase_list.append(shifted_phase[mask])
            flux_list.append(flux[mask])
            if np.any(flux_err): ferr_list.append(flux_err[mask])
            
    if phase_list:
        phase = np.concatenate(phase_list)
        flux = np.concatenate(flux_list)
        
        # sorting refer to phase
        order = np.argsort(phase)
        phase, flux = phase[order], flux[order]
        if np.any(flux_err): flux_err = np.concatenate(ferr_list)[order]
            
    else:
        phase, flux, flux_err = np.array([]), np.array([]), np.array([])
    
    return phase, flux, flux_err

# set proper axis limit
def set_ylim(y,yerr,amp=1):
    mean_y = np.mean(y)
    std_y = np.std(y)
    
    ymin = min(y) - 1.5 * np.max(yerr)
    ymax = max(y) + 1.5 * np.max(yerr)

    #3-sigma outliers
    sigma = 2
    if ymax > mean_y + sigma * std_y: ymax = mean_y + sigma * std_y
    if ymin < mean_y - sigma * std_y: ymin = mean_y - sigma * std_y
        
    yscale=0.25*abs(amp)
    return ymax+yscale, ymin-yscale

# plot lig ht curve
def plot_lc(sid, P0, mode='ogle', selected_filters = ['I'], phase_max = 2, scale = 1.0):
    cfg = get_data_config(mode)
    filters = cfg.filters
    prefixs = cfg.prefixs
    lc_colors = cfg.lc_colors
    lc_markers = cfg.lc_markers
    
    n_bands = len(selected_filters)
    t, mag, emag, bands = epoch_arrays(ls_data, sid, mode=mode, monitor=False)
    #t, mag, emag, bands = [data[key].values for key in [*phot_names,'band']]
    bmask = [(bands==band) for band in selected_filters]
    
    fig, ax = plt.subplots(n_bands,1,figsize=(10,3*n_bands),dpi=300)
    if n_bands == 1: ax = [ax]
    ax[0].set_title(f'{sid}',loc='left',fontsize = 18)
    
    for i, band in enumerate(selected_filters):
        mask = bmask[i]; ib = prefixs[filters == band][0] # from filter catalog
        t_ft, mag_ft, emag_ft = t[mask], mag[mask], emag[mask]
        if len(t_ft)==0: continue
        ext_phase, ext_mag, ext_emag = expand_light_curve(t_ft, mag_ft, emag_ft, P0, phase_range = (0, phase_max))
        ax[i].errorbar(ext_phase, ext_mag, yerr = ext_emag, ls='None', 
                       color=lc_colors[ib], marker=lc_markers[ib], lw = 1, zorder = 0)

        y_lb, y_ub = set_ylim(mag_ft, emag_ft, amp = scale)

        ax[i].text(0,y_lb-(y_lb-y_ub)*0.1,s=f'P={P0:.4f}days')
        ax[i].text(0.05, y_ub+(y_lb-y_ub)*0.2, s=f'{band} ({len(t_ft)} epochs)')
        ax[i].set_ylim(y_lb, y_ub)
        ax[i].set_xlim(-0.1,phase_max+0.1)
        ax[i].set_ylabel('Magnitude\n[mag]')
        if i==n_bands-1: ax[i].set_xlabel('Phase')
    
    return fig, ax


# -----------------------------------------------------------------------------
# Conditional HC3 confidence-band light curve
# -----------------------------------------------------------------------------

def plot_lc_conditional_ci(sid, P, E, M_fit, mode='ogle',
                           selected_filters=None, phase_max=2,
                           n_grid=400, n_draws=4000, random_state=0,
                           robust=True, epoch_data=None,
                           conditional_fits=None, nominal_record=None,
                           shared_conditional_fit=None, savepath=None):
    """Plot fixed-period/order HC3 confidence bands for fitted mean curves.

    With ``nominal_record``, all selected bands use one joint shared morphology
    and band-specific mean/amplitude parameters, matching the decomposition
    model.  A single-band call without ``nominal_record`` retains the legacy
    conditional linear fit.  The shaded regions are confidence intervals for
    the fitted mean curve, not photometric prediction intervals, and do not
    include period aliases or model/order selection uncertainty.

    Returns
    -------
    fig, axes, diagnostics
        ``diagnostics`` maps each band to the complete output of
        :func:`conditional_curve_uncertainty`.
    """
    cfg = get_data_config(mode)
    filters = np.asarray(cfg.filters).astype(str)

    if epoch_data is None:
        t, mag, emag, bands = epoch_arrays(
            ls_data, sid, mode=mode, monitor=False)
    else:
        t, mag, emag, bands = [np.asarray(value) for value in epoch_data]
    bands = np.asarray(bands).astype(str)

    if selected_filters is None:
        selected_filters = [
            str(cfg.filters[index]) for index in cfg.activated_bands
            if np.any(bands == str(cfg.filters[index]))
        ]
    else:
        selected_filters = [str(band) for band in selected_filters]
    if not selected_filters:
        raise ValueError("No requested band contains observations")

    phase_grid = np.linspace(0.0, 1.0, int(n_grid), endpoint=False)
    n_bands = len(selected_filters)
    fig, axes = plt.subplots(
        n_bands, 1, figsize=(10, 3 * n_bands), dpi=300,
        sharex=True, squeeze=False)
    axes = axes[:, 0]
    axes[0].set_title(f'{sid}', loc='left', fontsize=18)
    diagnostics = {}

    shared_ci = None
    if nominal_record is not None:
        shared_ci = conditional_shared_curve_uncertainty(
            t, mag, emag, bands, nominal_record, selected_filters,
            P=P, E=E, M_fit=M_fit,
            reference_band=selected_filters[0], phase_grid=phase_grid,
            n_draws=n_draws, random_state=random_state, robust=robust,
            conditional_fit=shared_conditional_fit)
        diagnostics["shared_fit"] = shared_ci["conditional_fit"]
        diagnostics["morphology"] = shared_ci["morphology"]
    elif len(selected_filters) > 1:
        raise ValueError(
            "nominal_record is required for a multi-band shared-morphology CI")

    for panel, (ax, band) in enumerate(zip(axes, selected_filters)):
        mask = bands == band
        if not np.any(mask):
            ax.set_visible(False)
            continue

        filter_match = np.flatnonzero(filters == band)
        color_index = int(filter_match[0]) if filter_match.size else panel
        color = cfg.lc_colors[color_index]
        marker = cfg.lc_markers[color_index]
        t_band, mag_band, emag_band = t[mask], mag[mask], emag[mask]

        phase_obs, mag_obs, emag_obs = expand_light_curve(
            t_band, mag_band, emag_band, P,
            phase_range=(0.0, float(phase_max)), E=E)
        ax.errorbar(
            phase_obs, mag_obs, yerr=emag_obs, ls='None', color=color,
            marker=marker, ms=3, lw=0.7, alpha=0.75, zorder=3,
            label=f'{band} epochs')

        if shared_ci is not None:
            band_ci = shared_ci["bands"][band]
        else:
            precomputed_fit = None
            if conditional_fits is not None:
                precomputed_fit = conditional_fits.get(band)
            band_ci = conditional_curve_uncertainty(
                t_band, mag_band, emag_band, P=P, E=E, M_fit=M_fit,
                phase_grid=phase_grid, n_draws=n_draws,
                random_state=int(random_state) + panel,
                robust=robust, return_draws=False,
                conditional_fit=precomputed_fit)
        diagnostics[band] = band_ci

        n_cycles = max(int(np.ceil(float(phase_max))), 1)
        phase_plot = np.concatenate(
            [phase_grid + cycle for cycle in range(n_cycles)])
        keep = phase_plot <= float(phase_max)

        def repeat_curve(values):
            return np.tile(np.asarray(values), n_cycles)[keep]

        phase_plot = phase_plot[keep]
        ci_prefix = 'joint HC3' if shared_ci is not None else 'conditional'
        ax.fill_between(
            phase_plot, repeat_curve(band_ci['q025']),
            repeat_curve(band_ci['q975']), color='tab:blue', alpha=0.12,
            linewidth=0, label=f'95% {ci_prefix} CI', zorder=1)
        ax.fill_between(
            phase_plot, repeat_curve(band_ci['q16']),
            repeat_curve(band_ci['q84']), color='tab:blue', alpha=0.28,
            linewidth=0, label=f'68% {ci_prefix} CI', zorder=2)
        ax.plot(
            phase_plot, repeat_curve(band_ci['nominal']), color='black',
            lw=1.4,
            label=('shared nominal fit' if shared_ci is not None
                   else 'conditional fit'), zorder=4)

        finite_bounds = np.r_[band_ci['q025'], band_ci['q975'], mag_band]
        finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
        if finite_bounds.size:
            ymin, ymax = np.min(finite_bounds), np.max(finite_bounds)
            margin = max(0.08 * (ymax - ymin), 0.03)
            ax.set_ylim(ymax + margin, ymin - margin)

        fit = (shared_ci['conditional_fit'] if shared_ci is not None
               else band_ci['conditional_fit'])
        max_leverage = fit.get('max_leverage_by_band', {}).get(
            band, fit['max_leverage'])
        ax.text(
            0.01, 0.05, f'P={float(P):.6f} d',
            transform=ax.transAxes, ha='left', va='bottom')
        ax.text(
            0.01, 0.92, f'{band} ({len(t_band)} epochs)',
            transform=ax.transAxes, ha='left', va='top')
        ax.text(
            0.99, 0.05,
            f"{fit['covariance_kind']}  max h={max_leverage:.2f}  "
            f"cond={fit['condition_number']:.1e}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
        ax.set_ylabel('Magnitude\n[mag]')
        ax.grid(alpha=0.15)
        ax.legend(loc='best', fontsize=8, ncol=2)

    axes[-1].set_xlabel('Phase')
    axes[-1].set_xlim(0.0, float(phase_max))
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    return fig, axes, diagnostics


def plot_source_period_review(
    source_id,
    candidate_fits: pd.DataFrame,
    *,
    mode: str = "gaia",
    output_path: str | Path | None = None,
    phase_cycles: int = 2,
):
    """Plot raw G/BP/RP folds and fitted curves on candidate panels."""

    from . import decomposition
    from .LSQ import H

    cfg = get_data_config(mode)
    sid_native = int(source_id) if str(source_id).isdigit() else source_id
    t, mag, emag, bands = epoch_arrays(
        decomposition.ls_data, sid_native, mode=mode, monitor=False)
    bands = np.asarray(bands).astype(str)
    fitted = candidate_fits.loc[
        candidate_fits["candidate_fit_status"].ne("failed")].reset_index(drop=True)
    if fitted.empty:
        raise ValueError("No successful candidate fit to plot")

    nrows = len(fitted)
    ncols = len(cfg.filters)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, 3.0 * nrows),
        squeeze=False, constrained_layout=True)
    for i, record in fitted.iterrows():
        period = float(record["P"])
        epoch = float(record["E"])
        order = int(record["M_fit"])
        a = np.asarray([
            record[f"A{k}"] for k in range(1, order + 1)], dtype=float)
        q = np.asarray([
            record[f"Q{k}"] for k in range(1, order + 1)], dtype=float)
        phase_grid = np.linspace(0.0, float(phase_cycles), 700)
        template = H((a, q), phase_grid, M_fit=order, coef_mode="AQ")
        for j, band in enumerate(cfg.filters):
            band = str(band)
            ax = axes[i, j]
            mask = bands == band
            phase = ((t[mask] - epoch) / period) % 1.0
            phase = np.concatenate([
                phase + cycle for cycle in range(phase_cycles)])
            y = np.tile(mag[mask], phase_cycles)
            yerr = np.tile(emag[mask], phase_cycles)
            ax.errorbar(
                phase, y, yerr=yerr, fmt=cfg.lc_markers[j], ms=3.1,
                color=cfg.lc_colors[j], ecolor=cfg.lc_colors[j], alpha=0.72,
                elinewidth=0.55, capsize=0)
            m0 = float(record.get(f"m0_{band}", np.nan))
            amp = float(record.get(f"amp_{band}", np.nan))
            if np.isfinite(m0) and np.isfinite(amp):
                ax.plot(
                    phase_grid, m0 + amp * template,
                    color="#E69F00", lw=1.8)
            ax.invert_yaxis()
            ax.grid(alpha=0.15)
            ax.set_xlabel("phase")
            ax.set_ylabel(f"{band} [mag]")
            if i == 0:
                ax.set_title(band.upper())
        label = record.get("candidate_id", f"P{i + 1}")
        axes[i, 0].text(
            0.02, 0.04,
            f"{label}: P={period:.8g} d, M={order}, "
            f"rms/sig={record.get('rms_ratio_max', np.nan):.3f}, "
            f"time-CV={record.get('cv_rmse', np.nan):.3f} mag",
            transform=axes[i, 0].transAxes, fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"})
    fig.suptitle(f"Gaia source {source_id}: period-candidate fixed-FD review")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig
