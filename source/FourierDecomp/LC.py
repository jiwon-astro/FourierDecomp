import numpy as np
import matplotlib.pyplot as plt
from . import params
from .IO import phot_names, epoch_arrays, get_data_config
from .uncertainty import conditional_curve_uncertainty

def compute_phase(t, P):
    return (t / P) % 1.0

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
def expand_light_curve(t, flux, flux_err,  period, phase_range = (0,1)):
    original_phase = compute_phase(t, period)
    
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
                           conditional_fits=None, savepath=None):
    """Plot fixed-period/order HC3 confidence bands for fitted mean curves.

    This follows :func:`plot_lc`, but independently refits the linear Fourier
    coefficients in each displayed band at fixed ``P``, ``E`` and ``M_fit``.
    The shaded regions are 68% and 95% confidence intervals for the fitted
    mean curve.  They are not photometric prediction intervals and do not
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
            phase_range=(0.0, float(phase_max)))
        ax.errorbar(
            phase_obs, mag_obs, yerr=emag_obs, ls='None', color=color,
            marker=marker, ms=3, lw=0.7, alpha=0.75, zorder=3,
            label=f'{band} epochs')

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
        ax.fill_between(
            phase_plot, repeat_curve(band_ci['q025']),
            repeat_curve(band_ci['q975']), color='tab:blue', alpha=0.12,
            linewidth=0, label='95% conditional CI', zorder=1)
        ax.fill_between(
            phase_plot, repeat_curve(band_ci['q16']),
            repeat_curve(band_ci['q84']), color='tab:blue', alpha=0.28,
            linewidth=0, label='68% conditional CI', zorder=2)
        ax.plot(
            phase_plot, repeat_curve(band_ci['nominal']), color='black',
            lw=1.4, label='conditional fit', zorder=4)

        finite_bounds = np.r_[band_ci['q025'], band_ci['q975'], mag_band]
        finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
        if finite_bounds.size:
            ymin, ymax = np.min(finite_bounds), np.max(finite_bounds)
            margin = max(0.08 * (ymax - ymin), 0.03)
            ax.set_ylim(ymax + margin, ymin - margin)

        fit = band_ci['conditional_fit']
        ax.text(
            0.01, 0.05, f'P={float(P):.6f} d',
            transform=ax.transAxes, ha='left', va='bottom')
        ax.text(
            0.01, 0.92, f'{band} ({len(t_band)} epochs)',
            transform=ax.transAxes, ha='left', va='top')
        ax.text(
            0.99, 0.05,
            f"{fit['covariance_kind']}  max h={fit['max_leverage']:.2f}  "
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
