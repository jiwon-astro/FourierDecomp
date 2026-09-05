from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
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
        nrows, ncols, figsize=(4.9 * ncols, 3.4 * nrows),
        squeeze=False, constrained_layout=True)
    for i, record in fitted.iterrows():
        period = float(record["P"])
        epoch = float(record["E"])
        order = int(record["M_fit"])
        a = np.asarray([
            record[f"A{k}"] for k in range(1, order + 1)], dtype=float)
        q = np.asarray([
            record[f"Q{k}"] for k in range(1, order + 1)], dtype=float)
        # Evaluate exactly one complete Fourier cycle, then repeat it.  H()
        # wraps its phase argument in-place, so passing a 0--2 grid directly
        # would reset the x coordinates at phase 1 and draw a folded-back line.
        phase_unit = np.linspace(0.0, 1.0, 401, endpoint=True)
        template_unit = H(
            (a, q), phase_unit.copy(), M_fit=order, coef_mode="AQ")
        phase_curve = np.concatenate([
            phase_unit + cycle for cycle in range(phase_cycles)])
        template_curve = np.tile(template_unit, phase_cycles)
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
                    phase_curve, m0 + amp * template_curve,
                    color="#E69F00", lw=1.8)
            ax.invert_yaxis()
            ax.grid(alpha=0.15)
            ax.set_xlim(0.0, float(phase_cycles))
            ax.set_xlabel("phase")
            ax.set_ylabel(f"{band} [mag]")
            if i == 0:
                ax.set_title(band.upper())
        label = record.get("candidate_id", f"P{i + 1}")
        axes[i, 0].text(
            0.025, 0.035,
            f"{label}:  P={period:.8g} d   |   M={order}\n"
            f"rms/sig={record.get('rms_ratio_max', np.nan):.3f}   |   "
            f"time-CV={record.get('cv_rmse', np.nan):.3f} mag",
            transform=axes[i, 0].transAxes, fontsize=9.5,
            ha="left", va="bottom", linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.42", "facecolor": "white",
                "alpha": 0.88, "edgecolor": "0.72", "linewidth": 0.8,
            })
    fig.suptitle(f"Gaia source {source_id}: period-candidate fixed-FD review")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def plot_rrfit_source_review(
    source_id,
    summary,
    templates,
    *,
    ls_data=None,
    mode: str = "gaia",
    phase_cycles: int = 2,
    max_period_families: int | None = None,
    output_path: str | Path | None = None,
):
    """Plot saved RRFit template solutions without running Fourier decomposition.

    Each row is one clustered period family.  For each band, the lowest
    within-band-pair relative-chi-square RRFit result that contains that band
    supplies its own period, epoch, amplitude, mean, and template index.
    """

    from . import RRFit, decomposition

    cfg = get_data_config(mode)
    if ls_data is None:
        ls_data = decomposition.ls_data
    sid_native = int(source_id) if str(source_id).isdigit() else source_id
    t, mag, emag, bands = epoch_arrays(
        ls_data, sid_native, mode=mode, monitor=False)
    bands = np.asarray(bands).astype(str)
    solutions = RRFit.build_rrfit_solution_families(summary)
    if solutions.empty:
        raise ValueError(f"No valid RRFit solution for source {source_id}")
    if isinstance(templates, (str, Path)):
        templates = RRFit.load_rrfit_templates(templates)

    family_order = solutions[[
        "period_family_index", "solution_id", "period_family", "n_bandpairs",
        "family_score",
    ]].drop_duplicates("period_family_index")
    family_order["solution_rank"] = family_order["solution_id"].str.extract(
        r"(\d+)", expand=False).astype(int)
    family_order = family_order.sort_values("solution_rank")
    if max_period_families is not None:
        family_order = family_order.head(int(max_period_families))
    nrows = len(family_order)
    ncols = len(cfg.filters)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.1 * ncols, 3.5 * nrows),
        squeeze=False, constrained_layout=True)
    phase_unit = np.linspace(0.0, 1.0, 501, endpoint=True)
    phase_curve = np.concatenate([
        phase_unit + cycle for cycle in range(phase_cycles)])

    for panel, family in enumerate(family_order.itertuples(index=False)):
        group = solutions.loc[
            solutions["period_family_index"].eq(family.period_family_index)]
        used = []
        for column, band_value in enumerate(cfg.filters):
            band = str(band_value)
            ax = axes[panel, column]
            band_rows = group.loc[group["bandpair"].astype(str).map(
                lambda value: band in value.split("+"))].sort_values(
                    ["chi2_relative", "chi2"])
            if band_rows.empty:
                fold_period = float(family.period_family)
                fold_epoch = float(group.iloc[0]["epoch"])
                model_row = None
            else:
                model_row = band_rows.iloc[0]
                fold_period = float(model_row["period"])
                fold_epoch = float(model_row["epoch"])
            mask = bands == band
            phase = ((t[mask] - fold_epoch) / fold_period) % 1.0
            phase = np.concatenate([
                phase + cycle for cycle in range(phase_cycles)])
            ax.errorbar(
                phase, np.tile(mag[mask], phase_cycles),
                yerr=np.tile(emag[mask], phase_cycles),
                fmt=cfg.lc_markers[column], ms=3.2,
                color=cfg.lc_colors[column], ecolor=cfg.lc_colors[column],
                alpha=0.72, elinewidth=0.55, capsize=0, zorder=2)
            if model_row is not None:
                pair = str(model_row["bandpair"]).split("+")
                pair_index = pair.index(band)
                amplitude = float(model_row[f"amp_{pair_index + 1}"])
                mean = float(model_row[f"mean_{pair_index + 1}"])
                template_index = int(model_row["template_index"])
                template = templates.get(template_index)
                if template is None:
                    raise KeyError(
                        f"Template {template_index} is absent from templates.dat")
                shape = RRFit.evaluate_rrfit_template(template, phase_unit)
                ax.plot(
                    phase_curve, mean + amplitude * np.tile(shape, phase_cycles),
                    color="#E69F00", lw=2.0, zorder=3)
                used.append(
                    f"{band}:{model_row['bandpair']} P={fold_period:.7g} "
                    f"T={template_index} chi2x={model_row['chi2_relative']:.3f}")
            ax.invert_yaxis()
            ax.set_xlim(0.0, float(phase_cycles))
            ax.grid(alpha=0.15)
            ax.set_xlabel("phase")
            ax.set_ylabel(f"{band} [mag]")
            if panel == 0:
                ax.set_title(band.upper())
        axes[panel, 0].text(
            0.025, 0.035,
            f"{family.solution_id}: median P={family.period_family:.8g} d   |   "
            f"pairs={family.n_bandpairs}\n" + "\n".join(used),
            transform=axes[panel, 0].transAxes, fontsize=8.6,
            ha="left", va="bottom", linespacing=1.20,
            bbox={
                "boxstyle": "round,pad=0.40", "facecolor": "white",
                "alpha": 0.88, "edgecolor": "0.72", "linewidth": 0.8,
            })
    fig.suptitle(
        f"Gaia source {source_id}: saved RRFit period/template solutions (no FD)")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig, solutions


def write_rrfit_review_pdfs(
    review_index,
    templates,
    output_dir,
    *,
    ls_data=None,
    mode="gaia",
    phase_cycles=2,
    max_period_families=8,
    overwrite=False,
    verbose=True,
):
    """Write resumable one-source RRFit review PDFs and a solution index.

    No Fourier decomposition is called.  One PDF per source is intentional:
    completed files are restart boundaries and a multi-thousand-source review
    does not have to rebuild one monolithic PDF after interruption.
    """

    from . import RRFit

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = review_index.copy()
    frame = frame.loc[frame["rrfit_ready"].astype(bool)].copy()
    iterator = frame.itertuples(index=False)
    if verbose:
        iterator = tqdm(iterator, total=len(frame), desc="RRFit review PDFs")
    pdf_rows = []
    solution_rows = []
    for record in iterator:
        sid = str(record.ID)
        period_bin = str(record.period_bin)
        safe_bin = (
            period_bin.replace(" ", "").replace("<", "lt")
            .replace(">=", "ge").replace(".", "p"))
        pdf_path = output_dir / safe_bin / f"{sid}.pdf"
        status = "existing"
        error = ""
        try:
            summary = RRFit.load_rrfit_summary(record.rrfit_summary)
            solutions = RRFit.build_rrfit_solution_families(summary)
            families = solutions[[
                "solution_id", "period_family", "period_min", "period_max",
                "n_bandpairs", "family_score",
            ]].drop_duplicates("solution_id")
            for family in families.itertuples(index=False):
                solution_rows.append({
                    "source_id": sid,
                    "period_bin": period_bin,
                    "solution_id": family.solution_id,
                    "period_family": float(family.period_family),
                    "period_min": float(family.period_min),
                    "period_max": float(family.period_max),
                    "n_bandpairs": int(family.n_bandpairs),
                    "family_score": float(family.family_score),
                    "pdf_path": str(pdf_path),
                })
            if overwrite or not pdf_path.exists():
                figure, _ = plot_rrfit_source_review(
                    sid, summary, templates, ls_data=ls_data, mode=mode,
                    phase_cycles=phase_cycles,
                    max_period_families=max_period_families,
                    output_path=pdf_path)
                plt.close(figure)
                status = "written"
        except Exception as exc:
            status = "failed"
            error = repr(exc)
            plt.close("all")
        pdf_rows.append({
            "source_id": sid,
            "period_bin": period_bin,
            "review_priority": getattr(record, "review_priority", np.nan),
            "pdf_path": str(pdf_path),
            "status": status,
            "error": error,
        })
    pdf_index = pd.DataFrame(pdf_rows)
    solution_index = pd.DataFrame(solution_rows)
    pdf_index.to_csv(output_dir / "rrfit_review_pdf_index.csv", index=False)
    solution_index.to_csv(
        output_dir / "rrfit_review_solution_index.csv", index=False)
    return pdf_index, solution_index
