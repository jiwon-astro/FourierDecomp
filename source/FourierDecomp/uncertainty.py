"""Fit-stability diagnostics and robust Fourier uncertainty estimation.

This module deliberately keeps the existing decomposition entry point intact.
It provides small helpers used during fitting and an optional bootstrap wrapper
for server-side uncertainty production.
"""

from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np

from . import params
from .LSQ import H, ab_to_AQ, cs_matrix, unpack_theta


# -----------------------------------------------------------------------------
# Adaptive phase-gap / steep-branch stability
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptivePenaltyWeights:
    """Penalty weights and diagnostics frozen before nonlinear optimization."""

    lam_spike: float
    lam_h: float
    lam_slope: float
    gap_severity: float
    sparse_severity: float
    steepness: float
    steep_strength: float
    steep_support: float
    supported_steepness: float

    def to_dict(self):
        return asdict(self)


def adaptive_initial_lambda(lam0, phase_gaps, n_epoch, M_fit,
                            lam_min=None, lam_max=None):
    """Return a continuous phase-support-aware L1 initialization weight.

    Large phase gaps, high order, and a small epoch count require stronger
    shrinkage.  This replaces the previous cubic gap heuristic, whose weight
    changed too abruptly around the hand-tuned gap scale.
    """
    if lam_min is None: lam_min = params.lam_min
    if lam_max is None: lam_max = params.lam_max

    gaps = np.asarray(phase_gaps, dtype=float)
    gaps = gaps[np.isfinite(gaps)]
    gmax = float(np.max(gaps)) if gaps.size else 1.0

    gap_ratio = max(gmax, 0.05) / max(params.GAP_REFERENCE, 1e-6)
    gap_factor = np.clip(gap_ratio**params.GAP_POWER, 0.35, 4.0)
    order_factor = np.clip((max(M_fit, 1) / 5.0)**1.5, 0.5, 4.0)
    sample_factor = np.clip(params.N_EPOCH_REFERENCE / max(n_epoch, 10),
                            0.5, 3.0)

    lam = lam0 * gap_factor * order_factor * sample_factor
    return float(np.clip(lam, lam_min, lam_max))


def _dilate_circular_mask(mask, radius):
    """Dilate a boolean phase mask without breaking the phase wrap."""
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    for offset in range(1, int(radius) + 1):
        out |= np.roll(mask, offset) | np.roll(mask, -offset)
    return out


def adaptive_penalty_weights(theta, args, M_fit, activated_bands, phase_gaps,
                             coef_mode=None, lam_spike=None, lam_h=None,
                             lam_slope=None):
    """Freeze adaptive global-penalty weights from the initial fit.

    The adaptation follows two separate ideas:

    1. Poor phase support increases spike/high-order suppression.
    2. A steep branch is preserved only when observations actually cover it;
       in that case harmonic smoothing is relaxed and the coherent branch
       residual term is activated.

    Computing these diagnostics once avoids repeating coverage work inside
    every objective evaluation.
    """
    if lam_spike is None: lam_spike = params.lam_spike
    if lam_h is None: lam_h = params.lam_h
    if lam_slope is None: lam_slope = params.lam_sl

    t, _, _, bmask = args
    activated_bands = np.asarray(activated_bands, dtype=int)
    n_bands = len(activated_bands)
    _, _, c1, c2, P, E = unpack_theta(
        theta, n_bands, M_fit=M_fit, include_amp=True, coef_mode=coef_mode)

    # -------------------------------------------------------------------------
    # Phase support severity
    # -------------------------------------------------------------------------
    gaps = np.asarray(phase_gaps, dtype=float)[activated_bands]
    gaps = gaps[np.isfinite(gaps)]
    gmax = float(np.max(gaps)) if gaps.size else 1.0
    gap_ratio = max(gmax, 0.05) / max(params.GAP_REFERENCE, 1e-6)
    gap_severity = float(np.clip((gap_ratio - 0.5) / 2.0, 0.0, 1.0))

    counts = np.asarray([np.sum(bmask[ib]) for ib in activated_bands], dtype=float)
    n_effective = float(np.max(counts)) if counts.size else 0.0
    sparse_severity = float(np.clip(
        (params.N_EPOCH_REFERENCE - n_effective) /
        max(params.N_EPOCH_REFERENCE - 5.0, 1.0), 0.0, 1.0))
    instability = max(gap_severity, sparse_severity)

    # -------------------------------------------------------------------------
    # Model steepness and observed support for the steep branch
    # -------------------------------------------------------------------------
    n_grid = max(int(params.STEEP_GRID_SIZE), 8 * int(M_fit))
    phi_grid = np.arange(n_grid, dtype=float) / n_grid
    shape = H((c1, c2), phi_grid, M_fit=M_fit, coef_mode=coef_mode)
    derivative = 0.5 * n_grid * (np.roll(shape, -1) - np.roll(shape, 1))
    abs_derivative = np.abs(derivative)
    steepness = float(np.percentile(abs_derivative, 95))
    steep_strength = float(np.clip(
        (steepness - params.STEEP_SLOPE_REFERENCE) /
        max(params.STEEP_SLOPE_SCALE, 1e-6), 0.0, 1.0))

    steep_cut = max(params.STEEP_SLOPE_REFERENCE,
                    float(np.percentile(abs_derivative, 80)))
    steep_mask = _dilate_circular_mask(
        abs_derivative >= steep_cut, params.STEEP_SUPPORT_RADIUS)

    support_values = []
    for ib in activated_bands:
        mask = bmask[ib]
        if not np.any(mask):
            continue
        phase = ((t[mask] - E) / P) % 1.0
        phase_idx = np.floor(phase * n_grid).astype(int) % n_grid
        occupied = np.zeros(n_grid, dtype=bool)
        occupied[phase_idx] = True
        denom = max(int(np.sum(steep_mask)), 1)
        support_values.append(np.sum(occupied & steep_mask) / denom)

    steep_support = float(np.mean(support_values)) if support_values else 0.0
    supported_steepness = steep_strength * steep_support * (1.0 - gap_severity)

    # -------------------------------------------------------------------------
    # Frozen effective weights
    # -------------------------------------------------------------------------
    harmonic_relief = 1.0 - params.STEEP_HARMONIC_RELIEF * supported_steepness
    lam_h_eff = lam_h * (1.0 + 1.5 * instability) * harmonic_relief
    lam_spike_eff = lam_spike * (1.0 + 2.0 * instability)
    lam_slope_eff = lam_slope * (0.25 + 1.75 * steep_strength * steep_support)
    lam_slope_eff *= (1.0 - 0.5 * gap_severity)

    return AdaptivePenaltyWeights(
        lam_spike=float(np.clip(lam_spike_eff, 0.1 * lam_spike, 4.0 * lam_spike)),
        lam_h=float(np.clip(lam_h_eff, 0.1 * lam_h, 4.0 * lam_h)),
        lam_slope=float(np.clip(lam_slope_eff, 0.0, 3.0 * lam_slope)),
        gap_severity=gap_severity,
        sparse_severity=sparse_severity,
        steepness=steepness,
        steep_strength=steep_strength,
        steep_support=steep_support,
        supported_steepness=float(supported_steepness),
    )


# -----------------------------------------------------------------------------
# Fourier invariants and circular statistics
# -----------------------------------------------------------------------------

INVARIANT_NAMES = (
    "R21", "R31",
    "sin_phi21", "cos_phi21",
    "sin_phi31", "cos_phi31",
)


def wrap_angle(angle):
    return np.mod(angle, 2.0 * np.pi)


def circular_difference(angle, reference):
    """Shortest signed angular difference in [-pi, pi)."""
    return np.angle(np.exp(1j * (np.asarray(angle) - reference)))


def fourier_invariants(A, Q, min_amplitude=1e-10,
                       min_relative_amplitude=1e-3):
    """Return low-order amplitude ratios and epoch-invariant phase terms."""
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    out = {"R21": np.nan, "R31": np.nan,
           "phi21": np.nan, "phi31": np.nan}

    if A.size < 2 or Q.size < 2 or not np.isfinite(A[0]) or A[0] <= min_amplitude:
        return out

    phase_threshold = max(min_amplitude, min_relative_amplitude * A[0])
    out["R21"] = float(A[1] / A[0])
    if np.isfinite(A[1]) and A[1] > phase_threshold:
        out["phi21"] = float(wrap_angle(Q[1] - 2.0 * Q[0]))
    if A.size >= 3 and Q.size >= 3:
        out["R31"] = float(A[2] / A[0])
        if np.isfinite(A[2]) and A[2] > phase_threshold:
            out["phi31"] = float(wrap_angle(Q[2] - 3.0 * Q[0]))
    return out


def robust_scale(values, axis=0):
    """Gaussian-consistent robust scale, 1.4826 * MAD."""
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(values - center), axis=axis)


def circular_summary(angles):
    """Circular center, robust scale, and central interval around the center."""
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return {"mean": np.nan, "robust_sigma": np.nan,
                "q16": np.nan, "q84": np.nan, "resultant_length": np.nan}

    mean = float(wrap_angle(np.angle(np.mean(np.exp(1j * angles)))))
    delta = circular_difference(angles, mean)
    q16, q84 = np.percentile(delta, [16, 84])
    return {
        "mean": mean,
        "robust_sigma": float(robust_scale(delta)),
        "q16": float(wrap_angle(mean + q16)),
        "q84": float(wrap_angle(mean + q84)),
        "resultant_length": float(np.abs(np.mean(np.exp(1j * angles)))),
    }


def robust_covariance(values, clip_sigma=5.0):
    """MAD-winsorized covariance for compact, mostly unimodal samples."""
    values = np.asarray(values, dtype=float)
    good = np.all(np.isfinite(values), axis=1)
    values = values[good]
    if values.ndim != 2 or len(values) < values.shape[1] + 2:
        n_dim = values.shape[1] if values.ndim == 2 else 0
        return np.full((n_dim, n_dim), np.nan)

    center = np.median(values, axis=0)
    scale = robust_scale(values, axis=0)
    fallback = np.std(values, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    clipped = np.clip(values, center - clip_sigma * scale,
                      center + clip_sigma * scale)
    return np.cov(clipped, rowvar=False, ddof=1)


# -----------------------------------------------------------------------------
# Fixed-period/order conditional covariance
# -----------------------------------------------------------------------------

def conditional_fourier_covariance(t, mag, emag, P, E, M_fit,
                                   err_floor=None, robust=True):
    """Weighted alpha/beta covariance conditional on fixed P, E, and M_fit.

    HC3 covariance is used by default.  This is a fast diagnostic and a lower
    bound: it does not include period aliases, order switching, clipping, or
    regularization selection.
    """
    if err_floor is None: err_floor = params.ERR_FLOOR
    t = np.asarray(t, dtype=float)
    mag = np.asarray(mag, dtype=float)
    emag = np.asarray(emag, dtype=float)
    good = np.isfinite(t) & np.isfinite(mag) & np.isfinite(emag) & (emag >= 0)
    t, mag, emag = t[good], mag[good], emag[good]

    X = cs_matrix(t, P, E, M_fit)
    if len(t) <= X.shape[1]:
        raise ValueError("Insufficient epochs for the requested Fourier order")

    sigma = np.maximum(emag, err_floor)
    Xw = X / sigma[:, None]
    yw = mag / sigma
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    residual_w = yw - Xw @ beta
    bread = np.linalg.pinv(Xw.T @ Xw, hermitian=True)

    leverage = np.sum((Xw @ bread) * Xw, axis=1)
    if robust:
        adjusted = residual_w / np.maximum(1.0 - leverage, 1e-6)
        meat = Xw.T @ ((adjusted**2)[:, None] * Xw)
        covariance = bread @ meat @ bread
        covariance_kind = "HC3"
    else:
        dof = max(len(t) - X.shape[1], 1)
        covariance = (np.sum(residual_w**2) / dof) * bread
        covariance_kind = "classical"

    return {
        "beta": beta,
        "covariance": covariance,
        "standard_error": np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        "condition_number": float(np.linalg.cond(Xw)),
        "max_leverage": float(np.max(leverage)),
        "dof": int(len(t) - X.shape[1]),
        "covariance_kind": covariance_kind,
    }


def conditional_invariant_uncertainty(t, mag, emag, P, E, M_fit,
                                      n_draws=4000, random_state=0,
                                      err_floor=None, robust=True):
    """Propagate conditional alpha/beta covariance into order-2/3 invariants."""
    fit = conditional_fourier_covariance(
        t, mag, emag, P, E, M_fit, err_floor=err_floor, robust=robust)

    rng = np.random.default_rng(random_state)
    cov = 0.5 * (fit["covariance"] + fit["covariance"].T)
    eigval, eigvec = np.linalg.eigh(cov)
    cov_psd = (eigvec * np.maximum(eigval, 0.0)) @ eigvec.T
    draws = rng.multivariate_normal(fit["beta"], cov_psd, size=n_draws)

    invariants = []
    for draw in draws:
        A, Q = ab_to_AQ(draw[1::2], draw[2::2])
        inv = fourier_invariants(A, Q)
        invariants.append(inv)

    summary = summarize_invariant_replicates(invariants)
    summary["conditional_fit"] = fit
    return summary


def conditional_curve_uncertainty(t, mag, emag, P, E, M_fit,
                                  phase_grid=None, n_draws=4000,
                                  random_state=0, err_floor=None,
                                  robust=True, return_draws=False,
                                  conditional_fit=None):
    """Propagate conditional coefficient covariance into a light-curve band.

    The returned intervals are conditional on fixed ``P``, ``E`` and
    ``M_fit``.  They describe uncertainty in the fitted mean curve, not a
    prediction interval for a new noisy epoch.  Period aliases, order
    switching, clipping and regularization selection require the full
    bootstrap instead.
    """
    if conditional_fit is None:
        fit = conditional_fourier_covariance(
            t, mag, emag, P, E, M_fit,
            err_floor=err_floor, robust=robust)
    else:
        fit = conditional_fit
        expected = 1 + 2 * int(M_fit)
        if np.asarray(fit["beta"]).shape != (expected,):
            raise ValueError("conditional_fit beta does not match M_fit")
        if np.asarray(fit["covariance"]).shape != (expected, expected):
            raise ValueError("conditional_fit covariance does not match M_fit")

    if phase_grid is None:
        phase_grid = np.linspace(0.0, 1.0, 400, endpoint=False)
    phase_grid = np.asarray(phase_grid, dtype=float)
    if phase_grid.ndim != 1 or phase_grid.size < 2:
        raise ValueError("phase_grid must be a one-dimensional array")

    covariance = 0.5 * (fit["covariance"] + fit["covariance"].T)
    eigval, eigvec = np.linalg.eigh(covariance)
    covariance_psd = (
        eigvec * np.maximum(eigval, 0.0)
    ) @ eigvec.T

    rng = np.random.default_rng(random_state)
    beta_draws = rng.multivariate_normal(
        fit["beta"], covariance_psd, size=int(n_draws))
    t_grid = float(E) + float(P) * phase_grid
    design_grid = cs_matrix(t_grid, P, E, M_fit)
    curve_draws = beta_draws @ design_grid.T
    q025, q16, q50, q84, q975 = np.nanpercentile(
        curve_draws, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)

    result = {
        "phase": phase_grid,
        "nominal": design_grid @ fit["beta"],
        "median": q50,
        "q025": q025,
        "q16": q16,
        "q84": q84,
        "q975": q975,
        "conditional_fit": fit,
        "n_draws": int(n_draws),
    }
    if return_draws:
        result["curve_draws"] = curve_draws
        result["beta_draws"] = beta_draws
    return result


# -----------------------------------------------------------------------------
# Full-pipeline epoch bootstrap
# -----------------------------------------------------------------------------

def resample_epoch_data(epoch_data, rng, group_ids=None):
    """Resample epochs with replacement, preserving bands or supplied groups."""
    t, mag, emag, bands = [np.asarray(x) for x in epoch_data]
    if not (len(t) == len(mag) == len(emag) == len(bands)):
        raise ValueError("epoch_data arrays must have identical lengths")

    sampled = []
    if group_ids is None:
        # No transit identifier is available: preserve each band's sample size.
        for band in np.unique(bands):
            idx = np.flatnonzero(bands == band)
            sampled.append(rng.choice(idx, size=len(idx), replace=True))
    else:
        group_ids = np.asarray(group_ids)
        if len(group_ids) != len(t):
            raise ValueError("group_ids must match epoch_data length")
        unique_groups = np.unique(group_ids)
        selected_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        for group in selected_groups:
            sampled.append(np.flatnonzero(group_ids == group))

    idx = np.concatenate(sampled) if sampled else np.array([], dtype=int)
    return t[idx], mag[idx], emag[idx], bands[idx]


def _numeric_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"median": np.nan, "robust_sigma": np.nan,
                "q16": np.nan, "q84": np.nan}
    q16, q84 = np.percentile(values, [16, 84])
    return {"median": float(np.median(values)),
            "robust_sigma": float(robust_scale(values)),
            "q16": float(q16), "q84": float(q84)}


def summarize_invariant_replicates(records):
    """Summarize dict records containing R21/R31/phi21/phi31."""
    summary = {}
    for name in ("R21", "R31"):
        summary[name] = _numeric_summary([row.get(name, np.nan) for row in records])
    for name in ("phi21", "phi31"):
        summary[name] = circular_summary([row.get(name, np.nan) for row in records])

    matrix = []
    for row in records:
        r21, r31 = row.get("R21", np.nan), row.get("R31", np.nan)
        p21, p31 = row.get("phi21", np.nan), row.get("phi31", np.nan)
        matrix.append([r21, r31, np.sin(p21), np.cos(p21),
                       np.sin(p31), np.cos(p31)])
    matrix = np.asarray(matrix, dtype=float).reshape(-1, len(INVARIANT_NAMES))
    summary["covariance_names"] = INVARIANT_NAMES
    summary["robust_covariance"] = robust_covariance(matrix)
    summary["phi21_valid_fraction"] = float(np.mean(np.isfinite(matrix[:, 2]))) if len(matrix) else np.nan
    summary["phi31_valid_fraction"] = float(np.mean(np.isfinite(matrix[:, 4]))) if len(matrix) else np.nan
    return summary


def bootstrap_fourier_decomp(sid, mode=None, n_boot=100, random_state=0,
                             epoch_data=None, group_ids=None, decomp_kwargs=None,
                             nominal_row=None, return_replicates=False,
                             progress: Optional[Callable] = None):
    """Run full-pipeline epoch bootstrap for one source.

    Parallelize this function across sources on a CPU server.  Keep the loop
    over bootstrap replicates serial to avoid nested process pools and excessive
    memory use.
    """
    from . import decomposition as decomp
    from .IO import build_fd_header, epoch_arrays, get_data_config

    if mode is None: mode = get_data_config().mode
    if decomp_kwargs is None: decomp_kwargs = {}
    decomp_kwargs = dict(decomp_kwargs)
    decomp_kwargs["verbose"] = False

    if epoch_data is None:
        if mode == "gaia" and group_ids is None:
            loaded = epoch_arrays(
                decomp.ls_data, sid, mode=mode, return_groups=True)
            epoch_data, group_ids = loaded[:4], loaded[4]
        else:
            epoch_data = epoch_arrays(decomp.ls_data, sid, mode=mode)
    epoch_data = tuple(np.asarray(x) for x in epoch_data)

    header = build_fd_header(mode)
    if nominal_row is None:
        nominal_row = decomp.fourier_decomp(
            sid, mode=mode, epoch_data=epoch_data, **decomp_kwargs)
    nominal = dict(zip(header, nominal_row))

    root_seed = np.random.SeedSequence(random_state)
    child_seeds = root_seed.spawn(int(n_boot))
    replicates, failures = [], []

    for i, child_seed in enumerate(child_seeds):
        rng = np.random.default_rng(child_seed)
        sample = resample_epoch_data(epoch_data, rng, group_ids=group_ids)
        try:
            row = decomp.fourier_decomp(
                sid, mode=mode, epoch_data=sample, **decomp_kwargs)
            record = dict(zip(header, row))
            m_fit = int(record["M_fit"])
            A = np.asarray([record[f"A{k}"] for k in range(1, m_fit + 1)], dtype=float)
            Q = np.asarray([record[f"Q{k}"] for k in range(1, m_fit + 1)], dtype=float)
            record.update(fourier_invariants(A, Q))
            replicates.append(record)
        except Exception as exc:
            failures.append({"replicate": i, "error": repr(exc)})

        if progress is not None:
            progress(i + 1, n_boot)

    invariant_summary = summarize_invariant_replicates(replicates)
    periods = np.asarray([row["P"] for row in replicates], dtype=float)
    orders = np.asarray([row["M_fit"] for row in replicates], dtype=int)
    nominal_period = float(nominal["P"])

    if len(periods):
        period_agreement = np.isclose(periods, nominal_period, rtol=1e-3, atol=0.0)
        harmonic_agreement = (
            np.isclose(periods, 0.5 * nominal_period, rtol=1e-3, atol=0.0) |
            np.isclose(periods, 2.0 * nominal_period, rtol=1e-3, atol=0.0))
    else:
        period_agreement = harmonic_agreement = np.array([], dtype=bool)

    unique_order, order_count = np.unique(orders, return_counts=True)
    order_probability = {
        int(order): float(count / max(len(orders), 1))
        for order, count in zip(unique_order, order_count)
    }
    order_p = np.asarray(list(order_probability.values()), dtype=float)
    order_entropy = float(-np.sum(order_p * np.log(order_p))) if order_p.size else np.nan

    harmonic_summary = {}
    for k in range(1, 4):
        values = [row.get(f"A{k}", np.nan) for row in replicates]
        stats = _numeric_summary(values)
        sigma = stats["robust_sigma"]
        stats["robust_snr"] = (
            float(stats["median"] / sigma)
            if np.isfinite(sigma) and sigma > 0 else np.nan)
        harmonic_summary[f"A{k}"] = stats

    result = {
        "sid": sid,
        "mode": mode,
        "n_boot": int(n_boot),
        "n_success": int(len(replicates)),
        "n_failure": int(len(failures)),
        "failure_fraction": float(len(failures) / max(n_boot, 1)),
        "period": _numeric_summary(periods),
        "period_nominal_fraction": float(np.mean(period_agreement)) if len(periods) else np.nan,
        "period_half_double_fraction": float(np.mean(harmonic_agreement)) if len(periods) else np.nan,
        "m_fit_probability": order_probability,
        "m_fit_entropy": order_entropy,
        "p_m_ge_2": float(np.mean(orders >= 2)) if len(orders) else np.nan,
        "p_m_ge_3": float(np.mean(orders >= 3)) if len(orders) else np.nan,
        "harmonics": harmonic_summary,
        "invariants": invariant_summary,
        "failures": failures,
    }
    if return_replicates:
        result["replicates"] = replicates
    return result
