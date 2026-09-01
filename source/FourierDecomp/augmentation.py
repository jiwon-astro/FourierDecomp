"""Gaia-like fixed-period/order light-curve uncertainty augmentation.

This module keeps the expensive period search and nonlinear order-selection
pipeline out of the training loop.  It supports two deliberately separate
uncertainty mechanisms:

``full_design_hc3_realizations``
    Draws Fourier coefficients from the HC3 covariance of an existing light
    curve.  This is conditional on the original observing window.

``synthetic_window_refit``
    Samples epochs from a supplied phase mask, adds noise in first-harmonic
    units, and performs a fixed-period/order weighted linear refit.  Repeating
    this operation captures the effect of a changed Gaia-like window on the
    fitted morphology without repeating a periodogram or nonlinear fit.

All returned ML shapes use the same convention as ``MoDNet.io``: harmonic
coefficients are divided by A1 and the magnitude minimum (maximum light) is
rolled to phase bin zero.  Every phase-resolved companion array is rolled by
the same amount.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from . import params
from .LSQ import cs_matrix
from .uncertainty import conditional_fourier_covariance


@dataclass(frozen=True)
class SyntheticRefitResult:
    """One fixed-period/order Gaia-like reconstruction."""

    shape: np.ndarray
    phase_mask: np.ndarray
    curve_sigma: np.ndarray
    phase: np.ndarray
    beta: np.ndarray
    covariance: np.ndarray
    peak_shift_bins: int
    requested_order: int
    fitted_order: int
    n_epoch: int
    noise_over_a1: float
    condition_number: float
    max_leverage: float
    dof: int
    covariance_kind: str


def stable_source_seed(base_seed: int, source_id: Any) -> int:
    """Return a process-independent deterministic seed for one source."""

    digest = hashlib.blake2b(
        str(source_id).encode("utf-8"), digest_size=8
    ).digest()
    source_seed = int.from_bytes(digest, "little", signed=False)
    return int((int(base_seed) + source_seed) % (2**32))


def parse_phase_mask(value: Any, n_grid: int) -> np.ndarray:
    """Parse a catalog phase-mask value into a clipped float array."""

    if isinstance(value, str):
        parsed = np.fromstring(
            value.strip().replace("[", " ").replace("]", " ").replace(",", " "),
            sep=" ",
            dtype=float,
        )
    else:
        parsed = np.asarray(value, dtype=float).reshape(-1)
    if parsed.size != int(n_grid):
        raise ValueError(
            f"phase mask has {parsed.size} bins; expected {int(n_grid)}"
        )
    parsed = np.nan_to_num(parsed, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(parsed, 0.0, 1.0)


def support_capped_order(
    n_epoch: int,
    requested_order: int,
    *,
    min_residual_dof: int = 3,
    minimum_order: int = 1,
) -> int:
    """Largest fixed Fourier order leaving the requested residual dof.

    A single-band cosine/sine design has ``1 + 2*M`` parameters.  The returned
    order never exceeds ``requested_order`` and is zero when even
    ``minimum_order`` is unsupported.
    """

    n_epoch = int(n_epoch)
    requested_order = int(requested_order)
    min_residual_dof = int(min_residual_dof)
    maximum = (n_epoch - min_residual_dof - 1) // 2
    fitted = min(requested_order, maximum)
    return int(fitted) if fitted >= int(minimum_order) else 0


def _stable_conditional_fit(
    t: np.ndarray,
    mag: np.ndarray,
    emag: np.ndarray,
    *,
    P: float,
    E: float,
    initial_order: int,
    err_floor: float,
    robust: bool,
    maximum_condition_number: float,
    maximum_leverage: float,
) -> tuple[dict[str, Any], int]:
    """Back off Fourier order until HC3 geometry is numerically supported."""

    last_reason = "no fit attempted"
    for order in range(int(initial_order), 0, -1):
        try:
            fit = conditional_fourier_covariance(
                t,
                mag,
                emag,
                P=P,
                E=E,
                M_fit=order,
                err_floor=err_floor,
                robust=robust,
            )
        except Exception as exc:
            last_reason = f"order={order}:{type(exc).__name__}:{exc}"
            continue
        condition = float(fit["condition_number"])
        leverage = float(fit["max_leverage"])
        if (
            np.isfinite(condition)
            and condition <= float(maximum_condition_number)
            and np.isfinite(leverage)
            and leverage <= float(maximum_leverage)
        ):
            return fit, order
        last_reason = (
            f"order={order}:condition={condition:.6g}:leverage={leverage:.6g}"
        )
    raise ValueError(f"no numerically supported HC3 order ({last_reason})")


def interleaved_coefficients_from_record(
    record: Mapping[str, Any],
    M_fit: int | None = None,
    *,
    normalize_a1: bool = True,
) -> np.ndarray:
    """Return ``alpha1,beta1,...`` from catalog A/Q coefficients."""

    if M_fit is None:
        M_fit = int(record["M_fit"])
    M_fit = int(M_fit)
    amplitude = np.asarray(
        [record[f"A{order}"] for order in range(1, M_fit + 1)],
        dtype=float,
    )
    phase = np.asarray(
        [record[f"Q{order}"] for order in range(1, M_fit + 1)],
        dtype=float,
    )
    if not np.all(np.isfinite(np.r_[amplitude, phase])):
        raise ValueError("record contains non-finite Fourier coefficients")
    if normalize_a1:
        if amplitude.size == 0 or not np.isfinite(amplitude[0]) or amplitude[0] <= 0:
            raise ValueError("A1 must be positive for ML-shape normalization")
        amplitude = amplitude / amplitude[0]
    coefficients = np.empty(2 * M_fit, dtype=float)
    coefficients[0::2] = amplitude * np.cos(phase)
    coefficients[1::2] = amplitude * np.sin(phase)
    return coefficients


def _shape_and_shift_from_beta(
    beta: np.ndarray,
    phase_grid: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Evaluate one WLS coefficient vector in the MoDNet shape convention."""

    beta = np.asarray(beta, dtype=float).reshape(-1)
    phase_grid = np.asarray(phase_grid, dtype=float).reshape(-1)
    if beta.size < 3 or (beta.size - 1) % 2:
        raise ValueError("beta must contain intercept and interleaved harmonics")
    M_fit = (beta.size - 1) // 2
    coefficients = beta[1:].copy()
    a1 = float(np.hypot(coefficients[0], coefficients[1]))
    if not np.isfinite(a1) or a1 <= 1e-12:
        raise ValueError("first harmonic is unsupported in the fitted realization")
    coefficients /= a1
    design = cs_matrix(phase_grid, 1.0, 0.0, M_fit)[:, 1:]
    shape = design @ coefficients
    shift = int(np.argmin(shape))
    return np.roll(shape, -shift), shift


def peak_align_phase_arrays(
    reference_curve: Sequence[float],
    *arrays: Sequence[float],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], int]:
    """Roll a magnitude curve and companion phase arrays to maximum light."""

    reference = np.asarray(reference_curve, dtype=float).reshape(-1)
    if reference.size == 0 or not np.all(np.isfinite(reference)):
        raise ValueError("reference_curve must be finite and non-empty")
    shift = int(np.argmin(reference))
    aligned: list[np.ndarray] = []
    for value in arrays:
        array = np.asarray(value).reshape(-1)
        if array.size != reference.size:
            raise ValueError("all phase arrays must match reference_curve")
        aligned.append(np.roll(array, -shift))
    return np.roll(reference, -shift), tuple(aligned), shift


def sample_phases_from_mask(
    phase_mask: Sequence[float],
    n_epoch: int,
    *,
    random_state: int | np.random.Generator = 0,
) -> np.ndarray:
    """Generate approximate epoch phases consistent with a binned mask.

    Occupied bins receive one epoch first when possible; remaining epochs are
    sampled uniformly from occupied bins.  A uniform within-bin jitter avoids
    singular designs caused by placing every epoch at a bin center.
    """

    mask = np.asarray(phase_mask, dtype=float).reshape(-1)
    occupied = np.flatnonzero(np.isfinite(mask) & (mask > 0))
    if occupied.size == 0:
        raise ValueError("phase_mask contains no occupied bins")
    n_epoch = int(n_epoch)
    if n_epoch < 1:
        raise ValueError("n_epoch must be positive")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    if n_epoch >= occupied.size:
        bins = np.r_[
            rng.permutation(occupied),
            rng.choice(occupied, size=n_epoch - occupied.size, replace=True),
        ]
    else:
        bins = rng.choice(occupied, size=n_epoch, replace=False)
    phases = (bins + rng.random(n_epoch)) / mask.size
    return np.sort(np.mod(phases, 1.0))


def _psd_parameter_draws(
    fit: Mapping[str, Any],
    n_draws: int,
    random_state: int | np.random.Generator,
) -> np.ndarray:
    covariance = np.asarray(fit["covariance"], dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    covariance_psd = (
        eigenvector * np.maximum(eigenvalue, 0.0)
    ) @ eigenvector.T
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    return rng.multivariate_normal(
        np.asarray(fit["beta"], dtype=float), covariance_psd, size=int(n_draws)
    )


def aligned_ml_shape_draws(
    conditional_fit: Mapping[str, Any],
    *,
    phase_grid: Sequence[float] | None = None,
    n_draws: int = 32,
    random_state: int | np.random.Generator = 0,
) -> np.ndarray:
    """Draw peak-aligned A1-normalized ML shapes from a coefficient covariance."""

    if phase_grid is None:
        phase_grid = np.linspace(0.0, 1.0, 50, endpoint=False)
    phase_grid = np.asarray(phase_grid, dtype=float)
    draws = _psd_parameter_draws(conditional_fit, n_draws, random_state)
    output = np.full((len(draws), len(phase_grid)), np.nan, dtype=float)
    for index, beta in enumerate(draws):
        try:
            output[index], _ = _shape_and_shift_from_beta(beta, phase_grid)
        except ValueError:
            continue
    return output


def _curve_sigma_from_fit(
    fit: Mapping[str, Any],
    phase_grid: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if int(n_draws) <= 1:
        return np.full(len(phase_grid), np.nan, dtype=float)
    draws = aligned_ml_shape_draws(
        fit, phase_grid=phase_grid, n_draws=int(n_draws), random_state=rng
    )
    valid = np.all(np.isfinite(draws), axis=1)
    if valid.sum() < 2:
        return np.full(len(phase_grid), np.nan, dtype=float)
    return np.std(draws[valid], axis=0, ddof=1)


def synthetic_window_refit(
    nominal_record: Mapping[str, Any],
    phase_mask: Sequence[float],
    n_epoch: int,
    noise_over_a1: float,
    *,
    n_grid: int = 50,
    requested_order: int | None = None,
    min_residual_dof: int = 3,
    heteroskedastic_log_scatter: float = 0.15,
    hc3_summary_draws: int = 0,
    random_state: int | np.random.Generator = 0,
    robust: bool = True,
    err_floor: float | None = None,
    maximum_condition_number: float = 1e8,
    maximum_leverage: float = 0.995,
) -> SyntheticRefitResult:
    """Forward-simulate and refit one Gaia-like normalized light curve.

    The simulated data are already one Monte Carlo realization of the
    measurement process.  Therefore ``shape`` is the fitted mean curve, not a
    second HC3 draw.  ``hc3_summary_draws`` only controls the optional
    phase-resolved uncertainty summary returned in ``curve_sigma``.
    """

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    phase_mask = parse_phase_mask(phase_mask, n_grid)
    n_epoch = int(n_epoch)
    noise_over_a1 = float(noise_over_a1)
    if not np.isfinite(noise_over_a1) or noise_over_a1 <= 0:
        raise ValueError("noise_over_a1 must be finite and positive")
    if requested_order is None:
        requested_order = int(nominal_record["M_fit"])
    simulated_order = support_capped_order(
        n_epoch,
        requested_order,
        min_residual_dof=min_residual_dof,
        minimum_order=1,
    )
    if simulated_order < 1:
        raise ValueError("phase realization has insufficient support for order 1")

    phases = sample_phases_from_mask(
        phase_mask, n_epoch, random_state=rng
    )
    true_coefficients = interleaved_coefficients_from_record(
        nominal_record, simulated_order, normalize_a1=True
    )
    truth_design = cs_matrix(phases, 1.0, 0.0, simulated_order)
    beta_truth = np.r_[0.0, true_coefficients]
    noiseless = truth_design @ beta_truth

    log_scatter = max(float(heteroskedastic_log_scatter), 0.0)
    if log_scatter:
        epoch_error = noise_over_a1 * np.exp(
            rng.normal(-0.5 * log_scatter**2, log_scatter, size=n_epoch)
        )
    else:
        epoch_error = np.full(n_epoch, noise_over_a1, dtype=float)
    observed = noiseless + rng.normal(0.0, epoch_error)
    if err_floor is None:
        # The simulation is in A1-normalized units, unlike the package's
        # magnitude-space default.  Avoid imposing a 0.01-mag floor in this
        # coordinate system unless the caller explicitly requests one.
        err_floor = max(np.finfo(float).eps, noise_over_a1 * 1e-3)
    fit, fitted_order = _stable_conditional_fit(
        phases,
        observed,
        epoch_error,
        P=1.0,
        E=0.0,
        initial_order=simulated_order,
        err_floor=err_floor,
        robust=robust,
        maximum_condition_number=maximum_condition_number,
        maximum_leverage=maximum_leverage,
    )
    phase_grid = np.linspace(0.0, 1.0, int(n_grid), endpoint=False)
    shape, shift = _shape_and_shift_from_beta(fit["beta"], phase_grid)
    aligned_mask = np.roll(phase_mask, -shift)
    curve_sigma = _curve_sigma_from_fit(
        fit, phase_grid, int(hc3_summary_draws), rng
    )
    return SyntheticRefitResult(
        shape=np.asarray(shape, dtype=float),
        phase_mask=np.asarray(aligned_mask, dtype=float),
        curve_sigma=np.asarray(curve_sigma, dtype=float),
        phase=phase_grid,
        beta=np.asarray(fit["beta"], dtype=float),
        covariance=np.asarray(fit["covariance"], dtype=float),
        peak_shift_bins=int(shift),
        requested_order=int(requested_order),
        fitted_order=int(fitted_order),
        n_epoch=int(n_epoch),
        noise_over_a1=float(noise_over_a1),
        condition_number=float(fit["condition_number"]),
        max_leverage=float(fit["max_leverage"]),
        dof=int(fit["dof"]),
        covariance_kind=str(fit["covariance_kind"]),
    )


def raw_window_refit(
    t: Sequence[float],
    mag: Sequence[float],
    emag: Sequence[float],
    nominal_record: Mapping[str, Any],
    phase_mask: Sequence[float],
    n_epoch: int,
    *,
    amplitude_key: str = "amp_I",
    additional_noise_over_a1: float = 0.0,
    n_grid: int = 50,
    requested_order: int | None = None,
    min_residual_dof: int = 3,
    hc3_summary_draws: int = 0,
    random_state: int | np.random.Generator = 0,
    robust: bool = True,
    maximum_condition_number: float = 1e8,
    maximum_leverage: float = 0.995,
) -> SyntheticRefitResult:
    """Subsample a real single-band light curve and run a fixed P/M refit.

    Raw photometry is used only while generating the cache.  Magnitudes and
    errors are converted to first-harmonic units using ``amp_band * A1``.
    Optional additional target noise is added in those normalized units.
    """

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    t = np.asarray(t, dtype=float)
    mag = np.asarray(mag, dtype=float)
    emag = np.asarray(emag, dtype=float)
    good = np.isfinite(t) & np.isfinite(mag) & np.isfinite(emag) & (emag >= 0)
    t, mag, emag = t[good], mag[good], emag[good]
    if len(t) < 3:
        raise ValueError("raw light curve contains too few valid epochs")
    P = float(nominal_record["P"])
    E = float(nominal_record["E"])
    A1 = float(nominal_record["A1"])
    band_amplitude = float(nominal_record[amplitude_key])
    scale = abs(A1 * band_amplitude)
    if not np.isfinite(P) or P <= 0 or not np.isfinite(E):
        raise ValueError("nominal P/E is invalid")
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("amp_band*A1 must be positive")

    phase_mask = parse_phase_mask(phase_mask, n_grid)
    phase = np.mod((t - E) / P, 1.0)
    bins = np.floor(phase * int(n_grid)).astype(int) % int(n_grid)
    eligible = np.flatnonzero(phase_mask[bins] > 0)
    n_epoch = min(int(n_epoch), len(eligible))
    if requested_order is None:
        requested_order = int(nominal_record["M_fit"])
    fitted_order = support_capped_order(
        n_epoch,
        requested_order,
        min_residual_dof=min_residual_dof,
        minimum_order=1,
    )
    if fitted_order < 1:
        raise ValueError("raw window has insufficient supported epochs")
    selected = rng.choice(eligible, size=n_epoch, replace=False)
    selected = selected[np.argsort(t[selected])]
    normalized_mag = mag[selected] / scale
    normalized_error = emag[selected] / scale
    extra = max(float(additional_noise_over_a1), 0.0)
    if extra:
        normalized_mag = normalized_mag + rng.normal(0.0, extra, size=n_epoch)
        normalized_error = np.sqrt(normalized_error**2 + extra**2)
    normalized_error = np.maximum(normalized_error, np.finfo(float).eps)

    fit, fitted_order = _stable_conditional_fit(
        t[selected],
        normalized_mag,
        normalized_error,
        P=P,
        E=E,
        initial_order=fitted_order,
        err_floor=np.finfo(float).eps,
        robust=robust,
        maximum_condition_number=maximum_condition_number,
        maximum_leverage=maximum_leverage,
    )
    phase_grid = np.linspace(0.0, 1.0, int(n_grid), endpoint=False)
    shape, shift = _shape_and_shift_from_beta(fit["beta"], phase_grid)
    aligned_mask = np.roll(phase_mask, -shift)
    curve_sigma = _curve_sigma_from_fit(
        fit, phase_grid, int(hc3_summary_draws), rng
    )
    effective_noise = float(np.sqrt(np.median(normalized_error**2)))
    return SyntheticRefitResult(
        shape=np.asarray(shape, dtype=float),
        phase_mask=np.asarray(aligned_mask, dtype=float),
        curve_sigma=np.asarray(curve_sigma, dtype=float),
        phase=phase_grid,
        beta=np.asarray(fit["beta"], dtype=float),
        covariance=np.asarray(fit["covariance"], dtype=float),
        peak_shift_bins=int(shift),
        requested_order=int(requested_order),
        fitted_order=int(fitted_order),
        n_epoch=int(n_epoch),
        noise_over_a1=effective_noise,
        condition_number=float(fit["condition_number"]),
        max_leverage=float(fit["max_leverage"]),
        dof=int(fit["dof"]),
        covariance_kind=str(fit["covariance_kind"]),
    )


def full_design_hc3_realizations(
    t: Sequence[float],
    mag: Sequence[float],
    emag: Sequence[float],
    *,
    P: float,
    E: float,
    M_fit: int,
    n_draws: int = 32,
    n_grid: int = 50,
    phase_mask: Sequence[float] | None = None,
    random_state: int | np.random.Generator = 0,
    robust: bool = True,
    err_floor: float | None = None,
) -> dict[str, Any]:
    """HC3 shape draws conditional on the original observing design.

    This intentionally does *not* claim to represent a changed phase window.
    It is the control arm for the window-refit augmentation.
    """

    fit = conditional_fourier_covariance(
        t,
        mag,
        emag,
        P=P,
        E=E,
        M_fit=int(M_fit),
        err_floor=params.ERR_FLOOR if err_floor is None else err_floor,
        robust=robust,
    )
    phase = np.linspace(0.0, 1.0, int(n_grid), endpoint=False)
    parameter_draws = _psd_parameter_draws(
        fit, int(n_draws), random_state
    )
    draws = np.full((int(n_draws), int(n_grid)), np.nan, dtype=float)
    shifts = np.full(int(n_draws), -1, dtype=int)
    for index, beta in enumerate(parameter_draws):
        try:
            draws[index], shifts[index] = _shape_and_shift_from_beta(beta, phase)
        except ValueError:
            continue
    result: dict[str, Any] = {
        "phase": phase,
        "shape_draws": draws,
        "shape_mean": np.nanmean(draws, axis=0),
        "shape_sigma": np.nanstd(draws, axis=0, ddof=1),
        "conditional_fit": fit,
        "window_changed": False,
    }
    if phase_mask is not None:
        mask = parse_phase_mask(phase_mask, n_grid)
        # Each draw is independently peak-aligned.  Return the matching mask
        # for every draw rather than pretending a single roll applies to all.
        masks = np.full((int(n_draws), int(n_grid)), np.nan, dtype=float)
        for index, shift in enumerate(shifts):
            if shift >= 0:
                masks[index] = np.roll(mask, -int(shift))
        result["phase_mask_draws"] = masks
    return result
