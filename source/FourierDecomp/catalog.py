"""Catalog-level Fourier uncertainty and quality helpers.

The nominal Fourier table intentionally stays compact.  This module adds only
the low-order Fourier quantities used by the ML auxiliary branch and their HC3
errors.  Fit diagnostics are written to a separate QA table.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import params
from .IO import epoch_arrays, get_data_config
from .quality import coverage_entropy, gap_max, occupied_fraction
from .uncertainty import (
    conditional_shared_invariant_uncertainty,
    fourier_invariants,
)


# -----------------------------------------------------------------------------
# Minimal ML-facing columns
# -----------------------------------------------------------------------------

FD_ERROR_COLUMNS = (
    "R21", "R31", "phi21", "phi31",
    "R21_err", "R31_err", "phi21_err", "phi31_err",
)

FD_QA_COLUMNS = (
    "ID", "status", "retryable", "stage", "reason",
    "nominal_flag", "rms_ratio_max", "occupied_fraction_min", "gmax_max",
)

_STATUS_RANK = {"ok": 0, "feature_missing": 1, "review": 2, "failed": 3}


def build_fd_error_header(base_header):
    """Return the nominal header plus the compact ML-facing HC3 columns."""
    return list(base_header) + list(FD_ERROR_COLUMNS)


def nan_error_record():
    """Return a complete error record filled with NaN values."""
    return {name: np.nan for name in FD_ERROR_COLUMNS}


def sanitize_reason(reason, max_length=500):
    """Keep QA rows parseable by the whitespace-delimited catalog reader."""
    text = "_".join(str(reason).replace("|", "/").split())
    return text[:max_length] if text else "unspecified"


def stable_source_seed(base_seed, sid):
    """Deterministic per-source seed independent of Python hash randomization."""
    digest = hashlib.blake2b(
        str(sid).encode("utf-8"), digest_size=8
    ).digest()
    sid_seed = int.from_bytes(digest, "little", signed=False)
    return int((int(base_seed) + sid_seed) % (2**32))


# -----------------------------------------------------------------------------
# Nominal invariant values and conditional HC3 errors
# -----------------------------------------------------------------------------

def _nominal_invariants(nominal_record):
    m_fit = int(nominal_record["M_fit"])
    amplitudes = np.asarray(
        [nominal_record[f"A{k}"] for k in range(1, m_fit + 1)],
        dtype=float,
    )
    phases = np.asarray(
        [nominal_record[f"Q{k}"] for k in range(1, m_fit + 1)],
        dtype=float,
    )
    return fourier_invariants(amplitudes, phases)


def _observed_active_filters(bands, mode):
    cfg = get_data_config(mode)
    bands = np.asarray(bands).astype(str)
    active = [str(cfg.filters[index]) for index in cfg.activated_bands]
    return [band for band in active if np.any(bands == band)]


def compute_minimal_hc3_errors(
        sid, nominal_record, mode=None, epoch_data=None, n_draws=4000,
        random_state=0, robust=True):
    """Compute the compact HC3 error fields used by Fourier-aware ML.

    The full parameter covariance remains internal.  The returned catalog
    fields are the nominal R/phi invariants and four propagated uncertainties.
    """
    if mode is None:
        mode = get_data_config().mode
    if epoch_data is None:
        epoch_data = epoch_arrays(
            _decomposition_data(), sid, mode=mode)
    t, mag, emag, bands = [np.asarray(value) for value in epoch_data]

    selected_filters = _observed_active_filters(bands, mode)
    if not selected_filters:
        raise ValueError("No observed active band is available for HC3")

    m_fit = int(nominal_record["M_fit"])
    if m_fit < 1:
        raise ValueError(f"Invalid M_fit={m_fit}")

    summary = conditional_shared_invariant_uncertainty(
        t, mag, emag, bands,
        nominal_record=nominal_record,
        selected_filters=selected_filters,
        reference_band=selected_filters[0],
        P=float(nominal_record["P"]),
        E=float(nominal_record["E"]),
        M_fit=m_fit,
        n_draws=int(n_draws),
        random_state=stable_source_seed(random_state, sid),
        robust=bool(robust),
    )
    nominal = _nominal_invariants(nominal_record)
    result = {
        "R21": nominal["R21"],
        "R31": nominal["R31"],
        "phi21": nominal["phi21"],
        "phi31": nominal["phi31"],
        "R21_err": summary["R21"]["robust_sigma"],
        "R31_err": summary["R31"]["robust_sigma"],
        "phi21_err": summary["phi21"]["robust_sigma"],
        "phi31_err": summary["phi31"]["robust_sigma"],
    }
    return result


def _decomposition_data():
    # Imported lazily to preserve the package's existing global wiring model.
    from . import decomposition as decomp
    return decomp.ls_data


# -----------------------------------------------------------------------------
# Lightweight fit-quality audit
# -----------------------------------------------------------------------------

def assess_nominal_fit_quality(
        sid, nominal_record, mode=None, epoch_data=None,
        rms_ratio_limit=0.7, min_occupied_fraction=None,
        max_phase_gap=None):
    """Assess active-band fit quality without adding QA columns to the ML table."""
    if mode is None:
        mode = get_data_config().mode
    if epoch_data is None:
        epoch_data = epoch_arrays(
            _decomposition_data(), sid, mode=mode)
    t, mag, emag, bands = [np.asarray(value) for value in epoch_data]
    bands = bands.astype(str)

    cfg = get_data_config(mode)
    active = [str(cfg.filters[index]) for index in cfg.activated_bands]
    reasons = []
    retryable = False
    rms_ratios = []
    occupations = []
    phase_gaps = []

    try:
        period = float(nominal_record["P"])
        epoch = float(nominal_record["E"])
        m_fit = int(nominal_record["M_fit"])
    except Exception as exc:
        return make_qa_record(
            sid, "failed", True, "nominal",
            f"invalid_core_parameter:{exc}", nominal_record)

    if not np.isfinite(period) or period <= 0 or not np.isfinite(epoch) or m_fit < 1:
        reasons.append("invalid_P_E_or_M_fit")
        retryable = True

    for band in active:
        mask = bands == band
        if not np.any(mask):
            reasons.append(f"missing_active_band:{band}")
            retryable = True
            continue

        scatter = float(nominal_record.get(f"sig_{band}", np.nan))
        residual = float(nominal_record.get(f"rms_{band}", np.nan))
        if np.isfinite(scatter) and scatter > 0 and np.isfinite(residual):
            ratio = residual / scatter
            rms_ratios.append(ratio)
            if rms_ratio_limit is not None and ratio > float(rms_ratio_limit):
                reasons.append(f"high_rms_ratio:{band}")
                retryable = True
        else:
            reasons.append(f"invalid_rms_or_scatter:{band}")
            retryable = True

        amplitude = float(nominal_record.get(f"amp_{band}", np.nan))
        if (
            not np.isfinite(amplitude) or
            amplitude < params.Amin or amplitude > params.Amax
        ):
            reasons.append(f"amplitude_boundary:{band}")
            retryable = True

        if np.isfinite(period) and period > 0 and np.isfinite(epoch):
            phase = ((t[mask] - epoch) / period) % 1.0
            occupation = occupied_fraction(phase, n_grid=params.n_grid)
            phase_gap = gap_max(phase)
            occupations.append(occupation)
            phase_gaps.append(phase_gap)
            # Calculate entropy for parity with build_quality_table.  It is not
            # persisted because it is not an ML input or a retry criterion.
            coverage_entropy(phase, n_grid=params.n_grid)
            if (
                min_occupied_fraction is not None and
                occupation < float(min_occupied_fraction)
            ):
                reasons.append(f"low_phase_occupation:{band}")
                retryable = True
            if (
                max_phase_gap is not None and
                phase_gap > float(max_phase_gap)
            ):
                reasons.append(f"large_phase_gap:{band}")
                retryable = True

    if reasons:
        status = "failed" if any(
            reason.startswith(("invalid_P", "missing_active_band"))
            for reason in reasons
        ) else "review"
    else:
        status = "ok"

    return make_qa_record(
        sid=sid,
        status=status,
        retryable=retryable,
        stage="quality",
        reason="|".join(reasons) if reasons else "ok",
        nominal_record=nominal_record,
        rms_ratio_max=max(rms_ratios) if rms_ratios else np.nan,
        occupied_fraction_min=min(occupations) if occupations else np.nan,
        gmax_max=max(phase_gaps) if phase_gaps else np.nan,
    )


def make_qa_record(
        sid, status, retryable, stage, reason, nominal_record=None,
        rms_ratio_max=np.nan, occupied_fraction_min=np.nan,
        gmax_max=np.nan):
    """Create a fixed-width QA row."""
    if status not in _STATUS_RANK:
        raise ValueError(f"Unknown QA status: {status}")
    nominal_flag = np.nan
    if nominal_record is not None:
        nominal_flag = nominal_record.get("flag", np.nan)
    return {
        "ID": str(sid),
        "status": status,
        "retryable": int(bool(retryable)),
        "stage": sanitize_reason(stage),
        "reason": sanitize_reason(reason),
        "nominal_flag": nominal_flag,
        "rms_ratio_max": rms_ratio_max,
        "occupied_fraction_min": occupied_fraction_min,
        "gmax_max": gmax_max,
    }


def merge_qa_records(left, right):
    """Merge two QA records while keeping the more severe status."""
    if left is None:
        return right
    if right is None:
        return left
    severe = (
        left if _STATUS_RANK[left["status"]] >= _STATUS_RANK[right["status"]]
        else right
    )
    merged = dict(severe)
    merged["retryable"] = int(
        bool(left.get("retryable", 0)) or bool(right.get("retryable", 0)))
    reasons = [
        item for item in (left.get("reason"), right.get("reason"))
        if item and item != "ok"
    ]
    merged["reason"] = sanitize_reason("|".join(dict.fromkeys(reasons)) or "ok")
    for name in ("rms_ratio_max", "gmax_max"):
        values = np.asarray([left.get(name, np.nan), right.get(name, np.nan)],
                            dtype=float)
        merged[name] = (
            float(np.nanmax(values)) if np.any(np.isfinite(values)) else np.nan
        )
    values = np.asarray([
        left.get("occupied_fraction_min", np.nan),
        right.get("occupied_fraction_min", np.nan),
    ], dtype=float)
    merged["occupied_fraction_min"] = (
        float(np.nanmin(values)) if np.any(np.isfinite(values)) else np.nan
    )
    return merged


def error_values(record):
    """Return error fields in stable catalog-column order."""
    return [record.get(name, np.nan) for name in FD_ERROR_COLUMNS]


def qa_values(record):
    """Return QA fields in stable catalog-column order."""
    return [record.get(name, np.nan) for name in FD_QA_COLUMNS]


def load_retry_ids(qa_path, statuses=("failed", "review")):
    """Read retryable IDs from a QA table produced by the catalog runners."""
    import pandas as pd

    qa_path = Path(qa_path)
    if not qa_path.exists():
        return []
    frame = pd.read_csv(qa_path, sep=r"\s+", dtype={"ID": str})
    if frame.empty:
        return []
    keep = (
        frame["status"].isin(tuple(statuses)) &
        frame["retryable"].astype(int).eq(1)
    )
    return frame.loc[keep, "ID"].drop_duplicates().tolist()
