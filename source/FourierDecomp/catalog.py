"""Catalog-level Fourier uncertainty and quality helpers.

The nominal Fourier table intentionally stays compact.  This module adds only
the low-order Fourier quantities used by the ML auxiliary branch and their HC3
errors.  Fit diagnostics are written to a separate QA table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

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

FD_REFIT_AUDIT_COLUMNS = (
    "ID", "decision", "selected_source", "reason",
    "base_status", "refit_status", "period_relative_change",
    "base_rms_ratio", "refit_rms_ratio", "base_chi2", "refit_chi2",
)

PERIOD_AUDIT_COLUMNS = (
    "ID", "status", "retryable", "reason", "P_catalog", "P_screen",
    "delta_score", "alternate_delta", "n_candidates", "cycles_observed",
)

REFERENCE_REFIT_AUDIT_COLUMNS = (
    "ID", "status", "retryable", "reason", "P_base", "P_reference",
    "P_refit", "base_relative_difference", "refit_relative_difference",
    "strategy",
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


# -----------------------------------------------------------------------------
# Reference-free period/alias audit for deep-refit selection
# -----------------------------------------------------------------------------

def _fixed_period_robust_score(t, mag, emag, bands, period, selected_filters,
                               order=3):
    """Low-order robust score used only to compare period candidates."""
    total_loss = 0.0
    n_used = 0
    n_parameter = 0
    for band in selected_filters:
        mask = bands == band
        tb = np.asarray(t[mask], dtype=float)
        yb = np.asarray(mag[mask], dtype=float)
        eb = np.asarray(emag[mask], dtype=float)
        finite = np.isfinite(tb) & np.isfinite(yb) & np.isfinite(eb) & (eb > 0)
        tb, yb, eb = tb[finite], yb[finite], eb[finite]
        supported_order = min(int(order), int((len(tb) - 2) // 2))
        if supported_order < 1:
            continue
        harmonics = 1 + np.arange(supported_order)
        phase = 2.0 * np.pi * (tb / float(period))[:, None] * harmonics
        design = np.column_stack(
            [np.ones(len(tb)), np.cos(phase), np.sin(phase)])
        weight = 1.0 / np.maximum(eb, params.ERR_FLOOR) ** 2
        root_weight = np.sqrt(weight)
        solution = np.linalg.lstsq(
            design * root_weight[:, None], yb * root_weight,
            rcond=None)[0]
        residual = yb - design @ solution
        # The scale must not depend on the candidate residual: doing so would
        # reward a bad period for producing a large residual MAD.  A fixed
        # data/error scale keeps the comparison meaningful while Huber loss
        # still limits individual outliers.
        data_mad = 1.4826 * np.median(np.abs(yb - np.median(yb)))
        scale = max(
            float(np.median(eb)), 0.05 * float(data_mad), params.ERR_FLOOR)
        standardized = residual / scale
        # Huber loss prevents a few outliers from selecting an alias period.
        delta = 2.5
        absolute = np.abs(standardized)
        loss = np.where(
            absolute <= delta,
            0.5 * standardized ** 2,
            delta * (absolute - 0.5 * delta),
        )
        total_loss += float(2.0 * np.sum(loss))
        n_used += len(tb)
        n_parameter += design.shape[1]
    if n_used <= n_parameter or n_used == 0:
        return np.inf
    return total_loss + n_parameter * np.log(n_used)


def _unique_periods(periods, log_tolerance=1e-5):
    values = np.asarray(periods, dtype=float)
    values = np.sort(values[np.isfinite(values) & (values > 0)])
    unique = []
    for value in values:
        if not unique or abs(np.log(value / unique[-1])) > log_tolerance:
            unique.append(float(value))
    return np.asarray(unique, dtype=float)


def assess_period_stability(
        sid, nominal_record, mode=None, epoch_data=None, deep_k=15,
        harmonic_depth=4, screen_order=3, better_score_threshold=10.0,
        ambiguity_threshold=2.0, minimum_cycles=2.0):
    """Audit an adopted period without a reference catalog.

    A deeper single-term Lomb--Scargle search proposes candidates.  Explicit
    integer multiples/divisors are then compared with a low-order robust
    Fourier score.  The routine only nominates sources for deep refitting; it
    never changes a catalog period by itself.
    """
    from .period_finder import robust_period_search

    if mode is None:
        mode = get_data_config().mode
    if epoch_data is None:
        epoch_data = epoch_arrays(_decomposition_data(), sid, mode=mode)
    t, mag, emag, bands = [np.asarray(value) for value in epoch_data]
    bands = bands.astype(str)
    cfg = get_data_config(mode)
    active = [str(cfg.filters[index]) for index in cfg.activated_bands]
    observed = [band for band in active if np.count_nonzero(bands == band) >= 2]
    period = float(nominal_record.get("P", np.nan))

    result = {
        "ID": str(sid), "status": "ok", "retryable": 0, "reason": "ok",
        "P_catalog": period, "P_screen": np.nan, "delta_score": np.nan,
        "alternate_delta": np.nan, "n_candidates": 0,
        "cycles_observed": np.nan,
    }
    if not observed:
        result.update({
            "status": "failed", "retryable": 0,
            "reason": "missing_observed_active_band"})
        return result
    use = np.isin(bands, observed)
    baseline = float(np.ptp(t[use]))
    if not np.isfinite(period) or period <= 0 or baseline <= 0:
        result.update({
            "status": "failed", "retryable": 1,
            "reason": "invalid_catalog_period_or_baseline"})
        return result
    result["cycles_observed"] = baseline / period

    try:
        proposed, _ = robust_period_search(
            t, mag, emag, bands, n0=params.n0, K=int(deep_k), snr=params.snr,
            harmonics=1, plot=False, mode=mode)
    except Exception as exc:
        result.update({
            "status": "review", "retryable": 1,
            "reason": sanitize_reason(f"period_screen_failed:{exc}")})
        return result

    source_period_max = min(params.pmax, baseline * (1.0 - 1e-8))
    family = [period, *np.asarray(proposed, dtype=float).tolist()]
    for base_period in list(family):
        for factor in range(2, int(harmonic_depth) + 1):
            family.extend([base_period / factor, base_period * factor])
    candidates = _unique_periods([
        candidate for candidate in family
        if params.pmin <= candidate <= source_period_max
    ])
    scores = np.asarray([
        _fixed_period_robust_score(
            t, mag, emag, bands, candidate, observed, order=screen_order)
        for candidate in candidates
    ])
    finite = np.isfinite(scores)
    candidates, scores = candidates[finite], scores[finite]
    result["n_candidates"] = int(len(candidates))
    if len(candidates) == 0:
        result.update({
            "status": "review", "retryable": 1,
            "reason": "period_screen_no_valid_candidate"})
        return result

    # First optimize locally around the catalog solution.  Otherwise a tiny
    # period refinement can hide a scientifically different half/double alias
    # by becoming the global best candidate itself.
    separation = np.abs(np.log(candidates / period))
    same_family = separation <= np.log(1.02)
    alternate = ~same_family
    if not np.any(same_family):
        same_family[np.argmin(separation)] = True
        alternate = ~same_family
    local_index = np.flatnonzero(same_family)[np.argmin(scores[same_family])]
    local_score = float(scores[local_index])
    catalog_score = _fixed_period_robust_score(
        t, mag, emag, bands, period, observed, order=screen_order)
    result["delta_score"] = float(catalog_score - local_score)

    if np.any(alternate):
        alternate_index = np.flatnonzero(alternate)[np.argmin(scores[alternate])]
        alternate_score = float(scores[alternate_index])
        result["P_screen"] = float(candidates[alternate_index])
        # Positive: the separated candidate is better than the locally
        # refined catalog solution.  Near zero: unresolved alias competition.
        result["alternate_delta"] = float(local_score - alternate_score)
    else:
        result["P_screen"] = float(candidates[local_index])

    better_candidate = (
        np.isfinite(result["alternate_delta"]) and
        result["alternate_delta"] > float(better_score_threshold))
    ambiguous = (
        np.isfinite(result["alternate_delta"]) and
        abs(result["alternate_delta"]) <= float(ambiguity_threshold))
    too_few_cycles = result["cycles_observed"] < float(minimum_cycles)
    near_lower_boundary = period <= 1.05 * float(params.pmin)
    near_upper_boundary = period >= 0.95 * source_period_max

    reasons = []
    if better_candidate:
        reasons.append("period_screen_better_candidate")
    if ambiguous:
        reasons.append("period_screen_ambiguous")
    if too_few_cycles:
        reasons.append("fewer_than_minimum_cycles")
    if near_lower_boundary:
        reasons.append("near_lower_period_boundary")
    if near_upper_boundary:
        reasons.append("near_upper_period_boundary")
    if reasons:
        result.update({
            "status": "review", "retryable": 1,
            "reason": sanitize_reason("|".join(reasons))})
    return result


def period_audit_values(record):
    return [record.get(name, np.nan) for name in PERIOD_AUDIT_COLUMNS]


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


def load_deep_refit_ids(qa_path, period_audit_path=None):
    """Union retryable fit-QA and reference-free period-audit selections."""
    import pandas as pd

    selected = list(load_retry_ids(qa_path))
    if period_audit_path is not None and Path(period_audit_path).exists():
        audit = pd.read_csv(
            period_audit_path, sep=r"\s+", dtype={"ID": str})
        keep = (
            audit["status"].eq("review") &
            audit["retryable"].astype(int).eq(1)
        )
        selected.extend(audit.loc[keep, "ID"].astype(str).tolist())
    return list(dict.fromkeys(str(sid) for sid in selected))


# -----------------------------------------------------------------------------
# OGLE-reference-guided rescue (development/training catalog only)
# -----------------------------------------------------------------------------

def load_ogle_reference_periods(fourier_dir, pattern="*.dat"):
    """Load the public OGLE period column from Fourier catalog files.

    OGLE Fourier tables are headerless; columns 0 and 3 contain source ID and
    period.  Duplicate IDs are accepted only when their periods agree.  The
    returned mapping is intentionally explicit so a guided refit cannot fall
    back silently to blind period search for a missing source.
    """
    import pandas as pd

    fourier_dir = Path(fourier_dir)
    paths = sorted(fourier_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No OGLE Fourier files matched {fourier_dir / pattern}")
    periods = {}
    for path in paths:
        frame = pd.read_csv(
            path, sep=r"\s+", header=None, comment="#", na_values="-")
        if frame.shape[1] < 4:
            raise ValueError(f"OGLE Fourier table has fewer than 4 columns: {path}")
        for sid, period in frame.iloc[:, [0, 3]].itertuples(index=False, name=None):
            sid = str(sid)
            try:
                period = float(period)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(period) or period <= 0:
                continue
            previous = periods.get(sid)
            if previous is not None and not np.isclose(
                    previous, period, rtol=1e-8, atol=0.0):
                raise ValueError(
                    f"Conflicting OGLE reference periods for {sid}: "
                    f"{previous} versus {period}")
            periods[sid] = period
    return periods


def select_reference_period_refit_ids(
        base_catalog, reference_periods, relative_tolerance=1e-3,
        period_column="P"):
    """Select base rows whose adopted period disagrees with OGLE reference."""
    import pandas as pd

    frame = pd.read_csv(
        base_catalog, sep=r"\s+", usecols=["ID", period_column],
        dtype={"ID": str})
    selected = []
    for sid, period in frame[["ID", period_column]].itertuples(
            index=False, name=None):
        reference = reference_periods.get(str(sid))
        if reference is None:
            continue
        try:
            relative = abs(float(period) / float(reference) - 1.0)
        except (TypeError, ValueError, ZeroDivisionError):
            relative = np.inf
        if not np.isfinite(relative) or relative > float(relative_tolerance):
            selected.append(str(sid))
    return selected


def audit_reference_period_refit_catalog(
        refit_catalog, reference_periods, output_path, base_catalog=None,
        relative_tolerance=1e-3, strategy="fixed", overwrite=False):
    """Write merge-compatible provenance/QA for reference-guided refits."""
    import pandas as pd

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; use overwrite=True")
    refit = pd.read_csv(
        refit_catalog, sep=r"\s+", usecols=["ID", "P"], dtype={"ID": str})
    base_map = {}
    if base_catalog is not None and Path(base_catalog).exists():
        base = pd.read_csv(
            base_catalog, sep=r"\s+", usecols=["ID", "P"],
            dtype={"ID": str})
        base_map = dict(zip(base["ID"].astype(str), base["P"]))

    rows = []
    for sid, period in refit[["ID", "P"]].itertuples(index=False, name=None):
        sid = str(sid)
        reference = reference_periods.get(sid)
        base_period = base_map.get(sid, np.nan)
        if reference is None:
            status, retryable, reason = "failed", 0, "reference_period_missing"
            refit_relative = np.nan
        else:
            try:
                refit_relative = abs(float(period) / float(reference) - 1.0)
            except (TypeError, ValueError, ZeroDivisionError):
                refit_relative = np.inf
            if np.isfinite(refit_relative) and (
                    refit_relative <= float(relative_tolerance)):
                status, retryable, reason = "ok", 0, "reference_period_agreement"
            else:
                status, retryable, reason = (
                    "review", 1, "reference_period_disagreement")
        try:
            base_relative = abs(float(base_period) / float(reference) - 1.0)
        except (TypeError, ValueError, ZeroDivisionError):
            base_relative = np.nan
        rows.append({
            "ID": sid, "status": status, "retryable": retryable,
            "reason": reason, "P_base": base_period,
            "P_reference": reference if reference is not None else np.nan,
            "P_refit": period,
            "base_relative_difference": base_relative,
            "refit_relative_difference": refit_relative,
            "strategy": sanitize_reason(strategy),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=REFERENCE_REFIT_AUDIT_COLUMNS).to_csv(
        output_path, sep=" ", index=False, na_rep="nan")
    return {
        "rows": int(len(rows)),
        "ok": int(sum(row["status"] == "ok" for row in rows)),
        "review": int(sum(row["status"] == "review" for row in rows)),
        "failed": int(sum(row["status"] == "failed" for row in rows)),
        "output": str(output_path),
    }


def split_refit_ids_by_support(ids, mode=None, ls_data=None):
    """Separate deep-refit candidates from structurally unsupported sources."""
    if mode is None:
        mode = get_data_config().mode
    if ls_data is None:
        ls_data = _decomposition_data()
    cfg = get_data_config(mode)
    source_map = {str(sid): sid for sid in ls_data.keys()}
    eligible, excluded = [], []
    for sid_key in dict.fromkeys(str(sid) for sid in ids):
        sid = source_map.get(sid_key, sid_key)
        try:
            _, _, _, bands = epoch_arrays(ls_data, sid, mode=mode)
            bands = np.asarray(bands).astype(str)
        except Exception as exc:
            excluded.append({"ID": sid_key, "reason": f"epoch_load_failed:{exc}"})
            continue
        counts = [
            int(np.count_nonzero(bands == str(cfg.filters[index])))
            for index in cfg.activated_bands
        ]
        if any(count < 2 for count in counts):
            excluded.append({
                "ID": sid_key, "reason": "missing_observed_active_band"})
            continue
        n_bands = len(counts)
        supported_order = int(np.floor(
            (sum(counts) - 2 * n_bands - 2 - 1) / 2.0))
        if supported_order < params.M_MIN:
            excluded.append({
                "ID": sid_key,
                "reason": "insufficient_active_epochs_for_M_MIN"})
            continue
        eligible.append(sid)
    return eligible, excluded


# -----------------------------------------------------------------------------
# Audited refit merge
# -----------------------------------------------------------------------------

def _read_qa_status_map(path):
    """Collapse repeated QA rows to the most severe status for each source."""
    import pandas as pd

    if path is None or not Path(path).exists():
        return {}
    frame = pd.read_csv(path, sep=r"\s+", dtype={"ID": str})
    if frame.empty:
        return {}
    result = {}
    for record in frame.to_dict(orient="records"):
        sid = str(record["ID"])
        previous = result.get(sid)
        if previous is None:
            result[sid] = record
            continue
        left_rank = _STATUS_RANK.get(str(previous.get("status")), -1)
        right_rank = _STATUS_RANK.get(str(record.get("status")), -1)
        if right_rank >= left_rank:
            result[sid] = record
    return result


def _catalog_rms_ratio(record, mode):
    cfg = get_data_config(mode)
    values = []
    for index in cfg.activated_bands:
        band = str(cfg.filters[index])
        scatter = float(record.get(f"sig_{band}", np.nan))
        residual = float(record.get(f"rms_{band}", np.nan))
        if np.isfinite(scatter) and scatter > 0 and np.isfinite(residual):
            values.append(residual / scatter)
    return float(max(values)) if values else np.nan


def _valid_error_row(record):
    core = ("P", "E", "M_fit")
    for name in core + FD_ERROR_COLUMNS:
        try:
            if not np.isfinite(float(record.get(name, np.nan))):
                return False
        except (TypeError, ValueError):
            return False
    return float(record["P"]) > 0 and int(float(record["M_fit"])) >= 1


def merge_refit_error_catalogs(
        base_catalog, refit_catalog, output_catalog, mode=None,
        base_qa=None, refit_qa=None, base_period_audit=None,
        refit_period_audit=None, audit_output=None, overwrite=False,
        improvement_fraction=0.02, require_refit_period_audit=True):
    """Merge accepted deep-refit rows into a new compact error catalog.

    The base catalog is never modified.  A recovered missing source is added
    only when its refit and all compact HC3 features are valid.  An existing
    row is replaced only when the deep refit has QA status ``ok`` and either
    resolves a base QA problem or improves the fit metric.  Every decision is
    written to a separate audit table.
    """
    import pandas as pd

    mode = get_data_config(mode).mode
    base_catalog = Path(base_catalog)
    refit_catalog = Path(refit_catalog)
    output_catalog = Path(output_catalog)
    if audit_output is None:
        audit_output = output_catalog.with_name(
            f"{output_catalog.stem}_merge_audit.dat")
    audit_output = Path(audit_output)

    for target in (output_catalog, audit_output):
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {target}; use overwrite=True or a new path")

    base = pd.read_csv(base_catalog, sep=r"\s+", dtype={"ID": str})
    refit = pd.read_csv(refit_catalog, sep=r"\s+", dtype={"ID": str})
    if base["ID"].duplicated().any():
        raise ValueError("base_catalog contains duplicate IDs")
    if refit["ID"].duplicated().any():
        raise ValueError("refit_catalog contains duplicate IDs")
    missing_columns = [name for name in base.columns if name not in refit.columns]
    extra_columns = [name for name in refit.columns if name not in base.columns]
    if missing_columns or extra_columns:
        raise ValueError(
            "Catalog schema mismatch: "
            f"missing_in_refit={missing_columns}, extra_in_refit={extra_columns}")
    refit = refit[list(base.columns)]

    base_map = {str(row["ID"]): row for row in base.to_dict(orient="records")}
    refit_map = {str(row["ID"]): row for row in refit.to_dict(orient="records")}
    base_qa_map = _read_qa_status_map(base_qa)
    refit_qa_map = _read_qa_status_map(refit_qa)
    base_period_map = _read_qa_status_map(base_period_audit)
    refit_period_map = _read_qa_status_map(refit_period_audit)
    if require_refit_period_audit and not refit_period_map:
        raise ValueError(
            "A non-empty refit_period_audit is required for an audited merge")

    def combined_status(sid, primary, period):
        statuses = [
            str(mapping.get(sid, {}).get("status", "ok"))
            for mapping in (primary, period)
        ]
        return max(statuses, key=lambda value: _STATUS_RANK.get(value, -1))

    merged_map = dict(base_map)
    audit = []
    for sid, candidate in refit_map.items():
        original = base_map.get(sid)
        base_status = combined_status(sid, base_qa_map, base_period_map)
        if require_refit_period_audit and sid not in refit_period_map:
            refit_status = "not_audited"
        else:
            refit_status = combined_status(sid, refit_qa_map, refit_period_map)
        candidate_valid = _valid_error_row(candidate)
        original_valid = _valid_error_row(original) if original is not None else False
        base_rms = _catalog_rms_ratio(original, mode) if original else np.nan
        refit_rms = _catalog_rms_ratio(candidate, mode)
        base_chi2 = float(original.get("chi2", np.nan)) if original else np.nan
        refit_chi2 = float(candidate.get("chi2", np.nan))
        if original is None:
            period_change = np.nan
        else:
            base_period = float(original.get("P", np.nan))
            refit_period = float(candidate.get("P", np.nan))
            period_change = (
                abs(refit_period - base_period) / base_period
                if np.isfinite(base_period) and base_period > 0 and
                np.isfinite(refit_period) else np.nan)

        rms_improved = (
            np.isfinite(base_rms) and np.isfinite(refit_rms) and
            refit_rms <= (1.0 - improvement_fraction) * base_rms)
        chi2_improved = (
            np.isfinite(base_chi2) and np.isfinite(refit_chi2) and
            refit_chi2 <= (1.0 - improvement_fraction) * base_chi2)

        if not candidate_valid:
            decision, selected, reason = (
                "reject", "base" if original is not None else "none",
                "invalid_refit_or_HC3")
        elif refit_status != "ok":
            decision, selected, reason = (
                "reject", "base" if original is not None else "none",
                f"refit_QA_{refit_status}")
        elif original is None:
            merged_map[sid] = candidate
            decision, selected, reason = "accept", "refit", "recovered_missing_source"
        elif not original_valid:
            merged_map[sid] = candidate
            decision, selected, reason = "accept", "refit", "replaced_invalid_base"
        elif base_status in ("failed", "review", "feature_missing"):
            merged_map[sid] = candidate
            decision, selected, reason = "accept", "refit", "resolved_base_QA"
        elif rms_improved or chi2_improved:
            merged_map[sid] = candidate
            decision, selected, reason = "accept", "refit", "fit_metric_improved"
        else:
            decision, selected, reason = "keep", "base", "no_verified_improvement"

        audit.append({
            "ID": sid,
            "decision": decision,
            "selected_source": selected,
            "reason": reason,
            "base_status": base_status if original is not None else "missing",
            "refit_status": refit_status,
            "period_relative_change": period_change,
            "base_rms_ratio": base_rms,
            "refit_rms_ratio": refit_rms,
            "base_chi2": base_chi2,
            "refit_chi2": refit_chi2,
        })

    # Preserve base order and append newly recovered IDs in refit order.
    ordered_ids = list(base["ID"].astype(str))
    ordered_ids.extend(sid for sid in refit["ID"].astype(str) if sid not in base_map)
    merged = pd.DataFrame(
        [merged_map[sid] for sid in ordered_ids if sid in merged_map],
        columns=base.columns)
    if merged["ID"].duplicated().any():
        raise RuntimeError("Internal merge error: duplicate IDs")

    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    # ``sep=r"\s+"`` is used throughout the analysis notebooks.  Empty
    # strings are therefore unsafe: consecutive spaces collapse and shift all
    # following values one column to the left.  Write an explicit token for
    # every missing value so the catalog remains rectangular when reloaded.
    merged.to_csv(output_catalog, sep=" ", index=False, na_rep="nan")
    pd.DataFrame(audit, columns=FD_REFIT_AUDIT_COLUMNS).to_csv(
        audit_output, sep=" ", index=False, na_rep="nan")
    return {
        "base_rows": int(len(base)),
        "refit_rows": int(len(refit)),
        "merged_rows": int(len(merged)),
        "accepted": int(sum(row["decision"] == "accept" for row in audit)),
        "output": str(output_catalog),
        "audit_output": str(audit_output),
    }


# -----------------------------------------------------------------------------
# Gaia catalog audit and provenance-safe manual revision workflow
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GaiaQualityThresholds:
    """Transparent defaults for catalog triage, never automatic rejection."""

    rms_ratio_limit: float = 0.70
    rms_ratio_high_priority: float = 1.00
    min_epochs_per_band: int = 10
    min_occupied_fraction: float = 0.40
    max_phase_gap: float = 0.30
    gaia_period_relative_tolerance: float = 0.01
    external_z_limit: float = 5.0
    r21_absolute_limit: float = 0.15
    r31_absolute_limit: float = 0.20
    r21_scale_floor: float = 0.02
    r31_scale_floor: float = 0.03
    period_bin_width: float = 0.20
    minimum_reference_group: int = 20


REVISION_COLUMNS = (
    "ID", "base_period", "proposed_period", "decision", "confidence",
    "reason_code", "notes", "reviewer", "reviewed_utc",
)

REVISION_DECISIONS = {
    "undecided", "keep_base", "refit_same_period", "adopt_period",
    "exclude_supervised", "defer",
}

REVISION_CONFIDENCE = {
    "not_set", "clear", "preferred", "ambiguous", "unusable",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 hash for provenance manifests."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_frame(
    value: str | Path | pd.DataFrame, *, sep: str | None = None,
) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if path.suffix.lower() == ".ecsv":
        from astropy.table import Table

        return Table.read(path, format="ascii.ecsv").to_pandas()
    if sep is None:
        sep = "," if path.suffix.lower() == ".csv" else r"\s+"
    return pd.read_csv(path, sep=sep, low_memory=False)


def load_fd_catalog(
    value: str | Path | pd.DataFrame, mode: str = "gaia",
) -> pd.DataFrame:
    """Load and validate a nominal or HC3-augmented Fourier catalog."""

    del mode  # reserved for schema-specific validation
    frame = _read_frame(value)
    if "source_id" in frame.columns and "ID" not in frame.columns:
        frame = frame.rename(columns={"source_id": "ID"})
    required = {"ID", "P", "E", "M_fit", "A1", "A2", "Q1", "Q2"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fourier catalog is missing columns: {missing}")
    frame["ID"] = frame["ID"].astype(str)
    if frame["ID"].duplicated().any():
        duplicates = frame.loc[frame["ID"].duplicated(False), "ID"].head().tolist()
        raise ValueError(f"Fourier catalog contains duplicate IDs: {duplicates}")
    return frame.reset_index(drop=True)


def load_gaia_sos_catalog(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load the Gaia DR3 SOS columns needed by the external audit."""

    frame = _read_frame(value)
    if "SOURCE_ID" in frame.columns and "ID" not in frame.columns:
        frame = frame.rename(columns={"SOURCE_ID": "ID"})
    if "source_id" in frame.columns and "ID" not in frame.columns:
        frame = frame.rename(columns={"source_id": "ID"})
    if "ID" not in frame.columns:
        raise ValueError("Gaia SOS table must contain SOURCE_ID, source_id, or ID")
    frame["ID"] = frame["ID"].astype(str)
    if frame["ID"].duplicated().any():
        raise ValueError("Gaia SOS table contains duplicate source IDs")
    return frame.reset_index(drop=True)


def add_fourier_invariants(frame: pd.DataFrame) -> pd.DataFrame:
    """Add local R21/R31 and phase invariants without changing fit values."""

    out = frame.copy()
    m_fit = pd.to_numeric(out["M_fit"], errors="coerce").to_numpy(float)
    a1 = pd.to_numeric(out["A1"], errors="coerce").to_numpy(float)
    a2 = pd.to_numeric(out["A2"], errors="coerce").to_numpy(float)
    q1 = pd.to_numeric(out["Q1"], errors="coerce").to_numpy(float)
    q2 = pd.to_numeric(out["Q2"], errors="coerce").to_numpy(float)
    valid_a1 = np.isfinite(a1) & (a1 > 0)
    out["R21"] = np.divide(
        a2, a1, out=np.full(len(out), np.nan), where=valid_a1 & (m_fit >= 2))
    out["phi21"] = np.where(
        m_fit >= 2, np.mod(q2 - 2.0 * q1, 2.0 * np.pi), np.nan)
    if {"A3", "Q3"}.issubset(out.columns):
        a3 = pd.to_numeric(out["A3"], errors="coerce").to_numpy(float)
        q3 = pd.to_numeric(out["Q3"], errors="coerce").to_numpy(float)
        out["R31"] = np.divide(
            a3, a1, out=np.full(len(out), np.nan), where=valid_a1 & (m_fit >= 3))
        out["phi31"] = np.where(
            m_fit >= 3, np.mod(q3 - 3.0 * q1, 2.0 * np.pi), np.nan)
    else:
        out["R31"] = np.nan
        out["phi31"] = np.nan
    return out


def _gaia_period(frame: pd.DataFrame) -> np.ndarray:
    mode = frame.get(
        "mode_best_classification", pd.Series("", index=frame.index)
    ).astype(str).str.upper()
    pf = pd.to_numeric(frame.get("pf", np.nan), errors="coerce")
    p1 = pd.to_numeric(frame.get("p1_o", np.nan), errors="coerce")
    period = np.where(mode.eq("FIRST_OVERTONE"), p1, pf).astype(float)
    fallback = ~np.isfinite(period) | (period <= 0)
    period[fallback] = np.asarray(pf, dtype=float)[fallback]
    return period


def _quality_aggregate(
    value: str | Path | pd.DataFrame | None,
) -> pd.DataFrame | None:
    if value is None:
        return None
    quality = _read_frame(value)
    if "source_id" in quality.columns and "ID" not in quality.columns:
        quality = quality.rename(columns={"source_id": "ID"})
    if "ID" not in quality.columns:
        raise ValueError("quality catalog must contain ID")
    quality["ID"] = quality["ID"].astype(str)
    numeric = [
        name for name in ("N", "occupied_fraction", "gmax", "coverage_entropy")
        if name in quality
    ]
    for name in numeric:
        quality[name] = pd.to_numeric(quality[name], errors="coerce")
    aggregations: dict[str, tuple[str, str]] = {}
    if "N" in quality:
        aggregations["quality_N_min"] = ("N", "min")
    if "occupied_fraction" in quality:
        aggregations["occupied_fraction_min"] = ("occupied_fraction", "min")
    if "gmax" in quality:
        aggregations["quality_gmax_max"] = ("gmax", "max")
    if "coverage_entropy" in quality:
        aggregations["coverage_entropy_min"] = ("coverage_entropy", "min")
    if not aggregations:
        return quality[["ID"]].drop_duplicates()
    return quality.groupby("ID", as_index=False).agg(**aggregations)


def _robust_location_scale(
    values: np.ndarray, floor: float,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, float(floor)
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    return center, max(scale, float(floor))


def _external_residual_columns(
    frame: pd.DataFrame,
    local_column: str,
    gaia_column: str,
    local_error_column: str,
    gaia_error_column: str,
    *,
    scale_floor: float,
    z_limit: float,
    absolute_limit: float,
    minimum_group: int,
) -> pd.DataFrame:
    """Compute a period-conditioned, error-aware local-minus-Gaia flag."""

    local = pd.to_numeric(
        frame.get(local_column, np.nan), errors="coerce").to_numpy(float)
    gaia = pd.to_numeric(
        frame.get(gaia_column, np.nan), errors="coerce").to_numpy(float)
    delta = local - gaia
    comparable = frame["gaia_fourier_comparable"].to_numpy(bool)
    finite = comparable & np.isfinite(delta)
    global_center, global_scale = _robust_location_scale(delta[finite], scale_floor)
    center = np.full(len(frame), global_center, dtype=float)
    robust_scale = np.full(len(frame), global_scale, dtype=float)

    group_columns = ["gaia_type", "gaia_mode", "period_bin"]
    for _, index in frame.loc[finite].groupby(
        group_columns, dropna=False
    ).groups.items():
        index = np.asarray(list(index), dtype=int)
        if len(index) < int(minimum_group):
            continue
        group_center, group_scale = _robust_location_scale(
            delta[index], scale_floor)
        center[index] = group_center
        robust_scale[index] = group_scale

    local_error = pd.to_numeric(
        frame.get(local_error_column, np.nan), errors="coerce").to_numpy(float)
    gaia_error = pd.to_numeric(
        frame.get(gaia_error_column, np.nan), errors="coerce").to_numpy(float)
    combined_error = np.sqrt(np.square(local_error) + np.square(gaia_error))
    combined_error[~np.isfinite(combined_error) | (combined_error <= 0)] = np.nan
    effective_scale = robust_scale.copy()
    use_error = np.isfinite(combined_error)
    effective_scale[use_error] = np.maximum(
        effective_scale[use_error], combined_error[use_error])
    z = np.abs(delta - center) / effective_scale
    z[~finite] = np.nan
    flag = finite & (
        (z >= float(z_limit)) | (np.abs(delta) >= float(absolute_limit)))

    prefix = local_column
    return pd.DataFrame({
        f"{prefix}_minus_gaia": delta,
        f"{prefix}_reference_center": center,
        f"{prefix}_reference_scale": robust_scale,
        f"{prefix}_combined_reported_error": combined_error,
        f"{prefix}_external_z": z,
        f"{prefix}_gaia_outlier": flag,
    }, index=frame.index)


def build_gaia_fit_quality_table(
    fd_catalog: str | Path | pd.DataFrame,
    gaia_sos_catalog: str | Path | pd.DataFrame | None = None,
    quality_catalog: str | Path | pd.DataFrame | None = None,
    *,
    thresholds: GaiaQualityThresholds | None = None,
) -> pd.DataFrame:
    """Build a fast, catalog-wide Gaia FD audit table without refitting."""

    thresholds = thresholds or GaiaQualityThresholds()
    frame = add_fourier_invariants(load_fd_catalog(fd_catalog, mode="gaia"))
    for name in ("R21_err", "R31_err", "phi21_err", "phi31_err"):
        if name not in frame:
            frame[name] = np.nan

    cfg = get_data_config("gaia")
    active_bands = [str(cfg.filters[index]) for index in cfg.activated_bands]
    ratios, counts, gaps = [], [], []
    for band in active_bands:
        scatter = pd.to_numeric(
            frame.get(f"sig_{band}", np.nan), errors="coerce").to_numpy(float)
        residual = pd.to_numeric(
            frame.get(f"rms_{band}", np.nan), errors="coerce").to_numpy(float)
        ratios.append(np.divide(
            residual, scatter, out=np.full(len(frame), np.nan),
            where=np.isfinite(scatter) & (scatter > 0) & np.isfinite(residual)))
        counts.append(pd.to_numeric(
            frame.get(f"N_{band}", np.nan), errors="coerce").to_numpy(float))
        gaps.append(pd.to_numeric(
            frame.get(f"gmax_{band}", np.nan), errors="coerce").to_numpy(float))
    frame["rms_ratio_max"] = np.nanmax(np.vstack(ratios), axis=0)
    frame["n_epoch_min"] = np.nanmin(np.vstack(counts), axis=0)
    frame["gmax_max"] = np.nanmax(np.vstack(gaps), axis=0)

    quality = _quality_aggregate(quality_catalog)
    if quality is not None:
        frame = frame.merge(quality, on="ID", how="left", validate="one_to_one")
    if "occupied_fraction_min" not in frame:
        frame["occupied_fraction_min"] = np.nan

    frame["gaia_reference_available"] = False
    frame["gaia_period"] = np.nan
    frame["gaia_type"] = ""
    frame["gaia_mode"] = ""
    for name in ("gaia_R21", "gaia_R21_err", "gaia_R31", "gaia_R31_err"):
        frame[name] = np.nan

    if gaia_sos_catalog is not None:
        gaia = load_gaia_sos_catalog(gaia_sos_catalog)
        keep = ["ID"] + [name for name in (
            "pf", "pf_error", "p1_o", "p1_o_error", "r21_g", "r21_g_error",
            "r31_g", "r31_g_error", "type_best_classification",
            "mode_best_classification",
        ) if name in gaia.columns]
        gaia = gaia[keep].copy()
        gaia["gaia_period"] = _gaia_period(gaia)
        gaia = gaia.rename(columns={
            "r21_g": "gaia_R21", "r21_g_error": "gaia_R21_err",
            "r31_g": "gaia_R31", "r31_g_error": "gaia_R31_err",
            "type_best_classification": "gaia_type",
            "mode_best_classification": "gaia_mode",
        })
        replace = [name for name in gaia.columns if name != "ID"]
        frame = frame.drop(columns=replace, errors="ignore").merge(
            gaia, on="ID", how="left", validate="one_to_one")
        frame["gaia_reference_available"] = np.isfinite(
            pd.to_numeric(frame["gaia_R21"], errors="coerce")) | np.isfinite(
            pd.to_numeric(frame["gaia_R31"], errors="coerce"))

    period = pd.to_numeric(frame["P"], errors="coerce").to_numpy(float)
    gaia_period = pd.to_numeric(
        frame["gaia_period"], errors="coerce").to_numpy(float)
    valid_period_pair = (
        np.isfinite(period) & (period > 0)
        & np.isfinite(gaia_period) & (gaia_period > 0))
    relative = np.full(len(frame), np.nan)
    relative[valid_period_pair] = np.abs(
        period[valid_period_pair] / gaia_period[valid_period_pair] - 1.0)
    frame["gaia_period_relative_difference"] = relative
    frame["gaia_period_agreement"] = valid_period_pair & (
        relative <= thresholds.gaia_period_relative_tolerance)
    frame["gaia_period_disagreement"] = (
        valid_period_pair & ~frame["gaia_period_agreement"].to_numpy(bool))
    frame["gaia_fourier_comparable"] = (
        frame["gaia_reference_available"].to_numpy(bool)
        & frame["gaia_period_agreement"].to_numpy(bool))
    logp = np.log10(np.where(period > 0, period, np.nan))
    width = float(thresholds.period_bin_width)
    frame["logP"] = logp
    frame["period_bin"] = np.floor(logp / width) * width

    for local, gaia_name, local_err, gaia_err, floor, absolute in (
        ("R21", "gaia_R21", "R21_err", "gaia_R21_err",
         thresholds.r21_scale_floor, thresholds.r21_absolute_limit),
        ("R31", "gaia_R31", "R31_err", "gaia_R31_err",
         thresholds.r31_scale_floor, thresholds.r31_absolute_limit),
    ):
        scored = _external_residual_columns(
            frame, local, gaia_name, local_err, gaia_err,
            scale_floor=floor, z_limit=thresholds.external_z_limit,
            absolute_limit=absolute,
            minimum_group=thresholds.minimum_reference_group)
        for name in scored:
            frame[name] = scored[name]

    invalid_core = ~(
        np.isfinite(period) & (period > 0)
        & np.isfinite(pd.to_numeric(frame["E"], errors="coerce"))
        & (pd.to_numeric(frame["M_fit"], errors="coerce") >= 1))
    nominal_flag = pd.to_numeric(
        frame.get("flag", 0), errors="coerce").fillna(1).to_numpy(float) != 0
    high_rms = frame["rms_ratio_max"].to_numpy(float) > thresholds.rms_ratio_limit
    very_high_rms = (
        frame["rms_ratio_max"].to_numpy(float)
        > thresholds.rms_ratio_high_priority)
    low_epoch = frame["n_epoch_min"].to_numpy(float) < thresholds.min_epochs_per_band
    large_gap = frame["gmax_max"].to_numpy(float) > thresholds.max_phase_gap
    occupied = pd.to_numeric(
        frame["occupied_fraction_min"], errors="coerce").to_numpy(float)
    low_occupation = np.isfinite(occupied) & (
        occupied < thresholds.min_occupied_fraction)
    r21_outlier = frame["R21_gaia_outlier"].to_numpy(bool)
    r31_outlier = frame["R31_gaia_outlier"].to_numpy(bool)

    reasons, priorities, statuses = [], [], []
    for index in range(len(frame)):
        row_reasons = []
        priority = 0
        if invalid_core[index]:
            row_reasons.append("invalid_core_fit")
            priority = 3
        if nominal_flag[index]:
            row_reasons.append("nominal_flag")
            priority = max(priority, 2)
        if high_rms[index]:
            row_reasons.append("high_rms_ratio")
            priority = max(priority, 1)
        if very_high_rms[index]:
            row_reasons.append("very_high_rms_ratio")
            priority = max(priority, 2)
        if low_epoch[index]:
            row_reasons.append("low_epoch_support")
            priority = max(priority, 1)
        if large_gap[index]:
            row_reasons.append("large_phase_gap")
            priority = max(priority, 2)
        if low_occupation[index]:
            row_reasons.append("low_phase_occupation")
            priority = max(priority, 2)
        if bool(frame.iloc[index]["gaia_period_disagreement"]):
            row_reasons.append("gaia_period_disagreement")
            priority = max(priority, 3)
        if r21_outlier[index]:
            row_reasons.append("gaia_R21_outlier")
            priority = max(priority, 2)
        if r31_outlier[index]:
            row_reasons.append("gaia_R31_outlier")
            priority = max(priority, 1)
        reasons.append(";".join(row_reasons) if row_reasons else "ok")
        priorities.append(priority)
        statuses.append(
            "failed" if invalid_core[index]
            else ("review" if row_reasons else "ok"))
    frame["qa_status"] = statuses
    frame["review_priority"] = priorities
    frame["review_reasons"] = reasons
    frame["review_selected"] = frame["review_priority"].ge(2)

    lead = [
        "ID", "pulsation", "qa_status", "review_priority", "review_reasons",
        "review_selected", "P", "gaia_period", "gaia_period_relative_difference",
        "M_fit", "flag", "n_epoch_min", "rms_ratio_max", "gmax_max",
        "occupied_fraction_min", "R21", "gaia_R21", "R21_err",
        "gaia_R21_err", "R21_external_z", "R21_gaia_outlier", "R31",
        "gaia_R31", "R31_err", "gaia_R31_err", "R31_external_z",
        "R31_gaia_outlier",
    ]
    lead = [name for name in lead if name in frame]
    rest = [name for name in frame.columns if name not in lead]
    return frame[lead + rest].sort_values(
        ["review_priority", "ID"], ascending=[False, True]).reset_index(drop=True)


def write_quality_audit_bundle(
    audit: pd.DataFrame,
    output_dir: str | Path,
    *,
    input_paths: Mapping[str, str | Path] | None = None,
    thresholds: GaiaQualityThresholds | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Write audit/review tables plus hashes and parameters."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "gaia_fd_quality_audit.csv"
    review_path = output_dir / "gaia_fd_review_queue.csv"
    manifest_path = output_dir / "gaia_fd_quality_audit_manifest.json"
    if not overwrite and any(
        path.exists() for path in (audit_path, review_path, manifest_path)
    ):
        raise FileExistsError(
            f"Audit bundle already exists in {output_dir}; "
            "set overwrite=True or use a new run tag")
    audit.to_csv(audit_path, index=False)
    audit.loc[audit["review_selected"].astype(bool)].to_csv(review_path, index=False)
    inputs = {}
    for name, value in (input_paths or {}).items():
        path = Path(value)
        inputs[name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path) if path.exists() else None,
        }
    payload = {
        "schema": "gaia-fd-quality-audit-v1",
        "created_utc": _utc_now(),
        "thresholds": asdict(thresholds or GaiaQualityThresholds()),
        "inputs": inputs,
        "counts": {
            "all": int(len(audit)),
            "ok": int(audit["qa_status"].eq("ok").sum()),
            "review": int(audit["qa_status"].eq("review").sum()),
            "failed": int(audit["qa_status"].eq("failed").sum()),
            "gaia_R21_outlier": int(audit["R21_gaia_outlier"].sum()),
            "gaia_R31_outlier": int(audit["R31_gaia_outlier"].sum()),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "audit": str(audit_path), "review": str(review_path),
        "manifest": str(manifest_path),
    }


def create_revision_manifest(
    audit: pd.DataFrame,
    output_path: str | Path,
    *,
    review_only: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create the small human-editable decision ledger."""

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    selected = audit.loc[
        audit["review_selected"].astype(bool)] if review_only else audit
    manifest = pd.DataFrame({
        "ID": selected["ID"].astype(str),
        "base_period": pd.to_numeric(selected["P"], errors="coerce"),
        "proposed_period": np.nan,
        "decision": "undecided",
        "confidence": "not_set",
        "reason_code": selected["review_reasons"].astype(str),
        "notes": "",
        "reviewer": "",
        "reviewed_utc": "",
    }, columns=REVISION_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def load_revision_manifest(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    manifest = _read_frame(value, sep=",")
    missing = sorted(set(REVISION_COLUMNS) - set(manifest.columns))
    if missing:
        raise ValueError(f"Revision manifest is missing columns: {missing}")
    manifest = manifest[list(REVISION_COLUMNS)].copy()
    manifest["ID"] = manifest["ID"].astype(str)
    return manifest


def validate_revision_manifest(
    value: str | Path | pd.DataFrame,
    base_catalog: str | Path | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate decisions and protect against editing the wrong base version."""

    manifest = load_revision_manifest(value)
    if manifest["ID"].duplicated().any():
        raise ValueError("Revision manifest contains duplicate IDs")
    unknown_decision = sorted(
        set(manifest["decision"].astype(str)) - REVISION_DECISIONS)
    unknown_confidence = sorted(
        set(manifest["confidence"].astype(str)) - REVISION_CONFIDENCE)
    if unknown_decision:
        raise ValueError(f"Unknown revision decisions: {unknown_decision}")
    if unknown_confidence:
        raise ValueError(f"Unknown confidence values: {unknown_confidence}")
    proposed = pd.to_numeric(manifest["proposed_period"], errors="coerce")
    adopt = manifest["decision"].eq("adopt_period")
    invalid_period = ~np.isfinite(proposed) | (proposed <= 0)
    if (adopt & invalid_period).any():
        bad = manifest.loc[adopt & invalid_period, "ID"].tolist()
        raise ValueError(
            f"adopt_period requires a positive proposed_period: {bad}")
    ambiguous = manifest["confidence"].isin(["ambiguous", "unusable", "not_set"])
    if (adopt & ambiguous).any():
        bad = manifest.loc[adopt & ambiguous, "ID"].tolist()
        raise ValueError(
            f"Ambiguous/unset decisions cannot adopt a single period: {bad}")
    if base_catalog is not None:
        base = load_fd_catalog(base_catalog)
        base_map = base.set_index("ID")["P"]
        missing_ids = sorted(set(manifest["ID"]) - set(base_map.index))
        if missing_ids:
            raise ValueError(
                f"Manifest IDs missing from base catalog: {missing_ids[:5]}")
        expected = manifest["ID"].map(base_map).astype(float)
        stored = pd.to_numeric(manifest["base_period"], errors="coerce")
        mismatch = ~np.isclose(
            expected, stored, rtol=1e-10, atol=0.0, equal_nan=False)
        if mismatch.any():
            raise ValueError(
                "Manifest base_period does not match the current base catalog for IDs: "
                + ",".join(manifest.loc[mismatch, "ID"].head().tolist()))
    return manifest


def upsert_revision_decision(
    manifest_path: str | Path,
    source_id: Any,
    *,
    decision: str,
    confidence: str,
    base_period: float | None = None,
    proposed_period: float | None = None,
    reason_code: str = "",
    notes: str = "",
    reviewer: str = "",
) -> pd.DataFrame:
    """Insert or update one source and immediately persist the decision."""

    manifest_path = Path(manifest_path)
    if decision not in REVISION_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(REVISION_DECISIONS)}")
    if confidence not in REVISION_CONFIDENCE:
        raise ValueError(f"confidence must be one of {sorted(REVISION_CONFIDENCE)}")
    if manifest_path.exists():
        manifest = load_revision_manifest(manifest_path)
    else:
        manifest = pd.DataFrame(columns=REVISION_COLUMNS)
    sid = str(source_id)
    mask = (
        manifest["ID"].astype(str).eq(sid)
        if len(manifest) else np.zeros(0, dtype=bool))
    if mask.any():
        index = manifest.index[mask][0]
        if base_period is None:
            base_period = float(manifest.at[index, "base_period"])
    else:
        if base_period is None:
            raise ValueError("base_period is required for a new source")
        index = len(manifest)
    if decision == "refit_same_period" and proposed_period is None:
        proposed_period = base_period
    record = {
        "ID": sid, "base_period": base_period,
        "proposed_period": proposed_period, "decision": decision,
        "confidence": confidence,
        "reason_code": reason_code or "manual_review", "notes": notes,
        "reviewer": reviewer, "reviewed_utc": _utc_now(),
    }
    for name, item in record.items():
        manifest.loc[index, name] = item
    manifest = manifest[list(REVISION_COLUMNS)].sort_values(
        "ID").reset_index(drop=True)
    validate_revision_manifest(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest.to_csv(temporary, index=False)
    temporary.replace(manifest_path)
    return manifest


def revision_jobs_from_manifest(
    value: str | Path | pd.DataFrame,
    base_catalog: str | Path | pd.DataFrame | None = None,
) -> tuple[list[Any], dict[str, float]]:
    manifest = validate_revision_manifest(value, base_catalog=base_catalog)
    selected = manifest["decision"].isin(["adopt_period", "refit_same_period"])
    jobs = manifest.loc[selected].copy()
    period = pd.to_numeric(jobs["proposed_period"], errors="coerce")
    same = jobs["decision"].eq("refit_same_period")
    period.loc[same & ~np.isfinite(period)] = pd.to_numeric(
        jobs.loc[same & ~np.isfinite(period), "base_period"], errors="coerce")
    period_map = dict(zip(jobs["ID"].astype(str), period.astype(float)))
    ids: list[Any] = [
        int(sid) if str(sid).isdigit() else sid for sid in jobs["ID"]]
    return ids, period_map


def _qa_status_map(path: str | Path | None) -> dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    qa = pd.read_csv(path, sep=r"\s+", dtype={"ID": str})
    if qa.empty:
        return {}
    severity = {"ok": 0, "feature_missing": 1, "review": 2, "failed": 3}
    result: dict[str, str] = {}
    for sid, group in qa.groupby("ID"):
        result[str(sid)] = max(
            group["status"].astype(str),
            key=lambda value: severity.get(value, 99))
    return result


def merge_manifest_revisions(
    base_catalog: str | Path | pd.DataFrame,
    refit_catalog: str | Path | pd.DataFrame,
    revision_manifest: str | Path | pd.DataFrame,
    output_catalog: str | Path,
    *,
    refit_qa: str | Path | None = None,
    audit_output: str | Path | None = None,
    exclusion_output: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Apply explicit human decisions to a new catalog, never to the base."""

    base = load_fd_catalog(base_catalog)
    refit = load_fd_catalog(refit_catalog)
    manifest = validate_revision_manifest(revision_manifest, base_catalog=base)
    output_catalog = Path(output_catalog)
    if audit_output is None:
        audit_output = output_catalog.with_name(output_catalog.stem + "_audit.csv")
    if exclusion_output is None:
        exclusion_output = output_catalog.with_name(
            output_catalog.stem + "_supervised_exclusions.csv")
    audit_output = Path(audit_output)
    exclusion_output = Path(exclusion_output)
    for target in (output_catalog, audit_output, exclusion_output):
        if target.exists() and not overwrite:
            raise FileExistsError(target)
    if (
        isinstance(base_catalog, (str, Path))
        and output_catalog.resolve() == Path(base_catalog).resolve()
    ):
        raise ValueError("output_catalog must not overwrite base_catalog")

    qa_map = _qa_status_map(refit_qa)
    base_map = {sid: row for sid, row in base.set_index("ID").iterrows()}
    refit_map = {sid: row for sid, row in refit.set_index("ID").iterrows()}
    manifest_map = manifest.set_index("ID")
    columns = list(base.columns) + [
        name for name in refit.columns if name not in base.columns]
    merged_rows, audit_rows = [], []
    for sid in base["ID"].astype(str):
        decision = (
            str(manifest_map.at[sid, "decision"])
            if sid in manifest_map.index else "keep_base")
        selected, reason, row = "base", "not_selected_for_revision", base_map[sid]
        if decision in {"adopt_period", "refit_same_period"}:
            candidate = refit_map.get(sid)
            qa_status = qa_map.get(sid, "ok")
            valid_core = candidate is not None and all(
                np.isfinite(float(candidate.get(name, np.nan)))
                for name in ("P", "E", "M_fit"))
            if candidate is None:
                reason = "missing_refit_row"
            elif qa_status in {"failed", "review"}:
                reason = f"refit_qa_{qa_status}"
            elif not valid_core:
                reason = "invalid_refit_core"
            else:
                row, selected, reason = (
                    candidate, "refit", "explicit_manifest_decision")
        output_row = {name: row.get(name, np.nan) for name in columns}
        output_row["ID"] = sid
        merged_rows.append(output_row)
        audit_rows.append({
            "ID": sid, "decision": decision, "selected_source": selected,
            "reason": reason,
            "base_period": float(base_map[sid].get("P", np.nan)),
            "output_period": float(output_row.get("P", np.nan)),
            "refit_qa_status": qa_map.get(sid, "not_listed_ok"),
        })
    merged = pd.DataFrame(merged_rows, columns=columns)
    exclusions = manifest.loc[
        manifest["decision"].eq("exclude_supervised")].copy()
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_catalog, sep=" ", index=False, na_rep="nan")
    pd.DataFrame(audit_rows).to_csv(audit_output, index=False)
    exclusions.to_csv(exclusion_output, index=False)
    return {
        "base_rows": int(len(base)), "merged_rows": int(len(merged)),
        "refit_selected": int(sum(
            row["selected_source"] == "refit" for row in audit_rows)),
        "supervised_exclusions": int(len(exclusions)),
        "output": str(output_catalog), "audit_output": str(audit_output),
        "exclusion_output": str(exclusion_output),
    }
