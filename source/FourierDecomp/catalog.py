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
