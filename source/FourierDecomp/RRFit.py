from __future__ import annotations

import os
import signal
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Union
from tqdm.auto import tqdm
from astropy.table import Table

import pickle
import tempfile
import subprocess
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from .IO import get_data_config, prepare_fitlc
from .period_finder import period_fit_boundary_search

_ACTIVE_PROCS = {}
_ACTIVE_LOCK = threading.Lock()


@dataclass(frozen=True)
class RRFitTemplate:
    """One 15-harmonic RRFit sine-series template."""

    index: int
    name: str
    A: np.ndarray
    Q: np.ndarray


def load_rrfit_templates(path):
    """Read the four-line-per-template ``templates.dat`` file."""

    path = Path(path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) % 4:
        raise ValueError(
            f"Invalid RRFit template file: expected groups of four lines, got {len(lines)}")
    templates = {}
    for offset in range(0, len(lines), 4):
        index = offset // 4 + 1
        name = lines[offset]
        amplitude = np.asarray(lines[offset + 2].split(), dtype=float)
        phase = np.asarray(lines[offset + 3].split(), dtype=float)
        if amplitude.size != phase.size or amplitude.size == 0:
            raise ValueError(f"Invalid RRFit template coefficients at template {index}")
        templates[index] = RRFitTemplate(index, name, amplitude, phase)
    return templates


def evaluate_rrfit_template(template, phase):
    """Evaluate the RRFit/Fortran sine-series convention at phase."""

    phase = np.asarray(phase, dtype=float) % 1.0
    orders = 1.0 + np.arange(len(template.A), dtype=float)
    angle = 2.0 * np.pi * orders[:, None] * phase[None, :] + template.Q[:, None]
    return np.sum(template.A[:, None] * np.sin(angle), axis=0)


def rrfit_summary_path(outdir, source_id, posfixs=""):
    """Return the canonical per-source summary path."""

    if posfixs and not str(posfixs).startswith("_"):
        posfixs = "_" + str(posfixs)
    return Path(outdir) / f"rrfit{posfixs}_{source_id}.summary"


def rrfit_metadata_path(workdir, source_id, posfixs=""):
    """Return the canonical per-source RRFit period-search metadata path."""

    if posfixs and not str(posfixs).startswith("_"):
        posfixs = "_" + str(posfixs)
    return Path(workdir) / f"rrfit_meta{posfixs}_{source_id}.ecsv"


def load_rrfit_metadata(workdir, source_id, posfixs=""):
    """Load Lomb--Scargle/window metadata written during RRFit planning."""

    path = rrfit_metadata_path(workdir, source_id, posfixs=posfixs)
    if not path.exists():
        raise FileNotFoundError(path)
    table = Table.read(path, format="ascii.ecsv")
    if len(table) != 1:
        raise ValueError(f"Expected one RRFit metadata row in {path}")
    row = table[0]
    return {
        "source_id": str(row["sid"]),
        "P0_LS": float(row["P0_LS"]),
        "Zmax": float(row["Zmax"]),
        "alias_freqs": np.asarray(row["alias_freqs"], dtype=float).ravel(),
        "logP_bounds": np.asarray(row["logP_bounds"], dtype=float),
        "n_jobs": int(row["n_jobs"]),
        "path": str(path),
    }


def period_relation_to_reference(candidate_period, reference_period,
                                 alias_frequencies=(),
                                 relative_tolerance=0.01,
                                 max_harmonic=4):
    """Label a candidate as direct/harmonic/window-alias relative to a period.

    This is a diagnostic relation, not evidence that the reference period is
    true.  The tested frequency families follow ``|m f_ref +/- f_window|`` in
    addition to integer harmonics and subharmonics.
    """

    candidate_period = float(candidate_period)
    reference_period = float(reference_period)
    if not (np.isfinite(candidate_period) and candidate_period > 0
            and np.isfinite(reference_period) and reference_period > 0):
        return {"alias_relation": "invalid", "alias_relative_error": np.nan,
                "alias_frequency": np.nan}
    candidate_frequency = 1.0 / candidate_period
    reference_frequency = 1.0 / reference_period
    targets = [("direct", reference_frequency, np.nan)]
    for harmonic in range(2, int(max_harmonic) + 1):
        targets.append((f"harmonic_{harmonic}",
                        harmonic * reference_frequency, np.nan))
        targets.append((f"subharmonic_{harmonic}",
                        reference_frequency / harmonic, np.nan))
    for alias_frequency in np.asarray(alias_frequencies, dtype=float).ravel():
        if not np.isfinite(alias_frequency) or alias_frequency <= 0:
            continue
        for harmonic in range(1, int(max_harmonic) + 1):
            for sign, token in ((1.0, "+"), (-1.0, "-")):
                target = abs(harmonic * reference_frequency
                             + sign * alias_frequency)
                if target > 0:
                    targets.append((
                        f"window_m{harmonic}{token}fw", target,
                        float(alias_frequency)))
    scored = [
        (abs(candidate_frequency / target - 1.0), name, alias_frequency)
        for name, target, alias_frequency in targets if target > 0
    ]
    error, relation, alias_frequency = min(scored, key=lambda value: value[0])
    if error > float(relative_tolerance):
        relation = "unexplained"
        alias_frequency = np.nan
    return {
        "alias_relation": relation,
        "alias_relative_error": float(error),
        "alias_frequency": float(alias_frequency)
        if np.isfinite(alias_frequency) else np.nan,
    }


def annotate_rrfit_alias_relations(solutions, reference_period,
                                   alias_frequencies=(),
                                   relative_tolerance=0.01):
    """Attach reference/window alias labels to RRFit period families."""

    frame = solutions.copy()
    annotations = [
        period_relation_to_reference(
            period, reference_period, alias_frequencies,
            relative_tolerance=relative_tolerance)
        for period in frame["period_family"]
    ]
    return pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(annotations)], axis=1)


def load_rrfit_summary(value):
    """Load one RRFit summary with stable, Python-friendly column aliases."""

    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = Table.read(path, format="ascii.basic").to_pandas()
    rename = {
        "sid": "source_id", "ID": "rrfit_source_id", "T": "template_index",
        "EPOCH": "epoch", "P": "period", "Amp(1)": "amp_1",
        "Amp(2)": "amp_2", "<M1>": "mean_1", "<M2>": "mean_2",
        "CHI^2": "chi2",
    }
    frame = frame.rename(columns={
        key: value for key, value in rename.items() if key in frame.columns})
    required = {
        "source_id", "bandpair", "template_index", "epoch", "period",
        "amp_1", "amp_2", "mean_1", "mean_2", "chi2",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"RRFit summary is missing columns: {missing}")
    frame["source_id"] = frame["source_id"].astype(str)
    numeric = [
        "template_index", "epoch", "period", "amp_1", "amp_2",
        "mean_1", "mean_2", "chi2", "pmin", "pmax", "returncode",
    ]
    for name in numeric:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    valid = (
        np.isfinite(frame["period"]) & frame["period"].gt(0)
        & np.isfinite(frame["chi2"]) & frame["template_index"].ge(1))
    if "returncode" in frame:
        valid &= frame["returncode"].fillna(1).eq(0)
    frame["rrfit_valid"] = valid
    return frame.reset_index(drop=True)


def build_rrfit_solution_families(summary, relative_tolerance=0.01):
    """Cluster valid RRFit job solutions by period for visual review.

    Raw chi-square is normalized only within a band pair; it is not treated as
    a cross-band posterior probability.  The family score is for display order
    and never chooses the scientific period automatically.
    """

    frame = load_rrfit_summary(summary)
    frame = frame.loc[frame["rrfit_valid"]].copy()
    if frame.empty:
        return frame
    frame["chi2_relative"] = frame["chi2"] / frame.groupby(
        "bandpair")["chi2"].transform("min")
    frame = frame.sort_values("period").reset_index(drop=True)
    centers = []
    family_periods = []
    family_index = []
    for period in frame["period"].to_numpy(float):
        match = next((
            index for index, center in enumerate(centers)
            if abs(period / center - 1.0) <= float(relative_tolerance)
        ), None)
        if match is None:
            centers.append(period)
            family_periods.append([])
            match = len(centers) - 1
        family_index.append(match)
        family_periods[match].append(period)
        centers[match] = float(np.median(family_periods[match]))
    frame["period_family_index"] = family_index
    family = frame.groupby("period_family_index").agg(
        period_family=("period", "median"),
        period_min=("period", "min"),
        period_max=("period", "max"),
        n_solutions=("period", "size"),
        n_bandpairs=("bandpair", "nunique"),
        median_chi2_relative=("chi2_relative", "median"),
    ).reset_index()
    max_bandpairs = int(frame["bandpair"].nunique())
    family["family_score"] = (
        family["median_chi2_relative"]
        + 0.5 * (max_bandpairs - family["n_bandpairs"])
    )
    family = family.sort_values(
        ["n_bandpairs", "family_score", "period_family"],
        ascending=[False, True, True]).reset_index(drop=True)
    family["solution_id"] = [f"R{i + 1}" for i in range(len(family))]
    frame = frame.merge(
        family, on="period_family_index", how="left", validate="many_to_one")
    return frame.sort_values(
        ["solution_id", "bandpair", "chi2_relative"]).reset_index(drop=True)


def _best_rrfit_row_per_bandpair(summary):
    """Return the best valid RRFit row in each independently fitted pair."""

    frame = load_rrfit_summary(summary)
    frame = frame.loc[frame["rrfit_valid"]].copy()
    if frame.empty:
        return frame
    indices = frame.groupby("bandpair", observed=True)["chi2"].idxmin()
    return frame.loc[indices].sort_values("bandpair").reset_index(drop=True)


def _epoch_phase_span(frame):
    """Largest circular epoch separation across band-pair solutions."""

    if len(frame) < 2:
        return np.nan
    period = float(np.nanmedian(frame["period"]))
    if not np.isfinite(period) or period <= 0:
        return np.nan
    phase = np.mod(frame["epoch"].to_numpy(float) / period, 1.0)
    distances = []
    for i in range(len(phase)):
        for j in range(i + 1, len(phase)):
            delta = abs(phase[i] - phase[j])
            distances.append(min(delta, 1.0 - delta))
    return float(max(distances)) if distances else np.nan


def compare_fixed_period_rrfit(
    source_ids, base_outdir, revised_outdir, *,
    base_posfixs="base_fixed", revised_posfixs="revised_fixed",
):
    """Compare paired fixed-period RRFit controls without choosing a period.

    The base and revised runs must use identical band pairs and template bank.
    Positive ``chi2_fractional_improvement`` means that the manually selected
    period improved the best-template reduced chi-square.  The function makes
    no automatic accept/reject decision and returns source and band-pair tables.
    """

    pair_rows = []
    source_cache = {}
    for source_id in source_ids:
        sid = str(source_id)
        base_path = rrfit_summary_path(
            base_outdir, sid, posfixs=base_posfixs)
        revised_path = rrfit_summary_path(
            revised_outdir, sid, posfixs=revised_posfixs)
        try:
            base = _best_rrfit_row_per_bandpair(base_path)
            revised = _best_rrfit_row_per_bandpair(revised_path)
        except (FileNotFoundError, ValueError):
            base = pd.DataFrame()
            revised = pd.DataFrame()
        source_cache[sid] = (base, revised)
        bandpairs = sorted(
            set(base.get("bandpair", pd.Series(dtype=str)).astype(str))
            | set(revised.get("bandpair", pd.Series(dtype=str)).astype(str)))
        for bandpair in bandpairs:
            base_row = base.loc[base["bandpair"].astype(str).eq(bandpair)]
            revised_row = revised.loc[
                revised["bandpair"].astype(str).eq(bandpair)]
            b = base_row.iloc[0] if len(base_row) else None
            r = revised_row.iloc[0] if len(revised_row) else None
            chi2_base = float(b["chi2"]) if b is not None else np.nan
            chi2_revised = float(r["chi2"]) if r is not None else np.nan
            ratio = (
                chi2_revised / chi2_base
                if np.isfinite(chi2_base) and chi2_base > 0
                and np.isfinite(chi2_revised) else np.nan)
            pair_rows.append({
                "source_id": sid,
                "bandpair": bandpair,
                "period_base_fixed": (
                    float(b["period"]) if b is not None else np.nan),
                "period_revised_fixed": (
                    float(r["period"]) if r is not None else np.nan),
                "chi2_base_fixed": chi2_base,
                "chi2_revised_fixed": chi2_revised,
                "chi2_ratio_revised_to_base": ratio,
                "chi2_fractional_improvement": (
                    1.0 - ratio if np.isfinite(ratio) else np.nan),
                "template_base": (
                    int(b["template_index"]) if b is not None else np.nan),
                "template_revised": (
                    int(r["template_index"]) if r is not None else np.nan),
                "epoch_base": float(b["epoch"]) if b is not None else np.nan,
                "epoch_revised": (
                    float(r["epoch"]) if r is not None else np.nan),
            })

    pair_table = pd.DataFrame(pair_rows)
    source_rows = []
    for source_id in source_ids:
        sid = str(source_id)
        group = pair_table.loc[pair_table["source_id"].eq(sid)]
        valid = group["chi2_ratio_revised_to_base"].notna()
        ratios = group.loc[valid, "chi2_ratio_revised_to_base"]
        base, revised = source_cache[sid]
        source_rows.append({
            "source_id": sid,
            "n_bandpairs_compared": int(valid.sum()),
            "n_bandpairs_improved": int((ratios < 1.0).sum()),
            "fraction_bandpairs_improved": (
                float((ratios < 1.0).mean()) if len(ratios) else np.nan),
            "median_chi2_ratio_revised_to_base": (
                float(ratios.median()) if len(ratios) else np.nan),
            "median_chi2_fractional_improvement": (
                float((1.0 - ratios).median()) if len(ratios) else np.nan),
            "all_bandpairs_improved": (
                bool((ratios < 1.0).all()) if len(ratios) else False),
            "base_epoch_phase_span": _epoch_phase_span(base),
            "revised_epoch_phase_span": _epoch_phase_span(revised),
            "base_template_agreement": (
                bool(base["template_index"].nunique() == 1)
                if len(base) else False),
            "revised_template_agreement": (
                bool(revised["template_index"].nunique() == 1)
                if len(revised) else False),
        })
    return pd.DataFrame(source_rows), pair_table


def build_rrfit_review_index(
    audit,
    outdir,
    *,
    decisions=None,
    posfixs="",
    period_bin_edges=(-np.inf, 0.0, 0.3, 0.6, 1.0, 1.4, np.inf),
):
    """Index the human review queue by base-period bin and RRFit availability."""

    frame = audit.loc[audit["review_selected"].astype(bool)].copy()
    frame["ID"] = frame["ID"].astype(str)
    frame["base_logP"] = np.log10(pd.to_numeric(frame["P"], errors="coerce"))
    edges = np.asarray(period_bin_edges, dtype=float)
    labels = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        if np.isneginf(lower):
            labels.append(f"< {upper:.1f}")
        elif np.isposinf(upper):
            labels.append(f">= {lower:.1f}")
        else:
            labels.append(f"{lower:.1f}-{upper:.1f}")
    frame["period_bin"] = pd.cut(
        frame["base_logP"], bins=edges, labels=labels,
        right=False, include_lowest=True)
    if decisions is not None:
        decision_frame = decisions.copy()
        decision_frame["ID"] = decision_frame["ID"].astype(str)
        frame = frame.drop(columns=["decision"], errors="ignore").merge(
            decision_frame[["ID", "decision"]], on="ID", how="left")
    if "decision" not in frame:
        frame["decision"] = "undecided"
    frame["decision"] = frame["decision"].fillna("undecided")
    frame["rrfit_summary"] = [
        str(rrfit_summary_path(outdir, sid, posfixs=posfixs))
        for sid in frame["ID"]]
    frame["rrfit_ready"] = frame["rrfit_summary"].map(
        lambda value: Path(value).exists())
    return frame.sort_values(
        ["period_bin", "review_priority", "base_logP", "ID"],
        ascending=[True, False, True, True]).reset_index(drop=True)


def next_rrfit_review_source(review_index, period_bin=None):
    """Return the next ready, undecided source without doing any fitting."""

    frame = review_index.copy()
    mask = frame["rrfit_ready"].astype(bool) & frame["decision"].eq("undecided")
    if period_bin is not None:
        mask &= frame["period_bin"].astype(str).eq(str(period_bin))
    pending = frame.loc[mask]
    if pending.empty:
        return None
    return str(pending.iloc[0]["ID"])

# =============================================
# Process management
# =============================================
def _register_proc(job_id, proc):
    with _ACTIVE_LOCK:
        _ACTIVE_PROCS[job_id] = proc

def _unregister_proc(job_id):
    with _ACTIVE_LOCK:
        _ACTIVE_PROCS.pop(job_id, None)

def kill_all_active_processes():
    """
    Kill all currently running rrfit.e subprocess groups.
    Works on POSIX systems.
    """
    with _ACTIVE_LOCK:
        items = list(_ACTIVE_PROCS.items())

    for job_id, proc in items:
        try:
            if proc.poll() is None: # subprocess.Popen -> checking the running status of process 
                os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass

    # forced termination
    for job_id, proc in items:
        try:
            proc.wait(timeout=1.0)
        except Exception:
            try:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass

# ==============================
# Helpers
# ==============================
@dataclass
class RRFitJob:
    sid: Union[str, int]
    fitlc_path: str
    filters: list
    selected_bands: list 
    P0: float # initial period: Lomb-Scargle
    p0flag: float
    window_idx: int
    tmpl_start: int = 1
    tmpl_end: int = 25
    pmin: float = 0.5
    pmax: float = 300
    Amin: float = 0.05
    Amax: float = 3.0
        
    @property
    def n_bands(self):
        return len(self.selected_bands)
    @property
    def prefixs(self):
        # assuming that the order of band prefixes are identical to input filter list order
        return np.arange(len(self.filters)) 
    @property
    def bands(self):
        return [int(self.prefixs[self.filters==b]) for b in self.selected_bands]
    @property
    def bandpair(self):
        # photometric band pairs for simultaneous fitting 
        return "+".join(self.selected_bands)
    @property
    def bandpair_prefixs(self):
        # photometric band prefixes pairs for simultaneous fitting
        return "+".join(map(str,self.bands))
    @property
    def job_id(self):
        return f"{self.sid}_{self.bandpair}_{self.window_idx:02d}"
    
def rrfit_job_to_dict(job):
    return {
        "sid": job.sid,
        "fitlc_path": str(job.fitlc_path),
        "filters": list(job.filters),
        "selected_bands": list(job.selected_bands),
        "P0": float(job.P0),
        "p0flag": int(job.p0flag),
        "window_idx": int(job.window_idx),
        "tmpl_start": int(job.tmpl_start),
        "tmpl_end": int(job.tmpl_end),
        "pmin": float(job.pmin),
        "pmax": float(job.pmax),
        "Amin": float(job.Amin),
        "Amax": float(job.Amax),
    }

def rrfit_job_from_dict(d):
    return RRFitJob(
        sid=d["sid"],
        fitlc_path=d["fitlc_path"],
        filters=np.array(d["filters"], dtype=object),
        selected_bands=list(d["selected_bands"]),
        P0=d["P0"],
        p0flag=d["p0flag"],
        window_idx=d["window_idx"],
        tmpl_start=d.get("tmpl_start", 1),
        tmpl_end=d.get("tmpl_end", 25),
        pmin=d.get("pmin", 0.5),
        pmax=d.get("pmax", 300.0),
        Amin=d.get("Amin", 0.05),
        Amax=d.get("Amax", 3.0),
    )
    
def parse_rrfit_outputs(fpath):
    # fpath: RRFit output file path
    if not fpath.exists(): return None
    try:  
        tbl = Table.read(fpath, format='ascii')
    except Exception: 
        return None
    if len(tbl)==0: return None
    return dict(tbl[-1])

# ==============================================
# Setup RRFit inputs / Process jobs
# ==============================================
def write_rrfit_inputs(job, workdir):
    # Open rrfit.param/fitlc_list/lomb_scargle.txt file, and write the inputs.
    if job.n_bands>2: 
        raise ValueError("RRFit supports only <=2-band fitting simultaneously.")
    # rrfit.param
    lines = []
    lines.append(" ".join(map(str, job.bands)) + " # SELECTED PHOTOMETRIC BAND PREFIXS")
    lines.append(f"{job.pmin:.6f} {job.pmax:.6f} # PERIOD RANGE")
    lines.append(f"{job.Amin:.6f} {job.Amax:.6f} # AMPLITUDE RANGE")
    lines.append(f"{job.tmpl_start:d} {job.tmpl_end:d} # TEMPLATE RANGE")
    (workdir / "rrfit.param").write_text("\n".join(lines) + "\n")
    # fitlc_list
    (workdir / "fitlc_list").write_text(str(job.fitlc_path) + "\n")
    # lomb_scargle.txt
    with open(workdir / "lomb_scargle.txt", "w") as f:
        f.write("SOURCE_ID P_LS p0flag\n")
        f.write(f"{job.sid} {job.P0:.10f} {job.p0flag}\n")

def build_rrfit_jobs(source_lc, workdir, mode='gaia',
                     bandpairs=(("g","bp"),("g","rp")),
                     A_bounds=(0.05, 3.0), p_bounds=None,
                     fixed_period=None,
                     n0=5, K=5, Kw=10, snr_LS=3., snr_window=5.,harmonics=2,
                     logP_tol=0.1, max_width=1.0, tmpl_start=1, tmpl_end=25,
                     overwrite=True, save=True, posfixs=""
):
    """
    workdir: working directory (for .fitlc, job, meta, and other temporary files)

    For a given source,
    1) LS/window-alias-based logP boundaries
    2) RRFitJob list + metadata
    3) export .json/.ecsv files(optional)
    """
    from . import params

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    cfg = get_data_config(mode)
    filters = cfg.filters

    # fitlc path 준비
    sid = source_lc.sid
    fitlc_path = source_lc.fitlc_path

    # Lomb-Scargle / Window-alias period window.  ``fixed_period`` is a
    # deliberately separate mode used only after a human period decision.
    # RRFit still optimizes epoch, amplitudes, means and template identity,
    # while pmin == pmax makes the Fortran period coordinate constant.
    P0_LS, Zmax = np.nan, np.nan
    if fixed_period is not None:
        fixed_period = float(fixed_period)
        if not np.isfinite(fixed_period) or fixed_period <= 0:
            raise ValueError("fixed_period must be finite and positive")
        if not (params.pmin <= fixed_period <= params.pmax):
            raise ValueError(
                f"fixed_period={fixed_period} is outside "
                f"[{params.pmin}, {params.pmax}]")
        P0_LS = fixed_period
        alias_freqs = []
        logP_bounds = [(np.log10(fixed_period), np.log10(fixed_period))]
    elif p_bounds is None:
        # boundary search
        P_LS, Z_LS, alias_freqs, logP_bounds = period_fit_boundary_search(
            source_lc.t, source_lc.mag, source_lc.emag, source_lc.bands, 
            n0=n0, K=K, Kw=Kw, snr_LS=snr_LS, snr_window=snr_window, 
            harmonics=harmonics, logP_tol=logP_tol, max_width=max_width
        )
        pidx = np.argmax(Z_LS)
        P0_LS, Zmax = P_LS[pidx], Z_LS[pidx] # best LS period
    elif isinstance(p_bounds,(list, tuple)) and len(p_bounds)==2:
        logP_bounds = [(np.log10(p_bounds[0]), np.log10(p_bounds[1]))] # ensure the dimension
        alias_freqs = []
    else:
        raise ValueError("unsupported p_bounds types")
    
    jobs = []
    if fixed_period is not None:
        logP_bounds_full = logP_bounds
    else:
        # The first job is the requested full-range scan.  The following jobs
        # are narrower LS/window-alias windows and must not be mistaken for a
        # fixed-period verification.
        logP_bounds_full = [
            (np.log10(params.pmin), np.log10(params.pmax))
        ] + logP_bounds
    for iw, logP_bound in enumerate(logP_bounds_full):
        if fixed_period is not None:
            logP0 = np.log10(fixed_period)
            P0 = fixed_period
            pmin = fixed_period
            pmax = fixed_period
        else:
            logP0 = np.mean(logP_bound)
            P0 = 10**logP0
            pmin = max(params.pmin, 10**(logP_bound[0]-logP_tol)) # add padding
            pmax = min(params.pmax, 10**(logP_bound[1]+logP_tol))
        if pmin > pmax: continue
        
        # previous definition of p0flag: relative offset from P_Gaia
        # Calculate p0flag by comparing the best Lomb-Scargle period and representative value of the given period range
        p0flag = 0 if abs(logP0 - np.log10(P0_LS)) < min(0.05, logP_tol) else 1
        
        for bp in bandpairs:
            jobs.append(RRFitJob(sid=sid,
                                 fitlc_path=fitlc_path,
                                 filters=filters, selected_bands=bp,
                                 P0=P0, p0flag=p0flag,
                                 window_idx=iw, pmin=pmin, pmax=pmax,
                                 tmpl_start=tmpl_start, tmpl_end=tmpl_end,
                                 Amin=A_bounds[0], Amax=A_bounds[1])
                       )
    # metadata
    meta = {"sid": sid, "P0_LS": P0_LS, "Zmax": Zmax,
            "alias_freqs": list(alias_freqs), "logP_bounds": logP_bounds_full,
            "n_jobs": len(jobs),
            "period_mode": "fixed" if fixed_period is not None else "search",
            "fixed_period": (float(fixed_period)
                             if fixed_period is not None else np.nan)}

    if save:
        if not posfixs.startswith("_"): posfixs = "_" + posfixs
        job_fpath = workdir / f"rrfit_jobs{posfixs}_{sid}.pkl"
        meta_fpath = workdir / f"rrfit_meta{posfixs}_{sid}.ecsv"
        # job file
        if overwrite or (not job_fpath.exists()):
            payload = {"sid": str(sid), "n_jobs": len(jobs),
                       "jobs": [rrfit_job_to_dict(j) for j in jobs]}
            with open(job_fpath, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        # meta file
        meta_tbl = Table({key:[val] for key, val in meta.items()}) 
        meta_tbl.write(meta_fpath, format="ascii.ecsv", overwrite=True)

        # return file path only
        return {"sid": str(sid), "job_file": str(job_fpath), "meta_file": str(meta_fpath), "n_jobs": len(jobs)}
    
    return jobs, meta

def build_rrfit_plan(sids, workdir, outdir, mode='gaia', ls_data=None, fitlc_list=None,
                          bandpairs=(("g","bp"),("g","rp")), max_workers=8,
                          overwrite=True, posfixs="", fixed_periods=None,
                          **kwargs):
    """
    Create RRFit jobs and Lomb-Scargle metadata for all sources.
    - multiprocessing
    - workdir: rrfit_jobs_<sid>.json / rrfit_meta_<sid>.ecsv
    - outdir: rrfit_plan.ecsv (summary)
    """
    workdir = Path(workdir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fixed_period_map = None
    if fixed_periods is not None:
        fixed_period_map = {
            str(key): float(value) for key, value in dict(fixed_periods).items()
        }
        missing_fixed = [
            str(sid) for sid in sids if str(sid) not in fixed_period_map
        ]
        if missing_fixed:
            raise ValueError(
                "fixed_periods is missing source IDs: "
                + ",".join(missing_fixed[:5]))

    tasks = []
    for i, sid in tqdm(enumerate(sids), total=len(sids), desc="Prepare .fitlc"):
        fitlc_path_i = None
        if isinstance(fitlc_list, (list, np.ndarray)):
            # assumming the same order
            if len(sids)!=len(fitlc_list):
                raise ValueError(f"Dimension mismatch between sids={len(sids)} and fitlc_list={len(fitlc_list)}")
            fitlc_path_i = fitlc_list[i]
        elif isinstance(fitlc_list, dict):
            fitlc_path_i = fitlc_list.get(sid, None)

        if fitlc_path_i is None:
            if ls_data is None:
                raise ValueError("Either fitlc_path or ls_data must be provided.")
            
        # create .fitlc file under working directory
        source_lc = prepare_fitlc(sid, mode=mode,
                                  ls_data=ls_data, fitlc_path=fitlc_path_i, workdir=workdir) 
            
        task = {
            "sid": sid, "source_lc": source_lc, "workdir": workdir,
            "mode": mode, "bandpairs": bandpairs, "overwrite": overwrite,
            "save": True, "posfixs": posfixs, **kwargs,
        }
        if fixed_period_map is not None:
            task["fixed_period"] = fixed_period_map[str(sid)]
        tasks.append(task)
    
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(build_rrfit_jobs, **{key: value for key, value in t.items()
                                          if key != "sid"}): t
            for t in tasks
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Constructing RRFit jobs"):
            task = futs[fut]; sid = task.get("sid")
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"[RRFit plan failed] sid={sid}: {repr(e)}")
                rows.append({
                    "ID": sid,
                    "status": "failed",
                    "error": repr(e),
                })
                raise RuntimeError(f"Failed while constructing RRFit job for sid={sid}") from e

    tbl = Table(rows)
    if posfixs and (not posfixs.startswith("_")): posfixs = "_" + posfixs
    plan_fpath = outdir / f"rrfit_plan{posfixs}.dat"
    tbl.write(plan_fpath, format="ascii.basic", overwrite=True)
    return plan_fpath

def load_rrfit_plan(plan_fpath, sids=None):
    """
    Create job_pool, meta_pool by reading rrfit_jobs_<sid>.json / rrfit_meta_<sid>.ecsv files
    """
    plan_tbl = Table.read(plan_fpath, format="ascii")

    sids_full = np.asarray(plan_tbl['sid'])
    job_files_full = np.asarray(plan_tbl['job_file'])
    meta_files_full = np.asarray(plan_tbl['meta_file'])
    
    if sids is not None:
        requested = {str(sid) for sid in sids}
        sid_mask = np.asarray([str(sid) in requested for sid in sids_full])
        selected_sids = sids_full[sid_mask]
        job_files = job_files_full[sid_mask]
        meta_files = meta_files_full[sid_mask]
    else:
        selected_sids = sids_full
        job_files = job_files_full
        meta_files = meta_files_full

    job_pool = []
    meta_pool = {}

    for sid, job_fpath, meta_fpath in zip(
        selected_sids, job_files, meta_files
    ):
        job_fpath, meta_fpath = Path(job_fpath), Path(meta_fpath)
        #job file
        if not job_fpath.exists(): continue
        with open(job_fpath, "rb") as f:
            payload = pickle.load(f)

        jobs = [rrfit_job_from_dict(d) for d in payload.get("jobs", [])]
        job_pool.extend(jobs)

        # meta file
        if meta_fpath.exists():
            meta_tbl = Table.read(meta_fpath, format="ascii.ecsv")
            if len(meta_tbl)>0:
                meta_row = meta_tbl[0]
                meta_pool[sid] = {
                    "sid": sid,
                    "P0_LS": meta_row["P0_LS"],
                    "Zmax": meta_row["Zmax"],
                    "alias_freqs": meta_row["alias_freqs"],
                    "logP_bounds": meta_row["logP_bounds"],
                    "n_jobs": int(meta_row["n_jobs"]),
                    "period_mode": (str(meta_row["period_mode"])
                                    if "period_mode" in meta_tbl.colnames
                                    else "search"),
                    "fixed_period": (float(meta_row["fixed_period"])
                                     if "fixed_period" in meta_tbl.colnames
                                     else np.nan),
                }
        else:
            meta_pool[sid] = {
                "sid": sid,
                "P0_LS": np.nan,
                "Zmax": np.nan,
                "alias_freqs": [],
                "logP_bounds": [],
                "n_jobs": len(jobs),
                "period_mode": "unknown",
                "fixed_period": np.nan,
            }

    return job_pool, meta_pool

# ========================================
# RRFit job executor 
# ========================================
# Individual jobs run in separated temporary folders.
def run_rrfit_job(job, rrfit_exe, base_workdir=None, is_test=False,
                  timeout=300):
    # typical RRFit execution time ~2m30s (for 25 templates)
    rrfit_exe = Path(rrfit_exe).resolve()
    base_dir = rrfit_exe.parent
    tmpl_path = base_dir / "templates.dat"
    outname = f"rrfit_{job.bandpair_prefixs}.out"
    if not tmpl_path.exists():
        raise FileNotFoundError(f"RRFit requires the Fourier templates file template.dat: {tmpl_path}")
    if base_workdir is None:
        base_workdir = base_dir / "temp"
    else: base_workdir = Path(base_workdir)
    base_workdir.mkdir(parents=True, exist_ok=True)

    # run multiple rrfit jobs in the temporary directory
    # e.g.) job A -> rrfit_sid_1, job B -> rrfit_sid_2,...
    with tempfile.TemporaryDirectory(prefix=f"rrfit_{job.sid}_",
                                     dir=base_workdir) as td:
        if is_test:
            workdir = base_workdir / f"test_{job.job_id}" # fixed directory
            workdir.mkdir(parents=True, exist_ok=True) 
            # remove remaining files
            for fname in ["rrfit.param","fitlc_list","lomb_scargle.txt",outname]:
                fpath = workdir / fname
                if fpath.exists(): fpath.unlink()
        else: workdir = Path(td)
        write_rrfit_inputs(job, workdir)

        # subprocess - run shell command
        # RRFit.e runs as a separated process from parent Python process
        try:
            proc = subprocess.Popen([str(rrfit_exe), str(workdir)], cwd=base_dir,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, 
                                text=True, # return strings
                                start_new_session=True, # process group separation
                                ) 
            _register_proc(job.job_id, proc)
            try:
                stdout, stderr = proc.communicate(timeout=timeout) # waiting for child process to finish
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate()
                stderr = (stderr or "") + f"\n[TIMEOUT] exceeded {timeout} sec"
                returncode = -9
        finally:
            _unregister_proc(job.job_id)

        # single result for a single job -> read last row
        row = parse_rrfit_outputs(workdir / outname) 
        return {
            "sid": job.sid,
            "job_id": job.job_id,
            "bandpair": job.bandpair,
            "P0": job.P0,
            "window_idx": job.window_idx,
            "pmin": job.pmin,
            "pmax": job.pmax,
            "p0flag": job.p0flag,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "result": row,
        }
    
# ================================
# Export results
# ================================
def write_source_rrfit_results(outdir, sid, results, posfixs=""):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # summary table
    rows = []
    for r in results:
        # Collect the results from the individual fits for each window
        row = {
            "sid": r["sid"],
            "job_id": r["job_id"],
            "bandpair": r["bandpair"],
            "pmin": r["pmin"],
            "pmax": r["pmax"],
            "p0flag": r["p0flag"],
            "returncode": r["returncode"],
        }
        if r["result"] is not None:
            # Unpack the results (read from RRFit output file)
            for k, v in r["result"].items(): row[k] = v
        rows.append(row)

    tbl = Table(rows) if rows else Table()
    if posfixs and (not posfixs.startswith("_")): posfixs = "_" + posfixs
    summary_fname = outdir / f"rrfit{posfixs}_{sid}.summary"
    tbl.write(summary_fname, format='ascii.basic', overwrite=True)
    return summary_fname

# ================================
# Main function
# ================================
def run_rrfit(
    sids, rrfit_exe, outdir, workdir=None, mode=None, fitlc_list=None,
    ls_data=None, bandpairs=(("g", "bp"), ("g", "rp")), max_workers=8,
    is_test=False, timeout=300, posfixs="", resume=True,
    checkpoint_every=100, fixed_periods=None, **kwargs,
):
    """Run the source RRFit plan in parallel with resumable source outputs.

    RRFit itself is an external CPU executable, so threads here supervise
    independent subprocesses.  Existing per-source summaries are the resume
    boundary.  Growing log tables are checkpointed periodically instead of
    being rewritten after every job.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if workdir is None:
        workdir = Path(rrfit_exe).parent / "temp"
    sids = list(dict.fromkeys(sids))
    fixed_period_map = (
        {str(key): float(value) for key, value in dict(fixed_periods).items()}
        if fixed_periods is not None else None)
    if fixed_period_map is not None:
        missing = [str(sid) for sid in sids if str(sid) not in fixed_period_map]
        if missing:
            raise ValueError(
                "fixed_periods is missing source IDs: "
                + ",".join(missing[:5]))

    # build / load jobs
    if posfixs and (not posfixs.startswith("_")): posfixs = "_" + posfixs
    plan_fpath = outdir / f"rrfit_plan{posfixs}.dat"
    rebuild_plan = not plan_fpath.exists()
    if not rebuild_plan:
        existing_plan = Table.read(plan_fpath, format="ascii")
        existing_ids = {str(value) for value in existing_plan["sid"]}
        rebuild_plan = not {str(value) for value in sids}.issubset(existing_ids)
    if rebuild_plan:
        plan_fpath = build_rrfit_plan(sids, workdir, outdir, mode=mode, 
                                      ls_data=ls_data, fitlc_list=fitlc_list,
                                      bandpairs=bandpairs, max_workers=max_workers,
                                      overwrite=True, posfixs=posfixs,
                                      fixed_periods=fixed_period_map, **kwargs)
    elif fixed_period_map is not None:
        # A resumable output directory is immutable with respect to adopted
        # periods.  Reusing it with a changed decision ledger would otherwise
        # silently mix incompatible summaries.
        existing_jobs, _ = load_rrfit_plan(plan_fpath, sids=sids)
        mismatched = []
        for job in existing_jobs:
            target = fixed_period_map.get(str(job.sid), np.nan)
            if (
                not np.isfinite(target)
                or not np.isclose(job.pmin, target, rtol=0.0, atol=1e-10)
                or not np.isclose(job.pmax, target, rtol=0.0, atol=1e-10)
            ):
                mismatched.append(str(job.sid))
        if mismatched:
            raise ValueError(
                "Existing RRFit plan does not match fixed_periods for IDs "
                + ",".join(sorted(set(mismatched))[:5])
                + "; use a new output directory")

    completed = set()
    if resume:
        for sid in sids:
            path = rrfit_summary_path(outdir, sid, posfixs=posfixs)
            if path.exists() and path.stat().st_size > 0:
                if fixed_period_map is not None:
                    target = fixed_period_map[str(sid)]
                    summary = load_rrfit_summary(path)
                    fitted = summary.loc[
                        summary["rrfit_valid"], "period"].to_numpy(float)
                    if (
                        fitted.size == 0
                        or not np.allclose(
                            fitted, target, rtol=0.0, atol=1.5e-6)
                    ):
                        raise ValueError(
                            f"Existing fixed-period summary for {sid} does "
                            f"not match P={target}; use a new output directory")
                completed.add(str(sid))
    pending_sids = [sid for sid in sids if str(sid) not in completed]
    if not pending_sids:
        print(f"RRFit resume: all {len(sids)} source summaries already exist")
        source_path = outdir / "rrfit_source_log.dat"
        job_path = outdir / "rrfit_job_log.ecsv"
        source_table = Table.read(source_path, format="ascii") if source_path.exists() else Table()
        job_table = Table.read(job_path, format="ascii.ecsv") if job_path.exists() else Table()
        return source_table, job_table

    job_pool, meta_pool = load_rrfit_plan(plan_fpath, sids=pending_sids)
    if not (job_pool and meta_pool):
        raise ValueError("Invalid Job data list or metadata list")

    # track source-wise job status
    results_pool = defaultdict(list)
    n_done_pool = defaultdict(int)
    n_total_pool = {
        str(sid): int(meta["n_jobs"]) for sid, meta in meta_pool.items()}
    source_written = set()

    job_log_path = outdir / "rrfit_job_log.ecsv"
    source_log_path = outdir / "rrfit_source_log.dat"
    log_rows = []
    source_rows = []
    if resume and job_log_path.exists():
        log_rows = Table.read(job_log_path, format="ascii.ecsv").to_pandas().to_dict(
            orient="records")
    if resume and source_log_path.exists():
        source_rows = Table.read(source_log_path, format="ascii").to_pandas().to_dict(
            orient="records")

    def write_checkpoints():
        if log_rows:
            job_frame = pd.DataFrame(log_rows).drop_duplicates(
                "job_id", keep="last")
            Table.from_pandas(job_frame).write(
                job_log_path, format="ascii.ecsv", overwrite=True)
        if source_rows:
            source_frame = pd.DataFrame(source_rows).drop_duplicates(
                "sid", keep="last")
            Table.from_pandas(source_frame).write(
                source_log_path, format="ascii.basic", overwrite=True)

    # run all jobs
    jobs_since_checkpoint = 0
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(run_rrfit_job, job, rrfit_exe, workdir, is_test, timeout): job
                     for job in job_pool}
            with tqdm(total=len(pending_sids), desc='Sources') as pbar, \
                 tqdm(total=len(job_pool), desc='RRFit Jobs') as job_pbar:
                for fut in as_completed(futs):
                    job = futs[fut]
                    sid = job.sid
                    sid_key = str(sid)
                    try:
                        r = fut.result()
                    except Exception as e:
                        r = {"sid": sid, "job_id": job.job_id,
                             "bandpair": job.bandpair,
                             "P0": job.P0, "window_idx": job.window_idx,
                             "pmin": job.pmin, "pmax": job.pmax,
                             "p0flag": job.p0flag, "returncode": -999,
                             "stdout": "","stderr": repr(e),"result": None,
                            }

                    job_pbar.update(1)
                    jobs_since_checkpoint += 1

                    results_pool[sid_key].append(r)
                    n_done_pool[sid_key] += 1

                    # log
                    log_rows.append({
                        "sid": r["sid"],
                        "job_id": r["job_id"],
                        "window_idx": r["window_idx"],
                        "bandpair": r["bandpair"],
                        "P0": r["P0"],
                        "pmin": r["pmin"],
                        "pmax": r["pmax"],
                        "p0flag": r["p0flag"],
                        "returncode": r["returncode"],
                        "result_ok": r["result"] is not None,
                        "stderr": (str(r["stderr"]).replace("\r", "\\r").replace("\n", "\\n")[:2000]
                                   if r.get("stderr") is not None else ""),
                    })

                    # If all jobs have finished for a given source
                    if (
                        sid_key not in source_written
                        and n_done_pool[sid_key] >= n_total_pool[sid_key]
                    ):
                        # Collect results from the separate jobs of a given source
                        meta_key = next(
                            key for key in meta_pool if str(key) == sid_key)
                        meta = meta_pool[meta_key]
                        results = results_pool.get(sid_key, [])
                        # Write result to summary/meta files
                        summary_fpath = write_source_rrfit_results(outdir, sid, results, 
                                                                   posfixs=posfixs)
                        n_success = sum(int(r["returncode"] == 0 and r["result"] is not None) 
                                        for r in results
                                        )
                        
                        source_rows.append({
                            "sid": sid,
                            "n_jobs_total": meta["n_jobs"],
                            "n_jobs_finished": len(results),
                            "n_jobs_success": n_success,
                            "summary_file": str(summary_fpath)
                        })
                        source_written.add(sid_key)
                        pbar.update(1) 

                    if jobs_since_checkpoint >= max(int(checkpoint_every), 1):
                        write_checkpoints()
                        jobs_since_checkpoint = 0
                        
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Terminating active rrfit.e processes...")
        kill_all_active_processes()
        write_checkpoints()
        raise

    write_checkpoints()
    source_log_tbl = (
        Table.read(source_log_path, format="ascii")
        if source_log_path.exists() else Table())
    job_log_tbl = (
        Table.read(job_log_path, format="ascii.ecsv")
        if job_log_path.exists() else Table())
    return source_log_tbl, job_log_tbl

# ==========================================
# Decrypted
# =========================================
"""
# (Old version): build rrfit job
def build_rrfit_job_pool(sids, mode='gaia', bandpairs=(("g","bp"),("g","rp")),
                         ls_data=None, fitlc_list=None, workdir=None, outdir=None, **kwargs):
    # Create RRFitJob and return Lomb-Scargle metadata for all sources
    job_pool = []
    meta_pool = {}
    for i, sid in enumerate(sids):
       if fitlc_list is None: fitlc_path_i = None
       elif isinstance(fitlc_list, (list, np.ndarray)):
           # presumming the same order
           if len(sids)!=len(fitlc_list):
               raise ValueError(f"Dimension mismatch between sids={len(sids)} and fitlc_list={len(fitlc_list)}")
           fitlc_path_i = fitlc_list[i]
       elif isinstance(fitlc_list, dict):
           fitlc_path_i = fitlc_list.get(sid, None)

       P0_LS, Zmax, jobs, alias_freqs, logP_bounds = build_rrfit_jobs(
           sid, mode=mode, bandpairs=bandpairs, 
           ls_data=ls_data, fitlc_path=fitlc_path_i, workdir=workdir, **kwargs
           )
       
       meta_pool[sid] = {"sid": sid, "P0_LS": P0_LS, "Zmax": Zmax,
                  "alias_freqs": alias_freqs.tolist(), "logP_bounds": logP_bounds, "n_jobs":len(jobs)
                  }
       job_pool.extend(jobs)
    return job_pool, meta_pool

def run_rrfit_single(sid, rrfit_exe, outdir, 
                     mode=None, fitlc_path=None, ls_data=None, workdir=None, 
                     bandpairs=(("g", "bp"), ("g", "rp")), 
                     max_workers=8, is_test=False, **kwargs):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if workdir is None:
        workdir = Path(rrfit_exe).parent / "temp"

    P0_LS, Zmax, jobs, alias_freqs, logP_bounds = build_rrfit_jobs(
        sid=sid, mode=mode,
        fitlc_path=fitlc_path, ls_data=ls_data, workdir=workdir,
        bandpairs=bandpairs, **kwargs
    )

    results = []
    try:
        if jobs:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(run_rrfit_job, job, rrfit_exe, workdir, is_test) for job in jobs]
                for fut in futs: results.append(fut.result())
    except KeyboardInterrupt:
        kill_all_active_processes()
        raise

    summary_fpath = write_source_rrfit_results(outdir, sid, results)
    return {"sid": sid, 
            "n_jobs": len(jobs), 
            "summary": str(summary_fpath)}

def build_rrfit_jobs(sid, mode='gaia', bandpairs=(("g","bp"),("g","rp")),
                     ls_data=None, fitlc_path=None, workdir=None,
                     A_bounds=(0.05, 3.0), p_bounds=None,
                     n0=5, K=5, Kw=10, snr_LS=3., snr_window=5.,harmonics=2,
                     logP_tol=0.1, max_width=1.0, tmpl_start=1, tmpl_end=25):
    
    from . import params
    
    cfg = get_data_config(mode)
    filters = cfg.filters
    
    # source_lc: RRFitLC
    source_lc = prepare_fitlc(sid, mode=mode, ls_data=ls_data, 
                              fitlc_path=fitlc_path, workdir=workdir)
    fitlc_path = source_lc.fitlc_path
    
    P0_LS, Zmax = np.nan, np.nan
    if p_bounds is None:
        # boundary search
        P_LS, Z_LS, alias_freqs, logP_bounds = period_fit_boundary_search(
            source_lc.t, source_lc.mag, source_lc.emag, source_lc.bands, 
            n0=n0, K=K, Kw=Kw, snr_LS=snr_LS, snr_window=snr_window, 
            harmonics=harmonics, logP_tol=logP_tol, max_width=max_width
        )
        pidx = np.argmax(Z_LS)
        P0_LS, Zmax = P_LS[pidx], Z_LS[pidx] # best LS period
    elif isinstance(p_bounds,(list, tuple)) and len(p_bounds)==1:
        logP_bounds = [(np.log10(p_bounds[0]), np.log10(p_bounds[1]))] # ensure dimension
    else:
        raise ValueError("unsupported p_bounds types")
    
    jobs = []
    logP_bounds = [(np.log10(params.pmin), np.log10(params.pmax))] + logP_bounds # global period scan
    for iw, logP_bound in enumerate(logP_bounds):
        logP0 = np.mean(logP_bound)
        P0 = 10**logP0
        pmin = max(params.pmin, 10**(logP_bound[0]-logP_tol)) # add padding
        pmax = min(params.pmax, 10**(logP_bound[1]+logP_tol))
        if pmin > pmax: continue
        
        # previous definition of p0flag: relative offset from P_Gaia
        # calculate p0flag by comparing the LS best period and representative value of given period range
        p0flag = 0 if abs(logP0 - np.log10(P0_LS)) < min(0.05, logP_tol) else 1
        
        for bp in bandpairs:
            jobs.append(RRFitJob(sid=sid,
                                 fitlc_path=fitlc_path,
                                 filters=filters,
                                 selected_bands=bp,
                                 P0=P0, p0flag=p0flag,
                                 window_idx=iw, pmin=pmin, pmax=pmax,
                                 tmpl_start=tmpl_start, tmpl_end=tmpl_end,
                                 Amin=A_bounds[0], Amax=A_bounds[1])
                       )
    return P0_LS, Zmax, jobs, alias_freqs, logP_bounds

def build_rrfit_plan(sid, outdir, mode='gaia', bandpairs=(("g","bp"),("g","rp")),
                     ls_data=None, fitlc_path=None, workdir=None, 
                     overwrite=True, return_summary=True, **kwargs
                    ):

    for a given source,
    - build_rrfit_jobs()
    - save jobs (.json) & meta (.ecsv)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    P0_LS, Zmax, jobs, alias_freqs, logP_bounds = build_rrfit_jobs(
        sid=sid,
        mode=mode,
        bandpairs=bandpairs,
        ls_data=ls_data,
        fitlc_path=fitlc_path,
        workdir=workdir,
        **kwargs,
    )

    jobs_dict = [rrfit_job_to_dict(job) for job in jobs]

    job_fpath = outdir / f"rrfit_jobs_{sid}.json"
    meta_fpath = outdir / f"rrfit_LS_{sid}.ecsv"

    if overwrite or (not job_fpath.exists()):
        with open(job_fpath, "w", encoding="utf-8") as f:
            json.dump({"sid": sid, "n_jobs": len(jobs_dict), "jobs": jobs_dict}, f, 
                      indent=2, ensure_ascii=False)

    meta_tbl = Table({
        "sid": [sid],
        "P0_LS": [P0_LS],
        "Zmax": [Zmax],
        "alias_freqs": [alias_freqs],
        "logP_bounds": [logP_bounds],
        "n_jobs": [len(jobs_dict)],
    })
    meta_tbl.write(meta_fpath, format="ascii.ecsv", overwrite=True)

    if return_summary:
        return {"sid": sid, "job_file": str(job_fpath), "meta_file": str(meta_fpath),
                "n_jobs": len(jobs_dict)}
    else:
        return jobs, meta_tbl

                    
"""
