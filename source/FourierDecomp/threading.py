from multiprocessing import Manager, get_context
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tqdm.notebook import tqdm
from typing import Any
import csv
import time
import pandas as pd

from .params import period_fit, use_optim, adaptive_lam, use_refit, mode_default, init
from .IO import build_fd_header, epoch_arrays
from .catalog import (
    FD_ERROR_COLUMNS,
    FD_QA_COLUMNS,
    PERIOD_AUDIT_COLUMNS,
    assess_period_stability,
    assess_nominal_fit_quality,
    build_fd_error_header,
    compute_minimal_hc3_errors,
    error_values,
    make_qa_record,
    merge_qa_records,
    nan_error_record,
    period_audit_values,
    qa_values,
    file_sha256,
    revision_jobs_from_manifest,
)

def _init_worker(ls_data, df_ident, df_rrfit, templates):
    """Runs once per worker process."""
    from . import decomposition as decomp_mod 
    decomp_mod.ls_data = ls_data
    decomp_mod.df_ident = df_ident
    decomp_mod.df_rrfit = df_rrfit
    decomp_mod.templates = templates


def _header_from_file(path):
    with open(path, "r", newline="") as handle:
        line = handle.readline().strip()
    return line.split()


def _ensure_output_header(path, header):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _header_from_file(path)
        if existing != list(header):
            raise ValueError(
                f"Header mismatch for {path}. "
                "Use a new output path instead of mixing catalog schemas.")
        return
    with open(path, "w", newline="") as handle:
        csv.writer(handle, delimiter=" ").writerow(header)


def _qa_output_path(fd_output):
    fd_output = Path(fd_output)
    return fd_output.with_name(f"{fd_output.stem}_failures.dat")


def _feature_missing_qa(sid, error_record, nominal_record):
    missing = [
        name for name in FD_ERROR_COLUMNS
        if not _is_finite(error_record.get(name))
    ]
    if not missing:
        return None
    return make_qa_record(
        sid=sid,
        status="feature_missing",
        retryable=False,
        stage="hc3_feature",
        reason="nonfinite:" + ",".join(missing),
        nominal_record=nominal_record,
    )


def _is_finite(value):
    try:
        import numpy as np
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _failure_qa_record(sid, stage, exc):
    """Classify structural data failures separately from retryable fit failures."""
    reason = repr(exc)
    nonretryable_tokens = (
        "missing_observed_active_band",
        "insufficient_active_epochs_for_M_MIN",
        "period_search_baseline_too_short",
        "period_search_requires_at_least_two_epochs",
        "period_search_requires_positive_baseline",
    )
    retryable = not any(token in reason for token in nonretryable_tokens)
    return make_qa_record(
        sid=sid,
        status="failed",
        retryable=retryable,
        stage=stage,
        reason=reason,
    )


def _calculate_error_payload(
        sid, nominal_record, mode, n_draws, random_state, robust,
        rms_ratio_limit, min_occupied_fraction, max_phase_gap):
    from . import decomposition as decomp_mod

    epoch_data = epoch_arrays(decomp_mod.ls_data, sid, mode=mode)
    qa_record = assess_nominal_fit_quality(
        sid=sid,
        nominal_record=nominal_record,
        mode=mode,
        epoch_data=epoch_data,
        rms_ratio_limit=rms_ratio_limit,
        min_occupied_fraction=min_occupied_fraction,
        max_phase_gap=max_phase_gap,
    )
    try:
        error_record = compute_minimal_hc3_errors(
            sid=sid,
            nominal_record=nominal_record,
            mode=mode,
            epoch_data=epoch_data,
            n_draws=n_draws,
            random_state=random_state,
            robust=robust,
        )
        qa_record = merge_qa_records(
            qa_record,
            _feature_missing_qa(sid, error_record, nominal_record),
        )
    except Exception as exc:
        error_record = nan_error_record()
        qa_record = merge_qa_records(
            qa_record,
            make_qa_record(
                sid=sid,
                status="failed",
                retryable=True,
                stage="hc3",
                reason=repr(exc),
                nominal_record=nominal_record,
            ),
        )
    return error_record, qa_record


def _worker_call(args):
    """Picklable wrapper for nominal fitting and optional HC3 propagation."""
    if len(args) == 15:
        # Backward compatibility with notebooks/tests written before the
        # per-run period-candidate controls were exposed.
        args = (*args, None, None)
    if len(args) == 17:
        # Backward compatibility before reference-guided refits were exposed.
        args = (*args, None, 0.0, 3)
    (
        sid, mode, init, period_fit, use_optim, adaptive_lam, use_refit,
        verbose, return_error, error_n_draws, error_random_state,
        error_robust, rms_ratio_limit, min_occupied_fraction, max_phase_gap,
        K, harmonics, reference_period, reference_period_window,
        reference_period_screen_order,
    ) = args
    from . import decomposition as decomp_mod

    try:
        row = decomp_mod.fourier_decomp(
            sid,
            mode=mode,
            init=init,
            period_fit=period_fit,
            use_optim=use_optim,
            adaptive_lam=adaptive_lam,
            use_refit=use_refit,
            verbose=verbose,
            K=K,
            harmonics=harmonics,
            reference_period=reference_period,
            reference_period_window=reference_period_window,
            reference_period_screen_order=reference_period_screen_order,
        )
        if row is None:
            qa_record = make_qa_record(
                sid, "failed", True, "nominal",
                "fourier_decomp_returned_none")
            return sid, None, qa_record, None

        if not return_error:
            return sid, row, None, None

        nominal_record = dict(zip(build_fd_header(mode), row))
        error_record, qa_record = _calculate_error_payload(
            sid=sid,
            nominal_record=nominal_record,
            mode=mode,
            n_draws=error_n_draws,
            random_state=error_random_state,
            robust=error_robust,
            rms_ratio_limit=rms_ratio_limit,
            min_occupied_fraction=min_occupied_fraction,
            max_phase_gap=max_phase_gap,
        )
        return sid, [*row, *error_values(error_record)], qa_record, None
    except Exception as exc:
        qa_record = _failure_qa_record(sid, "nominal", exc)
        return sid, None, qa_record, repr(exc)

def mp_run(
    fd_output,
    ids,
    period_fit=period_fit,
    use_optim=use_optim,
    adaptive_lam=adaptive_lam,
    use_refit=use_refit,
    mode=mode_default,
    init=init,
    max_workers=8,
    chunksize=1,
    verbose=False,
    mp_context="fork",   # "fork" 권장(리눅스). 안 되면 "spawn"으로.
    return_error=False,
    error_n_draws=4000,
    error_random_state=0,
    error_robust=True,
    qa_output=None,
    rms_ratio_limit=0.7,
    min_occupied_fraction=None,
    max_phase_gap=None,
    K=None,
    harmonics=None,
    reference_periods=None,
    reference_period_window=0.0,
    reference_period_screen_order=3,
    resume=True,
):
    """
    Run decomposition.fourier_decomp(sid, ...) over many IDs with multiprocessing.
    - progress monitoring: tqdm
    - memory: prefer fork so ls_data is shared (COW)
    - performance: imap_unordered + single-writer
    - return_error=True: append only ML-facing R/phi values and HC3 errors
    - fit/HC3 failures and quality-review IDs go to a separate QA table
    - K/harmonics override the defaults for an explicit deep-refit pass
    - reference_periods maps source ID -> trusted period and skips blind LS
    - reference_period_window=0 fixes P; a positive fraction permits only a
      bounded low-order local refinement around the trusted period
    """

    from . import decomposition as decomp_mod
    fd_output = Path(fd_output)
    base_header = build_fd_header(mode)
    output_header = (
        build_fd_error_header(base_header) if return_error else base_header)
    if qa_output is None and return_error:
        qa_output = _qa_output_path(fd_output)
    if qa_output is not None:
        qa_output = Path(qa_output)

    # 1) output header
    _ensure_output_header(fd_output, output_header)
    if qa_output is not None:
        _ensure_output_header(qa_output, FD_QA_COLUMNS)

    # 2) choose mp context
    # fork가 불가능한 환경이면 spawn로 자동 fallback
    try:
        ctx = get_context(mp_context)
    except ValueError:
        ctx = get_context("spawn")

    # 3) pool init: 워커에 큰 객체를 "한 번"만 세팅
    pool = ctx.Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(decomp_mod.ls_data, decomp_mod.df_ident, decomp_mod.df_rrfit, decomp_mod.templates),
    )

    ids = list(ids)
    if resume and fd_output.exists() and fd_output.stat().st_size > 0:
        try:
            import pandas as pd
            completed = set(pd.read_csv(
                fd_output, sep=r"\s+", usecols=["ID"],
                dtype={"ID": str})["ID"].astype(str))
            ids = [sid for sid in ids if str(sid) not in completed]
        except Exception:
            # Header validation above has already protected the schema.  If a
            # partially written row prevents resume parsing, fail visibly.
            raise ValueError(
                f"Could not read completed IDs from {fd_output}; "
                "repair the partial last row or use a new output path")

    reference_map = None
    if reference_periods is not None:
        reference_map = {
            str(key): float(value) for key, value in reference_periods.items()
        }
        missing_reference = [
            str(sid) for sid in ids if str(sid) not in reference_map
        ]
        if missing_reference:
            preview = ",".join(missing_reference[:5])
            raise ValueError(
                "reference_periods is missing selected IDs: " + preview)
    n_total = len(ids)
    n_ok, n_fail, n_review, n_feature_missing = 0, 0, 0, 0
    t0 = time.time()

    # 4) single-writer in main process
    f_qa = (
        open(qa_output, "a", newline="") if qa_output is not None else None)
    with open(fd_output, "a", newline="") as f_out:
        writer = csv.writer(f_out, delimiter=" ")
        qa_writer = csv.writer(f_qa, delimiter=" ") if f_qa is not None else None

        try:
            it = pool.imap_unordered(
                _worker_call,
                (
                    (
                        sid, mode, init, period_fit, use_optim, adaptive_lam,
                        use_refit, verbose, return_error, error_n_draws,
                        error_random_state, error_robust, rms_ratio_limit,
                        min_occupied_fraction, max_phase_gap, K, harmonics,
                        (reference_map[str(sid)]
                         if reference_map is not None else None),
                        reference_period_window,
                        reference_period_screen_order,
                    )
                    for sid in ids
                ),
                chunksize=chunksize,
            )

            for sid, row, qa_record, err in tqdm(
                    it, total=n_total, desc="Fourier Decomposition"):
                if row is not None:
                    writer.writerow(row)
                    n_ok += 1
                else:
                    n_fail += 1
                    print(f"[FAIL] sid={sid} err={err}")
                if qa_record is not None and qa_record["status"] != "ok":
                    if qa_writer is not None:
                        qa_writer.writerow(qa_values(qa_record))
                    if qa_record["status"] == "failed" and row is not None:
                        n_fail += 1
                    elif qa_record["status"] == "review":
                        n_review += 1
                    elif qa_record["status"] == "feature_missing":
                        n_feature_missing += 1

        except KeyboardInterrupt:
            print("\nTerminating worker pool...")
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()
        finally:
            if f_qa is not None:
                f_qa.close()

    dt = time.time() - t0
    rate = n_total / dt if dt > 0 else float("nan")
    print(
        f"Done. total={n_total}, rows={n_ok}, fail={n_fail}, "
        f"review={n_review}, feature_missing={n_feature_missing}, "
        f"elapsed={dt:.1f}s, rate={rate:.2f} obj/s")
    return {
        "total": n_total,
        "rows": n_ok,
        "failed": n_fail,
        "review": n_review,
        "feature_missing": n_feature_missing,
        "elapsed": dt,
        "rate": rate,
        "output": str(fd_output),
        "qa_output": str(qa_output) if qa_output is not None else None,
    }


# -----------------------------------------------------------------------------
# Existing nominal catalog -> compact HC3 error catalog
# -----------------------------------------------------------------------------

def _existing_catalog_worker(args):
    (
        sid, nominal_record, input_columns, mode, n_draws, random_state,
        robust, rms_ratio_limit, min_occupied_fraction, max_phase_gap,
    ) = args
    try:
        error_record, qa_record = _calculate_error_payload(
            sid=sid,
            nominal_record=nominal_record,
            mode=mode,
            n_draws=n_draws,
            random_state=random_state,
            robust=robust,
            rms_ratio_limit=rms_ratio_limit,
            min_occupied_fraction=min_occupied_fraction,
            max_phase_gap=max_phase_gap,
        )
    except Exception as exc:
        error_record = nan_error_record()
        qa_record = make_qa_record(
            sid, "failed", True, "catalog_hc3", repr(exc), nominal_record)
    row = [nominal_record.get(name) for name in input_columns]
    row.extend(error_values(error_record))
    return sid, row, qa_record


def add_hc3_errors_to_fd_catalog(
        fd_input,
        fd_output=None,
        mode=mode_default,
        qa_output=None,
        all_ids=None,
        max_workers=8,
        chunksize=1,
        mp_context="fork",
        n_draws=4000,
        random_state=0,
        robust=True,
        rms_ratio_limit=0.7,
        min_occupied_fraction=None,
        max_phase_gap=None,
):
    """Add compact HC3 errors to an existing nominal Fourier catalog.

    The nominal fit is not repeated.  Rows with HC3 failures are retained with
    NaN error fields and are listed in the separate QA file.
    """
    import pandas as pd
    from . import decomposition as decomp_mod

    fd_input = Path(fd_input)
    if not fd_input.exists():
        raise FileNotFoundError(fd_input)
    if fd_output is None:
        fd_output = fd_input.with_name(
            f"{fd_input.stem}_with_err{fd_input.suffix}")
    fd_output = Path(fd_output)
    if qa_output is None:
        qa_output = _qa_output_path(fd_output)
    qa_output = Path(qa_output)

    frame = pd.read_csv(fd_input, sep=r"\s+", dtype={"ID": str})
    input_columns = list(frame.columns)
    expected_columns = build_fd_header(mode)
    missing_columns = [
        name for name in expected_columns if name not in input_columns]
    if missing_columns:
        raise ValueError(
            f"Nominal catalog is missing columns: {missing_columns}")
    if any(name in input_columns for name in FD_ERROR_COLUMNS):
        raise ValueError(
            "Input already contains HC3 error columns; use the nominal catalog")

    output_header = build_fd_error_header(input_columns)
    _ensure_output_header(fd_output, output_header)
    _ensure_output_header(qa_output, FD_QA_COLUMNS)

    completed = set()
    if fd_output.exists() and fd_output.stat().st_size > 0:
        completed_frame = pd.read_csv(
            fd_output, sep=r"\s+", usecols=["ID"], dtype={"ID": str})
        completed = set(completed_frame["ID"].astype(str))

    source_map = {str(sid): sid for sid in decomp_mod.ls_data.keys()}
    duplicate_ids = set(
        frame.loc[frame["ID"].duplicated(keep=False), "ID"].astype(str))
    frame = frame.drop_duplicates(subset="ID", keep="last")

    catalog_ids = set(frame["ID"].astype(str))
    if all_ids is None:
        missing_nominal = []
    else:
        all_id_keys = {str(sid) for sid in all_ids}
        missing_nominal = sorted(all_id_keys - catalog_ids)

    pending_records = []
    for record in frame.to_dict(orient="records"):
        sid_key = str(record["ID"])
        if sid_key in completed:
            continue
        sid = source_map.get(sid_key, record["ID"])
        record["ID"] = sid_key
        pending_records.append((sid, record))

    try:
        ctx = get_context(mp_context)
    except ValueError:
        ctx = get_context("spawn")
    pool = ctx.Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(
            decomp_mod.ls_data, decomp_mod.df_ident,
            decomp_mod.df_rrfit, decomp_mod.templates,
        ),
    )

    n_rows = n_failed = n_review = n_feature_missing = 0
    with open(fd_output, "a", newline="") as f_out, open(
            qa_output, "a", newline="") as f_qa:
        writer = csv.writer(f_out, delimiter=" ")
        qa_writer = csv.writer(f_qa, delimiter=" ")

        for sid_key in sorted(duplicate_ids):
            qa_writer.writerow(qa_values(make_qa_record(
                sid_key, "review", True, "catalog",
                "duplicate_nominal_row")))
        for sid_key in missing_nominal:
            qa_writer.writerow(qa_values(make_qa_record(
                sid_key, "failed", True, "nominal",
                "missing_from_nominal_catalog")))

        try:
            iterator = pool.imap_unordered(
                _existing_catalog_worker,
                (
                    (
                        sid, record, input_columns, mode, n_draws,
                        random_state, robust, rms_ratio_limit,
                        min_occupied_fraction, max_phase_gap,
                    )
                    for sid, record in pending_records
                ),
                chunksize=chunksize,
            )
            for sid, row, qa_record in tqdm(
                    iterator, total=len(pending_records),
                    desc="HC3 catalog augmentation"):
                writer.writerow(row)
                n_rows += 1
                if qa_record is not None and qa_record["status"] != "ok":
                    qa_writer.writerow(qa_values(qa_record))
                    if qa_record["status"] == "failed":
                        n_failed += 1
                    elif qa_record["status"] == "review":
                        n_review += 1
                    elif qa_record["status"] == "feature_missing":
                        n_feature_missing += 1
        except KeyboardInterrupt:
            print("\nTerminating HC3 worker pool...")
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()

    print(
        f"HC3 catalog done. rows={n_rows}, failed={n_failed}, "
        f"review={n_review}, feature_missing={n_feature_missing}, "
        f"missing_nominal={len(missing_nominal)}")
    return {
        "rows": n_rows,
        "failed": n_failed,
        "review": n_review,
        "feature_missing": n_feature_missing,
        "missing_nominal": len(missing_nominal),
        "output": str(fd_output),
        "qa_output": str(qa_output),
    }


# -----------------------------------------------------------------------------
# Reference-free period/alias audit
# -----------------------------------------------------------------------------

def _period_audit_worker(args):
    sid, nominal_record, mode, audit_kwargs = args
    try:
        record = assess_period_stability(
            sid=sid, nominal_record=nominal_record, mode=mode,
            **audit_kwargs)
    except Exception as exc:
        record = {
            "ID": str(sid), "status": "review", "retryable": 1,
            "reason": f"period_audit_exception:{repr(exc)}",
        }
    return record


def audit_period_stability_catalog(
        fd_input, audit_output=None, mode=mode_default, max_workers=8,
        chunksize=8, mp_context="fork", deep_k=15, harmonic_depth=4,
        screen_order=3, better_score_threshold=10.0,
        ambiguity_threshold=2.0, minimum_cycles=2.0, overwrite=False,
        resume=True):
    """Screen every fitted source for a plausible reference-free alias.

    This is a selection audit only.  It writes no replacement periods and
    never changes the input catalog.
    """
    import pandas as pd
    from . import decomposition as decomp_mod

    fd_input = Path(fd_input)
    if audit_output is None:
        audit_output = fd_input.with_name(
            f"{fd_input.stem}_period_audit.dat")
    audit_output = Path(audit_output)
    completed = set()
    if audit_output.exists():
        if overwrite:
            audit_output.unlink()
        elif resume:
            existing_header = _header_from_file(audit_output)
            if existing_header != list(PERIOD_AUDIT_COLUMNS):
                raise ValueError(f"Header mismatch for {audit_output}")
            completed = set(pd.read_csv(
                audit_output, sep=r"\s+", usecols=["ID"],
                dtype={"ID": str})["ID"].astype(str))
        else:
            raise FileExistsError(
                f"Refusing to overwrite {audit_output}; use overwrite=True")
        audit_output.unlink()

    frame = pd.read_csv(fd_input, sep=r"\s+", dtype={"ID": str})
    if frame["ID"].duplicated().any():
        raise ValueError("fd_input contains duplicate IDs")
    source_map = {str(sid): sid for sid in decomp_mod.ls_data.keys()}
    records = []
    for nominal in frame.to_dict(orient="records"):
        sid_key = str(nominal["ID"])
        if sid_key in completed:
            continue
        records.append((source_map.get(sid_key, sid_key), nominal))

    audit_kwargs = {
        "deep_k": int(deep_k),
        "harmonic_depth": int(harmonic_depth),
        "screen_order": int(screen_order),
        "better_score_threshold": float(better_score_threshold),
        "ambiguity_threshold": float(ambiguity_threshold),
        "minimum_cycles": float(minimum_cycles),
    }
    try:
        ctx = get_context(mp_context)
    except ValueError:
        ctx = get_context("spawn")
    pool = ctx.Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(
            decomp_mod.ls_data, decomp_mod.df_ident,
            decomp_mod.df_rrfit, decomp_mod.templates,
        ),
    )
    _ensure_output_header(audit_output, PERIOD_AUDIT_COLUMNS)
    n_review = n_failed = 0
    with open(audit_output, "a", newline="") as handle:
        writer = csv.writer(handle, delimiter=" ")
        try:
            iterator = pool.imap_unordered(
                _period_audit_worker,
                ((sid, nominal, mode, audit_kwargs)
                 for sid, nominal in records),
                chunksize=chunksize,
            )
            for record in tqdm(
                    iterator, total=len(records), desc="Period stability audit"):
                writer.writerow(period_audit_values(record))
                n_review += int(record.get("status") == "review")
                n_failed += int(record.get("status") == "failed")
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()
    return {
        "rows": len(records), "review": n_review, "failed": n_failed,
        "output": str(audit_output),
    }


def thread_run(fd_output, ids, period_fit = period_fit,
               use_optim = use_optim, adaptive_lam = adaptive_lam, use_refit = use_refit,
               mode = mode_default, init = init, max_workers = 8):
    """
    Run decomposition.fourier_decomp(sid, ...) over many IDs with threads.
    Returns: list of result rows (one per successful sid).
    """
    from . import decomposition as decomp_mod

    if not fd_output.exists():
        with open(fd_output, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            columns = build_fd_header(mode)
            writer.writerow(columns)

    # pool
    manager = Manager()
    lock = manager.Lock()

    def callback(row):
        if row is not None:
            with lock:
                writer.writerow(row)
        pbar.update(1)

    # file
    f_out = open(fd_output, 'a', newline='')
    writer = csv.writer(f_out, delimiter=' ')
    pbar = tqdm(total = len(ids), desc = 'Fourier Decomposition', position = 0)

    # multiprocessing
    ctx = get_context("spawn")
    pool = ctx.Pool(processes = max_workers,
                    initializer=_init_worker,
                    initargs=(decomp_mod.ls_data, decomp_mod.df_ident,
                              decomp_mod.df_rrfit, decomp_mod.templates))
    try:
        # asychronous processing
        for sid in ids:
            pool.apply_async(decomp_mod.fourier_decomp, 
                             args =(sid, mode, init, period_fit, 
                                    use_optim, adaptive_lam, use_refit, False),
                            callback = callback)
        pool.close()
        pool.join() # close / return pool to os

    except KeyboardInterrupt:
        print("\nTerminating worker pool...")
        pool.terminate() # terminate pols
        pool.join()

    finally:
        pbar.close() 
        f_out.close() # close file pointer


def run_manifest_refits(
    manifest: str | Path | pd.DataFrame,
    base_catalog: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    max_workers: int = 8,
    chunksize: int = 1,
    reference_period_window: float = 0.0,
    return_error: bool = True,
    error_n_draws: int = 4000,
    error_random_state: int = 0,
    overwrite_request: bool = False,
) -> dict[str, Any]:
    """Refit explicit manifest jobs, guarded by a resumable request hash."""

    ids, period_map = revision_jobs_from_manifest(
        manifest, base_catalog=base_catalog)
    if not ids:
        raise ValueError("No adopt_period/refit_same_period jobs in manifest")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "refit_request_manifest.json"
    output_path = output_dir / (
        "gaia_fd_revision_refit_with_err.dat"
        if return_error else "gaia_fd_revision_refit.dat")
    qa_path = output_dir / "gaia_fd_revision_refit_qa.dat"
    request = {
        "schema": "gaia-fd-manual-refit-request-v1",
        "base_catalog": (
            str(Path(base_catalog).resolve())
            if not isinstance(base_catalog, pd.DataFrame) else "dataframe"),
        "base_sha256": (
            file_sha256(base_catalog)
            if not isinstance(base_catalog, pd.DataFrame) else None),
        "periods": period_map,
        "reference_period_window": float(reference_period_window),
        "return_error": bool(return_error),
        "error_n_draws": int(error_n_draws),
        "error_random_state": int(error_random_state),
    }
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
    request["request_sha256"] = request_hash
    if request_path.exists() and not overwrite_request:
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous.get("request_sha256") != request_hash:
            raise ValueError(
                "Existing refit output belongs to a different decision manifest; "
                "use a new output directory")
    else:
        request["created_utc"] = datetime.now(timezone.utc).replace(
            microsecond=0).isoformat()
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    result = mp_run(
        output_path, ids, mode="gaia", init="lsq", period_fit=False,
        use_optim=True, adaptive_lam=True, use_refit=True,
        max_workers=max_workers, chunksize=chunksize, mp_context="spawn",
        return_error=return_error, error_n_draws=error_n_draws,
        error_random_state=error_random_state, qa_output=qa_path,
        reference_periods=period_map,
        reference_period_window=reference_period_window,
        resume=True,
    )
    result.update({
        "request_manifest": str(request_path),
        "request_sha256": request_hash,
    })
    return result
