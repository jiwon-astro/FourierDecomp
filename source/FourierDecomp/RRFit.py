import os
import signal
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Union
from tqdm.notebook import tqdm
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

    # Lomb-Scargle / Window-alias period window
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
    elif isinstance(p_bounds,(list, tuple)) and len(p_bounds)==2:
        logP_bounds = [(np.log10(p_bounds[0]), np.log10(p_bounds[1]))] # ensure the dimension
        alias_freqs = []
    else:
        raise ValueError("unsupported p_bounds types")
    
    jobs = []
    logP_bounds_full = [(np.log10(params.pmin), np.log10(params.pmax))] + logP_bounds # global period scan
    for iw, logP_bound in enumerate(logP_bounds_full):
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
            "n_jobs": len(jobs)}

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
                          overwrite=True, posfixs="", **kwargs):
    """
    Create RRFit jobs and Lomb-Scargle metadata for all sources.
    - multiprocessing
    - workdir: rrfit_jobs_<sid>.json / rrfit_meta_<sid>.ecsv
    - outdir: rrfit_plan.ecsv (summary)
    """
    workdir = Path(workdir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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
            
        tasks.append({"source_lc": source_lc, "workdir": workdir, "mode": mode, "bandpairs": bandpairs,
                      "overwrite": overwrite, "save":True, "posfixs":posfixs, **kwargs})
    
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(build_rrfit_jobs, **t): t for t in tasks}
        for fut in tqdm(futs, total=len(futs), desc="Constructing RRFit jobs"):
            task = futs[fut]; sid = task.get("sid")
            try:
                rows.append(fut.result())
            except Exception as e:
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
        sid_mask = np.isin(sids_full, sids)
        job_files = job_files_full[sid_mask]
        meta_files = meta_files_full[sid_mask]
    else: 
        sids = sids_full
        job_files = job_files_full
        meta_files = meta_files_full

    job_pool = []
    meta_pool = {}

    for sid, job_fpath, meta_fpath in zip(sids, job_files, meta_files):
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
                }
        else:
            meta_pool[sid] = {
                "sid": sid,
                "P0_LS": np.nan,
                "Zmax": np.nan,
                "alias_freqs": [],
                "logP_bounds": [],
                "n_jobs": len(jobs),
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
def run_rrfit(sids, rrfit_exe, outdir, workdir=None, mode=None, fitlc_list=None, ls_data=None, 
              bandpairs=(("g", "bp"), ("g", "rp")), max_workers=8, is_test=False, timeout=300,
              posfixs="", **kwargs):
    
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if workdir is None:
        workdir = Path(rrfit_exe).parent / "temp"
    
    # build / load jobs
    if posfixs and (not posfixs.startswith("_")): posfixs = "_" + posfixs
    plan_fpath = outdir / f"rrfit_plan{posfixs}.dat"
    if not plan_fpath.exists():
        plan_fpath = build_rrfit_plan(sids, workdir, outdir, mode=mode, 
                                      ls_data=ls_data, fitlc_list=fitlc_list,
                                      bandpairs=bandpairs, max_workers=max_workers, 
                                      overwrite=True, posfixs=posfixs, **kwargs)
        
    job_pool, meta_pool = load_rrfit_plan(plan_fpath, sids=sids)
    
    if not (job_pool and meta_pool):
        raise ValueError("Invalid Job data list or metadata list")

    # track source-wise job status
    results_pool = defaultdict(list)
    n_done_pool = defaultdict(int)
    n_total_pool = {sid: meta_pool[sid]["n_jobs"] for sid in sids}
    source_written = set()

    log_rows = []
    source_rows = []

    # run all jobs
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(run_rrfit_job, job, rrfit_exe, workdir, is_test, timeout): job
                     for job in job_pool}
            with tqdm(total=len(sids), desc='Sources') as pbar, \
                 tqdm(total=len(job_pool), desc='RRFit Jobs') as job_pbar:
                for fut in as_completed(futs):
                    job = futs[fut]
                    sid = job.sid
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

                    results_pool[sid].append(r)
                    n_done_pool[sid] += 1

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
                        "stderr": r["stderr"][:500] if isinstance(r["stderr"], str) else "",
                    })

                    # If all jobs have finished for a given source
                    if (sid not in source_written) and (n_done_pool[sid]>=n_total_pool[sid]):
                        # Collect results from the separate jobs of a given source
                        meta = meta_pool[sid]
                        results = results_pool.get(sid,[]) 
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
                        source_written.add(sid)
                        pbar.update(1) 

                    # update/save logs  
                    job_log_tbl = Table(log_rows)
                    job_log_tbl.write(outdir / "rrfit_job_log.ecsv", 
                                    format="ascii.ecsv", overwrite=True)
                    if source_rows:
                        source_log_tbl = Table(source_rows)
                        source_log_tbl.write(outdir / "rrfit_source_log.dat", 
                                            format="ascii.basic", overwrite=True)
                        
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Terminating active rrfit.e processes...")
        kill_all_active_processes()
        # Completed sources already have their own summary/meta files -> partial output
        if log_rows:
            job_log_tbl = Table(log_rows)
            job_log_tbl.write(outdir / "rrfit_job_log.ecsv", 
                              format="ascii.ecsv", overwrite=True)
        if source_rows:
            source_log_tbl = Table(source_rows)
            source_log_tbl.write(outdir / "rrfit_source_log.dat", 
                                    format="ascii.basic", overwrite=True)
        raise

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