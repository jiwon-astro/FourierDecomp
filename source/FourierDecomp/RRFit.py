from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
from astropy.table import Table, vstack, unique, join

from dataclasses import dataclass
from typing import Union
from concurrent.futures import ProcessPoolExecutor
import subprocess
import tempfile

from FourierDecomp.IO import get_data_config, prepare_fitlc
from FourierDecomp.period_finder import period_fit_boundary_search

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
        # presuming the order of band prefixs are identical to input filter list order
        return np.arange(len(self.filters)) 
    @property
    def bands(self):
        return [int(self.prefixs[self.filters==b]) for b in self.selected_bands]
    @property
    def bandpair(self):
        # photometric band pairs for simultaneous fit 
        return "+".join(self.selected_bands)
    @property
    def job_id(self):
        return f"{self.sid}_{self.bandpair}_{self.window_idx:02d}"

def write_rrfit_inputs(job, workdir):
    # open rrfit.param/fitlc_list/lomb_scargle.txt file, and write the inputs
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
        p_bounds = [p_bounds] # ensure dimension
    else:
        raise ValueError("unsupported p_bounds types")
    
    jobs = []
    for iw, logP_bound in enumerate(logP_bounds):
        logP0 = np.mean(logP_bound)
        P0 = 10**logP0
        pmin = max(params.pmin, 10**(logP_bound[0]-logP_tol)) # add padding
        pmax = min(params.pmax, 10**(logP_bound[1]+logP_tol))
        
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

def parse_rrfit_outputs(fpath):
    # fpath: RRFit output file path
    if not fpath.exists(): return None
    try:  
        tbl = Table.read(fpath, format='ascii')
    except Exception: 
        return None
    if len(tbl)==0: return None
    return dict(tbl[-1])

def run_rrfit_job(job, rrfit_exe, base_workdir=None, is_test=False):
    if base_workdir is None:
        rrfit_exe = Path(rrfit_exe)
        base_workdir = rrfit_exe.parent / "temp"
    base_workdir.mkdir(parents=True, exist_ok=True)

    # run multiple rrfit jobs in the temporary directory
    # e.g.) job A -> rrfit_sid_1, job B -> rrfit_sid_2,...
    with tempfile.TemporaryDirectory(prefix=f"rrfit_{job.sid}_",
                                     dir=base_workdir) as td:
        if is_test:
            workdir = base_workdir / "test" # fixed directory
            workdir.mkdir(parents=True, exist_ok=True) 
        else: workdir = Path(td)
        write_rrfit_inputs(job, workdir)

        #subprocess - run shell command
        proc = subprocess.run([rrfit_exe], cwd=workdir, 
                              capture_output=True, text=True) # return string
        outname = f"rrfit_{job.bandpair}.out"
        # single result for single job -> read last column
        row = parse_rrfit_outputs(workdir / outname) 
        return {
            "sid": job.sid,
            "job_id": job.job_id,
            "bandpair": job.bandpair,
            "P0": job.P0,
            "pmin": job.pmin,
            "pmax": job.pmax,
            "p0flag": job.p0flag,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "result": row,
        }
    
def write_source_rrfit_results(outdir, sid, results,
                               P0_LS=np.nan, Zmax=np.nan, logP_bounds=None, alias_freqs=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # meta data
    meta = Table({"sid": [sid], "P0_LS": [P0_LS], "Zmax": [Zmax],
                  "alias_freqs": [alias_freqs.tolist()], "logP_bounds": [logP_bounds],})
    meta_fname = outdir / f"rrfit_{sid}_meta.ecsv"
    meta.write(meta_fname, format='ascii.ecsv', overwrite=True)

    # summary table
    rows = []
    for r in results:
        # collecting the results from the individual fitting for each window
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
            # unpack results (read from RRFit output file)
            for k, v in r["result"].items(): row[k] = v
        rows.append(row)
    if len(rows)>0: tbl = Table(rows)
    else: tbl = Table()
    summary_fname = outdir / f"rrfit_{sid}.summary"
    tbl.write(summary_fname, format='ascii.basic')
    return summary_fname, meta_fname

def run_rrfit_single(sid, rrfit_exe, outdir, mode=None, fitlc_path=None, ls_data=None, workdir=None, 
              bandpairs=(("g", "bp"), ("g", "rp")), 
              max_workers_job=6, is_test=False, **kwargs):
    if workdir is None:
        workdir = Path(rrfit_exe).parent / "temp"

    P0_LS, Zmax, jobs, alias_freqs, logP_bounds = build_rrfit_jobs(
        sid=sid, mode=mode,
        fitlc_path=fitlc_path, ls_data=ls_data, workdir=workdir,
        bandpairs=bandpairs, **kwargs
    )

    results = []
    if len(jobs) > 0:
        with ProcessPoolExecutor(max_workers=max_workers_job) as ex:
            futs = [ex.submit(run_rrfit_job, job, rrfit_exe, workdir, is_test) for job in jobs]
            for fut in futs: results.append(fut.result())

    summary_fpath, meta_fpath = write_source_rrfit_results(outdir, sid, results, 
                                            P0_LS=P0_LS, Zmax=Zmax, logP_bounds=logP_bounds, alias_freqs=alias_freqs)
    return {"sid": sid, "n_jobs": len(jobs), "summary": str(summary_fpath), "meta": str(meta_fpath)}