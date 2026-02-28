import importlib
from multiprocessing import Pool, Lock, Manager, get_context
from tqdm.notebook import tqdm
import csv

from .params import period_fit
from .IO import get_data_config

# --- dynamic header based on activated bands ---
def _build_fd_header(mode = None):
    """
    Build output header for Fourier decomposition using only activated bands.
    Output row format must match decomposition.fourier_decomp().

    Expected row layout (as in decomposition.fourier_decomp):
      [sid, pulsation, *N, *sig, *rms, Zmax, P0, chi2, P, E, phi_rise, M_fit, *theta_params_out, flag]
    where:
      N/sig/rms are per activated band
      theta_params_out = [*m0, *amp, *A(1..M_MAX), *Q(1..M_MAX)]
        but only for activated bands in m0/amp
    """
    from .params import M_MAX

    cfg = get_data_config(mode)
    active_idx = list(cfg.activated_bands)
    active_filters = [str(cfg.filters[i]) for i in active_idx]

    cols = []
    cols += ["ID", "pulsation"]

    # Per-band stats (only activated bands)
    cols += [f"N_{b}" for b in active_filters]
    cols += [f"sig_{b}" for b in active_filters]
    cols += [f"rms_{b}" for b in active_filters]

    # Period / fit summary
    cols += ["Zmax", "P0", "chi2", "P", "E", "phi_rise", "M_fit"]

    # Theta params (match decomposition.py theta_params_out ordering)
    cols += [f"m0_{b}" for b in active_filters]
    cols += [f"amp_{b}" for b in active_filters]

    # Fourier series params
    cols += [f"A{j}" for j in range(1, M_MAX + 1)]
    cols += [f"Q{j}" for j in range(1, M_MAX + 1)]

    cols += ["flag"]
    return cols

def _init_worker(ls_data, df_ident):
    """Runs once per worker process."""
    from . import decomposition as decomp_mod 
    decomp_mod.ls_data = ls_data
    decomp_mod.df_ident = df_ident

def thread_run(fd_output, ids, period_fit = period_fit, 
               max_workers = 8):
    """
    Run decomposition.fourier_decomp(sid, ...) over many IDs with threads.
    Returns: list of result rows (one per successful sid).
    """
    from . import decomposition as decomp_mod

    if not fd_output.exists():
        with open(fd_output, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            columns = _build_fd_header()
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
                    initargs=(decomp_mod.ls_data, decomp_mod.df_ident))
    try:
        # asychronous processing
        for sid in ids:
            pool.apply_async(decomp_mod.fourier_decomp, args = (sid, period_fit,),
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