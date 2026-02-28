import importlib
from multiprocessing import Pool, Lock, Manager, get_context
from tqdm.notebook import tqdm
import csv

from .params import period_fit
from .IO import get_data_config, build_fd_header

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
            columns = build_fd_header()
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