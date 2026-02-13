from multiprocessing import Pool, Lock, Manager
from tqdm.notebook import tqdm
import csv

from .params import M_MAX

columns = ['ID','pulsation', 'N_I','sig_I','rms_Ires',
           'Zmax','P(LS)','chi2','P','E','phi_rise', 'M_fit', # <-- M_fit 컬럼 추가
           'I0','Amp_I']+\
[f'A{i}' for i in range(1, M_MAX + 1)] + [f'Q{i}' for i in range(1, M_MAX + 1)] + ['flag']

def thread_run(fd_output, decomp_mod, ids, period_fit = False, 
               max_workers = 8):
    """
    Run decomposition.fourier_decomp(sid, ...) over many IDs with threads.
    Returns: list of result rows (one per successful sid).
    """
    fourier_decomp = decomp_mod.fourier_decomp
    
    if not fd_output.exists():
        with open(fd_output, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
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
    pool = Pool(processes = max_workers)
    try:
        # asychronous processing
        for sid in ids:
            pool.apply_async(fourier_decomp, args = (sid, period_fit,),
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