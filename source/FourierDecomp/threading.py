import importlib
from multiprocessing import Pool, Lock, Manager, get_context
from tqdm.notebook import tqdm
import csv
import time

from .params import period_fit, use_optim, mode_default, init
from .IO import get_data_config, build_fd_header

def _init_worker(ls_data, df_ident, df_rrfit, templates):
    """Runs once per worker process."""
    from . import decomposition as decomp_mod 
    decomp_mod.ls_data = ls_data
    decomp_mod.df_ident = df_ident
    decomp_mod.df_rrfit = df_rrfit
    decomp_mod.templates = templates


def _worker_call(args):
    """Picklable wrapper for imap."""
    sid, period_fit, use_optim, mode, init, verbose = args
    from . import decomposition as decomp_mod
    try:
        row = decomp_mod.fourier_decomp(
            sid,
            period_fit=period_fit,
            use_optim=use_optim,
            mode=mode,
            init=init,
            verbose=verbose,
        )
        return (sid, row, None)
    except Exception as e:
        return (sid, None, repr(e))

def mp_run(
    fd_output,
    ids,
    period_fit=period_fit,
    use_optim=use_optim,
    mode=mode_default,
    init=init,
    max_workers=8,
    chunksize=64,
    verbose=False,
    mp_context="fork",   # "fork" 권장(리눅스). 안 되면 "spawn"으로.
):
    """
    Run decomposition.fourier_decomp(sid, ...) over many IDs with multiprocessing.
    - progress monitoring: tqdm
    - memory: prefer fork so ls_data is shared (COW)
    - performance: imap_unordered + chunksize + single-writer
    """

    from . import decomposition as decomp_mod

    # 1) output header
    if not fd_output.exists():
        with open(fd_output, "w", newline="") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerow(build_fd_header())

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

    n_total = len(ids)
    n_ok, n_fail = 0, 0
    t0 = time.time()

    # 4) single-writer in main process
    with open(fd_output, "a", newline="") as f_out:
        writer = csv.writer(f_out, delimiter=" ")

        try:
            it = pool.imap_unordered(
                _worker_call,
                ((sid, period_fit, use_optim, mode, init, verbose) for sid in ids),
                chunksize=chunksize,
            )

            for sid, row, err in tqdm(it, total=n_total, desc="Fourier Decomposition"):
                if row is not None:
                    writer.writerow(row)
                    n_ok += 1
                else:
                    n_fail += 1
                    # 너무 많이 찍히면 느려지니 필요 시 일부만 출력
                    print(f"[FAIL] sid={sid} err={err}")

        except KeyboardInterrupt:
            print("\nTerminating worker pool...")
            pool.terminate()
        finally:
            pool.close()
            pool.join()

    dt = time.time() - t0
    rate = n_total / dt if dt > 0 else float("nan")
    print(f"Done. total={n_total}, ok={n_ok}, fail={n_fail}, elapsed={dt:.1f}s, rate={rate:.2f} obj/s")

def thread_run(fd_output, ids, period_fit = period_fit, use_optim = use_optim,  
               mode = mode_default, init = init, max_workers = 8):
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
                    initargs=(decomp_mod.ls_data, decomp_mod.df_ident,
                              decomp_mod.df_rrfit, decomp_mod.templates))
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