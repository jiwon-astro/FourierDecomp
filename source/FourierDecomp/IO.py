import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor # Threading
from .LC import filters, prefixs, ident_names, phot_names

def load_chunk(source_ids, phot_cat_dir):
    data = {}
    # extract epoch photometry data
    for sid in source_ids:
        dfs = []
        for band in filters:
            phot_path = phot_cat_dir / band / f'{sid}.dat'
            if phot_path.exists(): 
                phot_tmp = pd.read_csv(phot_path, sep = ' ', header = None,
                                  names = phot_names)
                phot_tmp['band'] = band
                dfs.append(phot_tmp) 
                # 누락된 band는 무시.
        if dfs: data[sid] = pd.concat(dfs) # stacking data
    return data

def data_loader(ls_ident, phot_dir, max_workers = 12, chunk_size = 200):
    # max workers = # cores in cpu * 1.5
    ls_data = {}
    cat = []
    for ident_path in ls_ident:
        n_infer = 100
        # identification catalog
        if ident_path.stem in ['blg_cep_ident']: n_infer = 10
        df_ident_cat = pd.read_fwf(ident_path, header = None, names = ident_names, infer_nrows = n_infer)
        cat_ids      = df_ident_cat['ID'] # ids
        cat_name     = (ident_path.stem).strip('_ident') # catalog name
        phot_cat_dir = phot_dir / cat_name # catalog specific photometry data path
        cat.append(df_ident_cat)
        # Threading
        nmax = len(cat_ids)
        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            for i in tqdm(range(0, nmax, chunk_size), desc = f'{cat_name}'):
                batch_ids = cat_ids[i:min(i+chunk_size, nmax)]
                future = executor.submit(load_chunk, batch_ids, phot_cat_dir)
                ls_data.update(future.result())
    return pd.concat(cat).reset_index(drop=True), ls_data

def wire_globals(decomp_mod,
                 ls_data,
                 df_ident):
    """
    Attach globals into `decomposition` module namespace.
    """
    decomp_mod.ls_data = ls_data
    decomp_mod.df_ident = df_ident
