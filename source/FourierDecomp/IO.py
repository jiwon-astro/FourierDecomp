
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor  # Threading
from pathlib import Path
from astropy.table import Table, vstack

# ============================================================================
# data configuration
# ============================================================================

@dataclass(frozen=True)
class DataConfig:
    mode: str
    filters: np.ndarray          # e.g. ['V','I'] or ['g','bp','rp']
    prefixs: np.ndarray          # e.g. [0,1] or [0,1,2]
    activated_bands: list        # indices in prefixs
    n_bands: int
    lc_colors: list
    lc_markers: list

def get_data_config(mode: Optional[str] = None) -> DataConfig:
    """
    params.py: data_class(=mode) -> filters/plot style

    mode:
      - None: params.data_class
      - 'ogle' or 'gaia'
    """
    from .params import mode_default, DATA_CONFIGS  # 런타임에 읽어오면 (노트북에서 params 수정 후) 반영이 쉬움
    
    if mode is None: mode = mode_default
    mode = mode.lower().strip()
    base = DATA_CONFIGS[mode]

    if mode not in ("ogle", "gaia"):
        raise ValueError(f"Unknown data_class/mode: {mode}")

    flt = np.array(base['filters'], dtype=object)
    pfx = np.array(base['prefixs'], dtype=int)
    act = np.array(base['activated_bands'], dtype=int)
    n_b = len(act)

    colors = list(base['lc_colors'])
    markers = list(base['lc_markers'])

    return DataConfig(
        mode=mode,
        filters=flt,
        prefixs=pfx,
        activated_bands=act,
        n_bands=n_b,
        lc_colors=colors,
        lc_markers=markers,
    )

# =============================================================================
# epoch photometry I/O 
# =============================================================================
# --- LC identifier ---
ident_names = ['ID','pulsation','RA','Dec','OGLE-IV ID','OGLE-III ID','OGLE-II ID','other']
phot_names = ['t','mag','emag']

def _load_chunk_ogle(source_ids, phot_dir, 
                    filters=("V","I"), phot_names=phot_names):
    data = {}
    if phot_dir is None:
        raise ValueError("phot_cat_dir is required for mode='ogle'")
    for sid in source_ids: # e.g.) OGLE-LMC-ACEP-001
        tbls = []
        for b in filters:
            fname = Path(phot_dir) / b / f"{sid}.dat"
            if not fname.exists(): continue
            tab = Table.read(str(fname), format="ascii.no_header", 
                                names=phot_names, guess=False,)
            tab['band'] = b
            tbls.append(tab)
        if tbls:
            data[sid] = vstack(tbls, metadata_conflicts="silent")
    return data

def _load_chunk_gaia(source_ids, phot_dir):
    data = {}
    if phot_dir is None:
        raise ValueError("phot_dir is required")
    for sid in source_ids:
        sid = int(sid)
        fname = Path(phot_dir) / f"epphot_DR3_{sid}.ecsv"
        if not fname.exists(): continue
        try:
            tab = Table.read(fname, format="ascii.ecsv")
            data[sid] = tab
        except Exception:
            continue
    return data

def _load_chunk(source_ids, phot_dir, mode="ogle",**kwargs):
    if mode == "ogle":
        return _load_chunk_ogle(source_ids, phot_dir, **kwargs)
    elif mode == "gaia":
        return _load_chunk_gaia(source_ids, phot_dir, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# --------------
# identifiers
# -------------
def _read_ident_gaia(query_fpath):
    # load gaia query data
    tab = Table.read(query_fpath, format='ascii.ecsv')
    return tab

def _read_ident_ogle(ident_fpath_list):
    cat = []
    jobs = [] # (cat_name, cat_ids)
    for ident_fpath in ident_fpath_list:
        ident_fpath = Path(ident_fpath)
        n_infer = 100
        # identification catalog
        if ident_fpath.stem in ['blg_cep_ident']: n_infer = 10
        df_ident_cat = pd.read_fwf(ident_fpath, header = None, names = ident_names, infer_nrows = n_infer)
        cat_ids      = df_ident_cat['ID'] # ids
        cat_name     = (ident_fpath.stem).strip('_ident') # catalog name

        cat.append(df_ident_cat)
        jobs.append((cat_name, cat_ids)) # using it to load catalog specific epoch photometry data
    
    # concatnation
    df_ident = pd.concat(cat).reset_index(drop=True) if cat else pd.DataFrame(columns=ident_names)
    return df_ident, jobs

# --------------------
# Bulk loader
# --------------------
def _chunk_loader_threading(source_ids, phot_dir, mode, desc = None, 
                            max_workers=12, chunk_size=200):
    nmax = len(source_ids)
    dl = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, nmax, chunk_size):
            batch_ids = source_ids[i:min(i + chunk_size, nmax)]
            futures.append(executor.submit(_load_chunk, batch_ids, phot_dir, mode=mode))
        for fut in tqdm(futures, desc=desc):
            dl.update(fut.result())
    return dl

def data_loader(ident_fpath, phot_dir, mode=None, max_workers=12, chunk_size=200):
    #if (ident_fpath, str) or (ident_fpath, Path):
    #    ident_fpath = [ident_fpath]
    phot_dir = Path(phot_dir)
    if mode is None: mode = get_data_config().mode
    # ---- OGLE ----
    if mode == "ogle":
        df_ident, jobs = _read_ident_ogle(ident_fpath) #list of identifier paths
        dl = {}
        for cat_name, cat_ids in jobs:
            print(f"Load data from {phot_dir/cat_name}")
            dl_cat = _chunk_loader_threading(
                cat_ids,
                phot_dir / cat_name,
                mode="ogle",
                max_workers=max_workers,
                chunk_size=chunk_size,
                desc=f"{cat_name}")
            dl.update(dl_cat)

        return df_ident, dl

    # ---- GAIA ----
    elif mode=="gaia":
        df_ident = _read_ident_gaia(ident_fpath) # single file path
        source_ids = df_ident["SOURCE_ID"].astype(int).tolist()

        dl = _chunk_loader_threading(
            source_ids,
            phot_dir=phot_dir,
            mode="gaia",
            max_workers=max_workers,
            chunk_size=max(chunk_size, 500),  # 파일 열기 위주라 크게 잡는 편이 효율적일 때 많음
            desc="gaia_epoch_phot",
        )
    return df_ident, dl


def wire_globals(module, ls_data, df_ident):
    """Attach globals into given module namespace."""
    module.ls_data = ls_data
    module.df_ident = df_ident

# =============================================================================
# Gaia epoch photometry support (DR3 ECSV)
# =============================================================================

def gaia_ZP_cal(flux, ferr, band):
    """
    Zeropoint calibration (VEGAMAG), following Gaia documentation.
    Returns calibrated mag and mag_err.
    """
    ZP = {'g': 25.6874, 'bp': 25.3385, 'rp': 24.7479}
    ZPerr = {'g': 0.0028, 'bp': 0.0028, 'rp': 0.0028}
    flux = np.asarray(flux, dtype=float)
    ferr = np.asarray(ferr, dtype=float)
    flux = np.maximum(flux, 1e-30)
    mag = -2.5 * np.log10(flux) + ZP[band]
    mag_err = np.sqrt((-2.5 * ferr / flux) ** 2 + ZPerr[band] ** 2)
    return mag, mag_err


def extract_gaia_band_data(dl_dict, source_id, band, monitor=True):
    """
    Extract (time, mag, mag_err) for a Gaia band from a loaded astropy.Table.

    Expected columns (per your notebook):
      - variability_flag_{band}_reject
      - {band}_mag  (or g_transit_mag for g)
      - {band}_flux, {band}_flux_error (or g_transit_flux, g_transit_flux_error)
      - time key:
          g: g_transit_time
          bp/rp: {band}_obs_time
    """
    tab = dl_dict[int(source_id)]
    vari_flag_mask = np.array(tab[f"variability_flag_{band}_reject"])

    if band == "g":
        timekey = "g_transit_time"
        photkey = "g_transit_mag" if "g_transit_mag" in tab.colnames else "g_mag"
        fluxkey = "g_transit_flux" if "g_transit_flux" in tab.colnames else "g_flux"
        errkey = "g_transit_flux_error" if "g_transit_flux_error" in tab.colnames else "g_flux_error"
    else:
        timekey = f"{band}_obs_time"
        photkey = f"{band}_mag"
        fluxkey = f"{band}_flux"
        errkey = f"{band}_flux_error"

    phot = tab[photkey]
    if hasattr(phot, "mask"):
        mask = np.array(phot.mask) | np.array(vari_flag_mask)
        mag = np.array(phot[~mask])
        flux = np.array(tab[fluxkey][~mask])
        ferr = np.array(tab[errkey][~mask])
        time = np.array(tab[timekey][~mask])
    else:
        mask = np.array(vari_flag_mask)
        mag = np.array(phot[~mask])
        flux = np.array(tab[fluxkey][~mask])
        ferr = np.array(tab[errkey][~mask])
        time = np.array(tab[timekey][~mask])

    if monitor:
        print(f"{band} / rejected = {np.sum(mask)} / {len(mask)}")

    # Use flux errors to compute mag_err (more stable than raw mag_error columns when absent)
    _, mag_err = gaia_ZP_cal(flux, ferr, band)
    return time.astype(float), np.asarray(mag, dtype=float), np.asarray(mag_err, dtype=float)


def gaia_epoch_arrays(dl_dict, source_id, filters=("g", "bp", "rp"), monitor=True):
    """Return concatenated arrays t, mag, emag, band_label for FourierDecomp compatibility."""
    t_all, m_all, e_all, b_all = [], [], [], []
    for b in filters:
        t, m, e = extract_gaia_band_data(dl_dict, source_id, b, monitor=monitor)
        if len(t) == 0:
            continue
        t_all.append(t)
        m_all.append(m)
        e_all.append(e)
        b_all.append(np.array([b] * len(t), dtype=object))
    if not t_all:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (np.concatenate(t_all), np.concatenate(m_all), np.concatenate(e_all), np.concatenate(b_all))

def ogle_epoch_arrays(data_dict, source_id, filters=("V","I"), monitor=True):
    tbl = data_dict[source_id]
    # band validity
    m = np.isin(tbl["band"], filters)
    # extract columns
    t    = np.asarray(tbl[phot_names[0]][m], dtype=float)
    mag  = np.asarray(tbl[phot_names[1]][m], dtype=float)
    emag = np.asarray(tbl[phot_names[2]][m], dtype=float)
    band = np.asarray(tbl["band"][m], dtype=object)
    if monitor:
        flts, n_phots = np.unique(band, return_counts=True)
        print(f'{source_id}')
        for flt, n_phot in zip(flts,n_phots):
            print(f'{flt} / {n_phot} epochs')
    return t, mag, emag, band

# =============================================================================
# unified epoch array getter for downstream code
# =============================================================================

def epoch_arrays(data_dict, source_id, mode = None, filters=None, monitor=False):
    """
    Unified accessor to get (t, mag, emag, band) arrays for a given source_id.

    mode='ogle': expects data_dict[source_id] is pandas.DataFrame with columns phot_names + ['band'].
    mode='gaia': expects data_dict[source_id] is astropy.Table loaded by `load_gaia_epoch_phot`.

    bands:
      - ogle: subset of filterss (default: ('V','I'))
      - gaia: subset of ('g','bp','rp') (default: ('g','bp','rp'))
    """
    if mode is None: 
        cfg = get_data_config()
        mode = cfg.mode
    else: cfg = get_data_config(mode)
    
    if filters is None:
        filters = cfg.filters.tolist()

    if mode == "ogle":
        return ogle_epoch_arrays(data_dict, source_id, filters=filters, monitor=monitor)
    elif mode == "gaia":
        return gaia_epoch_arrays(data_dict, int(source_id), filters=filters, monitor=monitor)

    raise ValueError("mode must be 'ogle' or 'gaia'")


