
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor  # Threading
from pathlib import Path
from astropy.table import Table, vstack

from .LC import filters, prefixs

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
    if filters is None:
        raise ValueError("filters is required for mode='ogle'")
    for sid in source_ids:
        sid = int(sid)
        tbls = []
        for band in filters:
            fname = Path(phot_dir) / band / f"{sid}.dat"
            if not fname.exists(): continue
            try:
                tab = Table.read(fname, format="ascii.no_header", 
                                 names=phot_names, guess=False,)
                tab['band'] = band
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

def _load_chunk(source_ids, mode="ogle",**kwargs):
    if mode == "ogle":
        return _load_chunk_ogle(source_ids, **kwargs)
    elif mode == "gaia":
        return _load_chunk_gaia(source_ids, **kwargs)
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
        cat_name     = (ident_path.stem).strip('_ident') # catalog name

        cat.append(df_ident_cat)
        jobs.append((cat_name, cat_ids)) # using it to load catalog specific epoch photometry data
    
    # concatnation
    df_ident = pd.concat(cat).reset_index(drop=True) if cat else pd.DataFrame(columns=ident_names)
    return df_ident, jobs

def data_loader(ident_fpath, mode="ogle", ax_workers=12, chunk_size=200):
    if (ident_fpath, str) or (ident_fpath, Path):
        ident_fpath = [ident_fpath]
    
    source_ids = [int(x) for x in source_ids]
    nmax = len(source_ids)
    dl = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, nmax, chunk_size):
            batch_ids = source_ids[i:min(i + chunk_size, nmax)]
            futures.append(executor.submit(_load_chunk, batch_ids, mode=mode))
        for fut in tqdm(futures, desc=):
            dl.update(fut.result())


def wire_globals(decomp_mod, ls_data, df_ident):
    """Attach globals into `decomposition` module namespace."""
    decomp_mod.ls_data = ls_data
    decomp_mod.df_ident = df_ident


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


def load_gaia_epoch_phot(query_result, directory, max_workers=5, chunk_size=1000,
                        filename_pattern="epphot_DR3_{source_id}.ecsv"):
    """
    Load Gaia DR3 epoch photometry ECSV files into a dict[source_id] -> astropy.Table.

    Mirrors the I/O pattern in your notebook:
      filename = directory / f"epphot_DR3_{source_id}.ecsv"

    query_result must provide 'SOURCE_ID' column/field.
    """
    directory = Path(directory)
    if hasattr(query_result, "__getitem__"):
        source_ids = np.array(query_result["SOURCE_ID"])
    else:
        raise ValueError("query_result must provide SOURCE_ID field/column")

    n_max = len(source_ids)
    dl = {} # Datalink

    #Threading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, n_max, chunk_size):
            ids = source_ids[i:min(i + chunk_size, n_max)]
            futures.append(executor.submit(_load_chunk, ids))
        for fut in tqdm(futures, desc="gaia_epoch_phot"):
            dl.update(fut.result())
    return dl


def extract_gaia_band_data(dl_dict, source_id, band, monitor=False):
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


def gaia_epoch_arrays(dl_dict, source_id, bands=("g", "bp", "rp")):
    """Return concatenated arrays t, mag, emag, band_label for FourierDecomp compatibility."""
    t_all, m_all, e_all, b_all = [], [], [], []
    for b in bands:
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

def ogle_epoch_arrays(data_dict, source_id, bands=("V","I")):
    df = data_dict[source_id]
    m = df["band"].isin(list(bands))
    t = df.loc[m, phot_names[0]].to_numpy(dtype=float)
    mag = df.loc[m, phot_names[1]].to_numpy(dtype=float)
    emag = df.loc[m, phot_names[2]].to_numpy(dtype=float)
    band = df.loc[m, "band"].to_numpy(dtype=object)
    return t, mag, emag, band

# =============================================================================
# unified epoch array getter for downstream code
# =============================================================================

def epoch_arrays(data_dict, source_id, mode: str = "ogle", bands=None):
    """
    Unified accessor to get (t, mag, emag, band) arrays for a given source_id.

    mode='ogle': expects data_dict[source_id] is pandas.DataFrame with columns phot_names + ['band'].
    mode='gaia': expects data_dict[source_id] is astropy.Table loaded by `load_gaia_epoch_phot`.

    bands:
      - ogle: subset of filters (default: filters)
      - gaia: subset of ('g','bp','rp') (default: ('g','bp','rp'))
    """
    mode = mode.lower().strip()
    source_id = int(source_id)
    if mode == "ogle":
        if bands is None:
            bands = ("V","I")
        return ogle_epoch_arrays(data_dict, source_id, bands=bands)

    if mode == "gaia":
        if bands is None:
            bands = ("g", "bp", "rp")
        return gaia_epoch_arrays(data_dict, int(source_id), bands=bands)

    raise ValueError("mode must be 'ogle' or 'gaia'")
