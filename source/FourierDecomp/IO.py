
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Union
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
    n_bands_full: int
    n_bands: int
    R: float
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

    if mode not in ("ogle", "gaia", "ztf"):
        raise ValueError(f"Unknown data_class/mode: {mode}")

    flt = np.array(base['filters'], dtype=object)
    pfx = np.array(base['prefixs'], dtype=int)
    act = np.array(base['activated_bands'], dtype=int)
    n_b_full = len(flt)
    n_b = len(act)

    colors = list(base['lc_colors'])
    markers = list(base['lc_markers'])
    R = base['wesenheit_factor']

    return DataConfig(
        mode=mode,
        filters=flt,
        prefixs=pfx,
        activated_bands=act,
        n_bands_full=n_b_full,
        n_bands=n_b,
        R=R,
        lc_colors=colors,
        lc_markers=markers,
    )

# ===================================================
# output header
# ===================================================
def build_fd_header(mode = None):
    """
    Build output header for Fourier decomposition.
    Output row format must match decomposition.fourier_decomp().

    Expected row layout (as in decomposition.fourier_decomp):
      [sid, pulsation, *N, *sig, *rms, Zmax, P0, chi2, P, E, phi_rise, M_fit, *theta_params_out, flag]
    where:
      N/sig/rms are
      theta_params_out = [*m0, *amp, *A(1..M_MAX), *Q(1..M_MAX)]
        but only for activated bands in m0/amp
    """
    from .params import M_MAX

    cfg = get_data_config(mode)
    filters = cfg.filters
    #active_idx = list(cfg.activated_bands)
    #active_filters = [str(cfg.filters[i]) for i in active_idx]f

    cols = []
    cols += ["ID", 'pulsation']

    # Per-band stats
    cols += [f"N_{b}" for b in filters]
    cols += [f"sig_{b}" for b in filters]
    cols += [f"rms_{b}" for b in filters]
    cols += [f"gmax_{b}" for b in filters] # maximum phase gap folded by P0
    
    # Period / fit summary
    cols += ["Zmax", "P0", "chi2", "fobj", "P", "E", "phi_rise", "M_fit"]

    # Theta params (match decomposition.py theta_params_out ordering)
    cols += [f"m0_{b}" for b in filters]
    cols += [f"amp_{b}" for b in filters]

    # Fourier series params
    cols += [f"A{j}" for j in range(1, M_MAX + 1)]
    cols += [f"Q{j}" for j in range(1, M_MAX + 1)]

    cols += ["flag"]
    return cols

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

def _load_chunk_ztf(source_ids, phot_dir, match_tbl, sep_tol=0.1,
                    id_col="ID", ztf_col="ztf_id", sep_col="sep"):
    """
    Gaia source_id 기준으로 match_table을 읽고,
    sep <= sep_tol 인 ZTF oid 파일들을 읽어 하나의 source로 합침.
    """
    data = {}
    phot_dir = Path(phot_dir)

    target_ids = match_tbl[id_col]
    ztf_ids = match_tbl[ztf_col]
    seps = match_tbl[sep_col]

    for sid in source_ids:
        # Gaia cross-match가 된 것만 사용
        m = (target_ids == sid) & np.isfinite(seps) & (seps <= sep_tol) & (ztf_ids >= 0)
        if not np.any(m): continue

        ztf_ids_matched = np.unique(ztf_ids[m])

        tbls = []
        for ztf_id in ztf_ids_matched:
            candidates = [
                phot_dir / f"ztf_{ztf_id}.ecsv",
                phot_dir / "epoch_phot" / f"ztf_{ztf_id}.ecsv",
            ]

            fname = None
            for cand in candidates:
                if cand.exists():
                    fname = cand
                    break

            if fname is None:
                continue

            try:
                tbl = Table.read(fname, format="ascii.ecsv")
                tbl["gaia_source_id"] = np.full(len(tbl), sid)
                tbl["matched_ztf_id"] = np.full(len(tbl), ztf_id)
                tbls.append(tbl)
            except Exception:
                continue

        if not tbls:
            continue

        data[sid] = vstack(tbls, metadata_conflicts="silent")
    return data

def _load_chunk(source_ids, phot_dir, mode="ogle",**kwargs):
    if mode == "ogle":
        return _load_chunk_ogle(source_ids, phot_dir, **kwargs)
    elif mode == "gaia":
        return _load_chunk_gaia(source_ids, phot_dir, **kwargs)
    elif mode == "ztf":
        return _load_chunk_ztf(source_ids, phot_dir, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

# --------------
# identifiers
# -------------
def _read_ident_gaia(query_fpath):
    # load gaia query data
    tab = Table.read(query_fpath, format='ascii.ecsv')
    return tab

def _read_match_table_ztf(match_fpath):
    # Gaia-ZTF match
    match_fpath = Path(match_fpath)
    return Table.read(match_fpath, format="ascii.ecsv")

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
                            max_workers=12, chunk_size=200, **kwargs):
    nmax = len(source_ids)
    dl = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, nmax, chunk_size):
            batch_ids = source_ids[i:min(i + chunk_size, nmax)]
            futures.append(executor.submit(_load_chunk, batch_ids, phot_dir, mode=mode, **kwargs))
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

    # ---- ZTF ----
    elif mode=="ztf":
        df_match = _read_match_table_ztf(ident_fpath)
        source_ids = df_match["ID"]
        dl = _chunk_loader_threading(
            source_ids,
            phot_dir=phot_dir,
            mode="ztf",
            max_workers=max_workers,
            chunk_size=max(chunk_size, 200),
            desc="ztf_epoch_phot_from_match",
            match_tbl=df_match
        )

    return df_ident, dl


def wire_globals(module, ls_data, df_ident, 
                 df_rrfit = None, templates = None):
    """Attach globals into given module namespace."""
    module.ls_data = ls_data
    module.df_ident = df_ident
    module.df_rrfit = df_rrfit
    module.templates = templates


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

def ztf_epoch_arrays(data_dict, source_id, filters=("zg", "zr", "zi"), monitor=True):
    tbl = data_dict[source_id]
    magerr = np.asarray(tbl["magerr"], dtype=float)
    
    # photometric error / quality cut
    good = np.isfinite(magerr) & (magerr >= 0) & np.asarray(tbl["catflags"]) == 0
    tbl = tbl[good]

    # band subset
    m = np.isin(tbl["filtercode"], filters)
    tbl = tbl[m]

    # time/mag/emag/band
    t    = np.asarray(tbl["mjd"], dtype=float) 
    mag  = np.asarray(tbl["mag"], dtype=float) 
    emag = np.asarray(tbl["magerr"], dtype=float) 
    band = np.asarray(tbl["filtercode"], dtype=object) 

    # epoch-order sorting
    if len(t) > 0:
        srt = np.argsort(t)
        t, mag, emag, band = t[srt], mag[srt], emag[srt], band[srt]

    if monitor:
        print(f"{source_id}")
        ztf_ids = np.unique(np.asarray(tbl["matched_ztf_id"]))
        print(f"matched ztf ids = {ztf_ids}")
        if len(band) > 0:
            flts, n_phots = np.unique(band, return_counts=True)
            for flt, n_phot in zip(flts, n_phots):
                print(f"{flt} / {n_phot} epochs")

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
    elif mode == "ztf":
        return ztf_epoch_arrays(data_dict, int(source_id), filters=filters, monitor=monitor)

    raise ValueError("mode must be 'ogle', 'gaia' or 'ztf'")

# ============================
# RRFit input lightcurve
# ============================
@dataclass
class RRFitLC:
    sid: Union[str, int]
    fitlc_path: Union[str, Path] # .fitlc path
    t: np.ndarray
    mag: np.ndarray
    emag: np.ndarray
    bands: np.ndarray
        
def prepare_fitlc(sid, mode='gaia', ls_data=None, fitlc_path=None, workdir=None):
    cfg = get_data_config(mode)
    filters = cfg.filters; prefixs = cfg.prefixs
    
    fitlc_columns = ["Band","time(days)","mag(mag)","mag_err"]
    fitlc_hdr = " ".join(fitlc_columns)
    
    # 1) .fitlc file already exists
    if fitlc_path is not None:
        # flts: band prefixs
        arr = np.loadtxt(fitlc_path, ndmin=2)
        flts = arr[:, 0].astype(int)
        t    = arr[:, 1].astype(float)
        mag  = arr[:, 2].astype(float)
        emag = arr[:, 3].astype(float)
        bands = np.asarray([filters[np.where(prefixs==i)[0][0]] for i in flts], 
                           dtype='object')
        return RRFitLC(sid=sid, fitlc_path=str(fitlc_path),
                           t=t, mag=mag, emag=emag, bands=bands)
        
    # 2) load lightcurve from ls_data, write new .fitlc file
    if ls_data is not None:
        if workdir is None: 
            raise ValueError("workdir is required to export a lightcurve as .fitlc")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    
        t, mag, emag, bands = epoch_arrays(ls_data, sid, mode=mode)
        flts = np.asarray([prefixs[np.where(filters==b)[0][0]] for b in bands], 
                          dtype='int')
        # write .fitlc file
        fitlc_path = workdir / f"{sid}.fitlc"
        with open(fitlc_path, "w") as f:
            f.write(fitlc_hdr+"\n")
            for bi, ti, mi, ei in zip(flts, t, mag, emag):
                f.write(f"{bi:d} {ti:.8f} {mi:.6f} {ei:.6f}\n")
        return RRFitLC(sid=sid, fitlc_path=str(fitlc_path),
                       t=t, mag=mag, emag=emag, bands=bands)
    
    raise ValueError("Either fitlc_path or ls_data must be provided.")


