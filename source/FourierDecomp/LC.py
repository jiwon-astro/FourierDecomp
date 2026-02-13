import numpy as np
import matplotlib.pyplot as plt
from .params import M_MAX

# --- LC identifier ---
ident_names = ['ID','pulsation','RA','Dec','OGLE-IV ID','OGLE-III ID','OGLE-II ID','other']
phot_names = ['t','mag','emag']

# --- photometric bands ---
filters = np.array(['V','I'])
prefixs = np.array([0, 1])
lc_colors = ['yellowgreen','orange']
lc_markers = ['o','s']

def compute_phase(t, P):
    return (t / P) % 1.0

def phase_gap_exceeds(t, P, threshold=0.05, M_fit=M_MAX): 
    if len(t) < (M_fit + 2): return True # M_fit 사용
    phi = compute_phase(t, P)
    phi_sorted = np.sort(phi)
    gaps = np.diff(np.r_[phi_sorted, phi_sorted[0] + 1.0])
    return gaps.max() > threshold

# Expand phased light curve (arbitary phase range)
def expand_light_curve(t, flux, flux_err,  period, phase_range = (0,1)):
    original_phase = compute_phase(t, period)
    
    #phase range [0,1) -> [n,n+1)
    phase_i, phase_f = phase_range
    n_i, n_f = int(np.floor(phase_i)), int(np.ceil(phase_f))-1 # determine duplication range
    
    # expanding data
    phase_list, flux_list, ferr_list = [], [], []
    for n in range(n_i, n_f+1):
        shifted_phase = original_phase + n
        mask = (shifted_phase >= phase_i) & (shifted_phase <= phase_f)
        if np.any(mask):
            phase_list.append(shifted_phase[mask])
            flux_list.append(flux[mask])
            if np.any(flux_err): ferr_list.append(flux_err[mask])
            
    if phase_list:
        phase = np.concatenate(phase_list)
        flux = np.concatenate(flux_list)
        
        # sorting refer to phase
        order = np.argsort(phase)
        phase, flux = phase[order], flux[order]
        if np.any(flux_err): flux_err = np.concatenate(ferr_list)[order]
            
    else:
        phase, flux, flux_err = np.array([]), np.array([]), np.array([])
    
    return phase, flux, flux_err

# set proper axis limit
def set_ylim(y,yerr,amp=1):
    mean_y = np.mean(y)
    std_y = np.std(y)
    
    ymin = min(y) - 1.5 * np.max(yerr)
    ymax = max(y) + 1.5 * np.max(yerr)

    #3-sigma outliers
    sigma = 2
    if ymax > mean_y + sigma * std_y: ymax = mean_y + sigma * std_y
    if ymin < mean_y - sigma * std_y: ymin = mean_y - sigma * std_y
        
    yscale=0.25*abs(amp)
    return ymax+yscale, ymin-yscale

# plot light curve
def plot_lc(sid, P0, selected_filters = ['I'], phase_max = 2):
    n_bands = len(selected_filters)
    data = ls_data[sid]
    t, mag, emag, bands = [data[key].values for key in [*phot_names,'band']]
    bmask = [(bands==band) for band in selected_filters]

    fig, ax = plt.subplots(n_bands,1,figsize=(10,3*n_bands),dpi=300)
    if n_bands == 1: ax = [ax]
    ax[0].set_title(f'{sid}',loc='left',fontsize = 18)
    
    for i, band in enumerate(selected_filters):
        mask = bmask[i]; ib = prefixs[filters == band][0] # from filter catalog
        t_ft, mag_ft, emag_ft = t[mask], mag[mask], emag[mask]
        if len(t_ft)==0: continue
        ext_phase, ext_mag, ext_emag = expand_light_curve(t_ft, mag_ft, emag_ft, P0, phase_range = (0, phase_max))
        ax[i].errorbar(ext_phase, ext_mag, yerr = ext_emag, ls='None', 
                       color=lc_colors[ib], marker=lc_markers[ib], lw = 1, zorder = 0)

        y_lb, y_ub = set_ylim(mag_ft, emag_ft)

        ax[i].text(0,y_lb-(y_lb-y_ub)*0.1,s=f'P={P0:.4f}days')
        ax[i].text(0.05, y_ub+(y_lb-y_ub)*0.2, s=f'{band} ({len(t_ft)} epochs)')
        ax[i].set_ylim(y_lb, y_ub)
        ax[i].set_xlim(-0.1,phase_max+0.1)
        ax[i].set_ylabel('Magnitude\n[mag]')
        if i==n_bands-1: ax[i].set_xlabel('Phase')