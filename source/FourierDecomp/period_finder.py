import numpy as np
import matplotlib.pyplot as plt

from gatspy import periodic # multiband lomb-scargle
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import minimize_scalar # brent method
from astropy.stats import sigma_clip

from .params import pmin, pmax, n0, delta_P_tol
from .IO import get_data_config

# ==================================
fmin = 1/pmax # expected minimum frequency [days^-1] (<100d)
fmax = 1/pmin # expected maximum frequency [days^-1](anomalous cepheids - 0.4d)
Nterms = 1 # truncated fourier series for Lomb-Scargle 
#f_alias = np.array([1.0]) # list of aliased frequency [days^-1]
#f_alias_tol = 0.005
# =====================================

# Define Lomb-Scargle model (default Nbase = 1, Nband = 1) + Fast
model = periodic.LombScargleMultibandFast(fit_period = True, Nterms=Nterms, 
                                          optimizer_kwds={'quiet':True}) 
model.optimizer.period_range = (pmin,pmax)

def calc_fgrid(t, n0 = 5):
    # n0 : oversampling ratio
    t0, T = t[0], t[-1]-t[0] #initial epoch & length of time window
    tn = np.expand_dims(t-t0, axis=0).T
    
    # Frequency Grid
    delta_f = 1/(n0*T) # grid spacing (VanderPlas 2018)
    #delta_f = (len(t)/2)/(n0*T) # grid spacing (Frescura 2008)
    f = np.arange(fmin,fmax+delta_f,delta_f)
    period = 1/f
    
    return f, period, delta_f

def window_function(t, bmask, f):
    t0, T = t[0], t[-1]-t[0] #initial epoch & length of time window    
    # calculate window function power spectrum
    P_W = np.zeros_like(f)
    for bm in bmask :
        t_ft = t[bm]
        tn_ft = np.expand_dims(t_ft-t0, axis=0).T
        # window power spectrum of current band
        phase = np.exp(-2*np.pi*1j*f*tn_ft)
        P_W += np.abs(np.sum(phase,axis=0))**2
    
    # Normalize (maximum = 1)
    P_W /= max(P_W) 
    return P_W

def window_alias_mask(freqs, f0, f_alias, f_alias_tol=0.01, n_alias = 2):
    # f_alias (scalar)
    # masking ±f_alias [d^-1] aliases: any strong power where |f' - (f ± f_alias)| is tiny
    mask = np.zeros_like(freqs, dtype = 'bool')
    for n in range(1, n_alias+1):
        mask|=(np.abs(freqs - (f0 + n*f_alias)) < f_alias_tol) | (np.abs(freqs - (f0 - n*f_alias)) < f_alias_tol)
    return mask

def harmonic_periods(P0, harmonics = 2):
    Ps = [P0]
    for n in range(2, harmonics + 1):
        if n * P0 <= pmax: Ps.append(n * P0) # subharmonic in frequency (overtone in period)
        if P0 / n >= pmin: Ps.append(P0 / n) # harmonic in frequency (subharmonic in period)
    return Ps

# ========= Period Search with Lomb-Scargle algorithm ==========
def robust_period_search(t, mag, emag, bands, 
                         n0 = 5, K = 8, snr = 3, harmonics = 2,
                         plot = False):
    freqs, periods, delta_f = calc_fgrid(t, n0 = n0)
    
    cfg = get_data_config()
    filters = cfg.activated_bands
    activated_bands = cfg.activated_bands
    bmask = [(bands==filters[ib]) for ib in activated_bands] # original definition: include all passbands?

    # 1) evaluate Lomb-Scargle power
    model.fit(t, mag, emag, bands)
    Pf_LS = model.periodogram(periods)

    # 2) Finding peak (coarse search)
    #sep = int(sep_frac*fmin/delta_f)
    sigma_Pf_LS = np.std(Pf_LS)
    pidx = find_peaks(Pf_LS, height = snr * sigma_Pf_LS)[0] 
    # select peaks having large contrast (prominence)
    prom = peak_prominences(Pf_LS, pidx)[0]
    prom_thres = np.median(prom)
    pidx = find_peaks(Pf_LS, height = snr * sigma_Pf_LS, prominence = prom_thres)[0] 
    # select K peaks
    pidx = pidx[np.argsort(Pf_LS[pidx])[::-1]][:K]
    P_coarse = periods[pidx]
    
    # zoom-in search
    def objective_func(P):
        if P <= pmin or P >= pmax: return np.inf 
        return -model.score([P])[0]
   
    P_refined = []
    for P0 in P_coarse:
        # grid refinement to reach to the desired accuracy
        if delta_f * P0 > delta_P_tol:
            f0 = 1.0 / P0
            # setting bounds at nearby f0
            f_low = max(fmin, f0 - delta_f)
            f_high = min(fmax, f0 + delta_f)
            # minimization
            res = minimize_scalar(objective_func, 
                                  bounds=(1.0/f_high, 1.0/f_low), 
                                  method='bounded')
            if res.success: 
                #print(f'{P0} / {res.x}')
                P_refined.append(res.x)
        else: P_refined.append(P0)

    # 3) period candidates
    Ps = []
    for P0 in P_refined: 
         Ps+=harmonic_periods(P0, harmonics)
    Ps = np.array(Ps); Zs = model.score(Ps)
    mask = (Zs > 3*sigma_Pf_LS) # if peaks are significant
    Ps, Zs = Ps[mask], Zs[mask]

    if plot: plot_LS(freqs, periods, Pf_LS, peaks = [Ps, Zs],
                     thresh = [sigma_Pf_LS, snr * sigma_Pf_LS])
        
    return Ps, Zs

# ======== plot lomb-scargle ===========
def plot_LS(freqs, periods, Pf_LS, peaks = None, thresh = None):
    fig, ax = plt.subplots(2,1, figsize = (15,8))
    # freq-domain
    ax[0].plot(freqs, Pf_LS, color = 'k')
    ax[0].hlines(thresh, fmin-0.05, fmax, ls = 'dotted', color='darkred')
    ax[0].set_xlim(fmin-0.05, fmax)
    ax[0].set_xlabel('Frequencies [$\\rm day^{-1}$]')
    ax[0].set_ylabel('$P_{LS}$')

    # period domain (lograithmic)
    ax[1].plot(periods, Pf_LS, color='k')
    ax[1].hlines(thresh, pmin, pmax, ls = 'dotted', color='darkred')
    ax[1].set_xlim(pmin, pmax)
    ax[1].set_xlabel('Period [day]')
    ax[1].set_ylabel('$P_{LS}$')
    ax[1].set_xscale('log')
    
    if peaks:
        Ps, Zs = peaks
        ax[0].scatter(1/Ps, Zs,marker='+',color='r')
        ax[1].scatter(Ps, Zs,marker='+',color='r')
        
    plt.tight_layout()