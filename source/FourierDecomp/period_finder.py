import numpy as np
from gatspy import periodic # multiband lomb-scargle
from astropy.stats import sigma_clip

from .params import pmin, pmax

# ==================================
fmin = 1/pmax # expected minimum frequency [days^-1] (<100d)
fmax = 1/pmin # expected maximum frequency [days^-1](anomalous cepheids - 0.4d)
f_alias = np.array([1.0]) # list of aliased frequency [days^-1]
tol = 0.01
n0 = 2
# =====================================

# Define Lomb-Scargle model (default Nbase = 1, Nband = 1) + Fast
model = periodic.LombScargleMultibandFast(fit_period = True, Nterms=1, optimizer_kwds={'quiet':True}) 
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
    
    return f, period 

def window_function(t, n0 = 5):
    t0, T = t[0], t[-1]-t[0] #initial epoch & length of time window
    tn = np.expand_dims(t-t0, axis=0).T
    
    # Frequency Grid
    delta_f = 1/(n0*T) # grid spacing (VanderPlas 2018)
    #delta_f = (len(t)/2)/(n0*T) # grid spacing (Frescura 2008)
    f = np.arange(fmin,fmax+delta_f,delta_f)
    period = 1/f
    
    # calculate window function power spectrum
    phase = np.exp(-2*np.pi*1j*f*tn)
    P_W = np.abs(np.sum(phase,axis=0))**2
    P_W /= max(P_W) # Normalized (maximum = 1)
    return f, period, P_W

def window_alias_mask(freqs, f0, f_alias, tol=0.01):
    # masking ±f_alias [d^-1] aliases: any strong power where |f' - (f ± f_alias)| is tiny
    return (np.abs(freqs - (f0 + f_alias)) < tol) | (np.abs(freqs - (f0 - f_alias)) < tol)

# ========= Period Search with Lomb-Scargle algorithm ==========
def robust_period_search(t, mag, emag, bands, 
                         K = 5, snr = 5, harmonics = 2, maxit = 5):
    freqs, periods = calc_fgrid(t, n0 = n0)
    
    # evaluate Lomb-Scargle power
    model.fit(t, mag, emag, bands)
    Pf_LS = model.periodogram(periods)
    
    # 1) Cleaning aliased frequency
    # To DO - Window function based aliasing? 
    # To Do - How to do better cleaning?
    cleaned = False
    n_it = 0
    while (not cleaned) and (n_it < maxit):
        # find top K prominent peaks (sorted by Lomb-Scargle power)
        pidx = np.argpartition(Pf_LS, -K)[-K:]
        pidx = pidx[np.argsort(Pf_LS[pidx])][::-1]

        # masking aliased frequencies nearby prominent peaks
        freq_mask = np.zeros_like(freqs, dtype = 'bool')
        for k in pidx:
            for fa in f_alias:
                freq_mask|=window_alias_mask(freqs, freqs[k], fa, tol=tol)
                freq_mask&=(Pf_LS>Pf_LS[k]/snr)
        if not freq_mask.any(): cleaned = True
        else:
            cleaned = False
            Pf_LS[freq_mask] = 0
        n_it+=1
    
    # 2) Finding genuine peak
    P0 = periods[np.argmax(Pf_LS)]
    Ps = [P0]
    if harmonics>=2:
        for n in range(2,harmonics+1):
            if n*P0 <= pmax: Ps.append(n*P0)
            if P0/n <= pmax: Ps.append(P0/n)
    Ps = np.array(Ps)
    Zs = model.score(Ps)
    
    return Ps, Zs