import numpy as np

from scipy.signal import find_peaks
from scipy.optimize import minimize

from gatspy import periodic # multiband lomb-scargle

from .params import M_MAX, M_MIN, THRESHOLD, Amin, Amax, pmin, pmax, n0, K, harmonics, snr, opt_method, lam
from .LC import phase_gap_exceeds
from .IO import epoch_arrays, get_data_config 
from .LSQ import LSQ_fit, chisq, chisq_single, unpack_theta
from .period_finder import robust_period_search

import warnings
warnings.simplefilter("ignore")

def bic(reduced_chi2, N, k):
    dof = max(N - k, 1)
    chi2 = reduced_chi2 * dof
    return chi2 + k * np.log(max(N, 1))

def _fit_wrapper(P0, args, M_fit, n_bands, bounds_full, phase_flag, period_fit = False):
    """
    Auxilary function to minimize objective function for given (P, M_fit)
    """
    t, mag, emag, bmask = args
    n_dim = 2 * n_bands + 2 * M_fit + 2 # n_dim 
    
    # initial parameter
    theta0 = LSQ_fit(P0, args, M_fit, phase_flag=phase_flag, opt_method = opt_method, lam = lam)
    
    if period_fit:
        # P와 E의 bound를 theta0
        P_init, E_init = theta0[-2], theta0[-1]
        bounds_full[-2] = (max(P_init * 0.9, pmin), min(P_init * 1.1, pmax)) # P 범위
        bounds_full[-1] = (E_init - P_init, E_init + P_init)     # E 범위

        # 2. Global minimization
        res = minimize(chisq, theta0,
                       args=(t, mag, emag, bmask, M_fit, n_dim),
                       method='L-BFGS-B',
                       bounds=bounds_full)

        if res.success:
            return res.x, res.fun #, M_fit, n_dim
        else: flag = True
    else: flag = True
        
    if flag:
        # optimization failure
        chi2_init = chisq(theta0, *args, M_fit=M_fit, n_dim=n_dim)
        return theta0, chi2_init #, M_fit, n_dim

# === Main Function ===
def fourier_decomp(sid, period_fit=False, verbose=False, plot_LS = False,
                  K = K, harmonics = harmonics, mode='ogle'):
    # Load data
    pulsation = df_ident[df_ident['ID'] == sid]['pulsation'].values[0]
    
    if mode is None: mode = get_data_config().mode
    cfg = get_data_config(mode)
    filters = cfg.filters; activated_bands = cfg.activated_bands; n_bands = cfg.n_bands

    t, mag, emag, bands = epoch_arrays(ls_data, sid, mode=mode)
    #t, mag, emag, bands = [data[key].values for key in [*phot_names, 'band']]
    bmask = [(bands == band) for band in filters]

    # ======================================
    # 1) Lomb-Scargle - find initial period
    # =====================================
    P0s, Zs = robust_period_search(t, mag, emag, bands, 
                                   n0 = n0, K = K, harmonics = harmonics, 
                                   snr = snr, plot = plot_LS)
    Zmax = Zs.max()
    if verbose: print(f'Lomb-Scargle Period = {P0s} / Z = {Zs}')
    
    # =====================================
    # 2) Perform fitting
    # =====================================
    args = (t, mag, emag, bmask)
    
    # bounds (M_MAX)
    m_bounds = [(-np.inf, np.inf)] * n_bands
    a_bounds = [(Amin, Amax)] * n_bands
    A_bounds_max = [(Amin, Amax)] * M_MAX
    Q_bounds_max = [(0, 2 * np.pi)] * M_MAX
    P_bounds = [(pmin, pmax)] # (임시)
    E_bounds = [(-np.inf, np.inf)] # (임시)

    # --- 1st fit (M_MAX) ---
    M_fit_1 = M_MAX
    bounds_1 = m_bounds + a_bounds + A_bounds_max + Q_bounds_max + P_bounds + E_bounds
    
    # _fit_wrapper: return = (theta0, chi2)
    chi2_opt_1 = np.inf
    P0 = pmax
    for Pi, Zi in zip(P0s, Zs):
        #if Zi<0.2*Zmax: continue # non significant component
        # phase filling check
        phase_flag_i = np.array([phase_gap_exceeds(t[mask], Pi, M_fit=M_MAX) for mask in bmask])
        theta_1_tmp, chi2_1_tmp = _fit_wrapper(Pi, args, M_fit_1, n_bands, bounds_1, 
                                               phase_flag = phase_flag_i, period_fit= False)
        if verbose:
            print(f"{Pi:.4f} days / chi2 = {chi2_1_tmp:.4f}")
        if np.isfinite(chi2_1_tmp) and chi2_opt_1 > chi2_1_tmp: 
            if not np.isclose(Pi, P0, rtol = 0.01) and (chi2_opt_1 - chi2_1_tmp) < 5: # compare degree of improvement
                continue
            theta_opt_1 = theta_1_tmp
            chi2_opt_1 = chi2_1_tmp
            P0 = Pi; Zmax = Zi 
            phase_flag = phase_flag_i
            
    if not np.isfinite(chi2_opt_1):
        print(f'ID = {sid} / M_MAX fit failed.')
        return None # 1차 피팅 실패 시 중단
    
    if verbose: print(f'P0 = {P0:.4f} days')

    # --- truncation ---
    _, _, A_vec_1, _, _, _ = unpack_theta(theta_opt_1, n_bands, M_fit=M_fit_1, include_amp=True)

    M_trunc = M_MAX
    if A_vec_1[0] > 1e-5: # not close to 0
        dA = np.diff(A_vec_1)
        significant = np.where(dA/A_vec_1[:-1]>THRESHOLD)[0]
        
        if len(significant) > 0:
            M_trunc = 1 + np.min(significant)
            M_trunc = np.max([M_trunc, M_MIN])
    else:
        M_trunc = M_MIN # M_min

    # --- secondary fitting ---
    if verbose:
        print(f'ID = {sid} / M_MAX fit done. Found M_trunc = {M_trunc}. Refitting...')
            
    M_fit_2 = M_trunc

    # slicing 
    bounds_2 = m_bounds + a_bounds + A_bounds_max[:M_fit_2] + Q_bounds_max[:M_fit_2] + P_bounds + E_bounds
    
    theta_opt_2, chi2_opt_2 = _fit_wrapper(P0, args, M_fit_2, n_bands, bounds_2,
                                           phase_flag = phase_flag, period_fit= period_fit)

    # 2nd fitting is better than 1st fitting
    if verbose:
        print(f'chi1 = {chi2_opt_1:.4e} / chi2 = {chi2_opt_2:.4e}')
    
    bic_1 = bic(chi2_opt_1, len(t), len(theta_opt_1))
    bic_2 = bic(chi2_opt_2, len(t),len(theta_opt_2))
    if np.isfinite(chi2_opt_2) and bic_2 <= bic_1:
        theta_opt_final = theta_opt_2
        chi2_opt_final = chi2_opt_2
        M_fit_final = M_fit_2
    else:
        # 2nd fitting failed or worsen
        theta_opt_final = theta_opt_1
        chi2_opt_final = chi2_opt_1
        M_fit_final = M_fit_1
    
    # =====================================
    # 3) summary statistics
    # =====================================
    # M_fit_final
    m0, amp, A_fit, Q_fit, P, E = unpack_theta(theta_opt_final, n_bands, M_fit=M_fit_final, include_amp=True)
    
    # zero-padding (to fit M_MAX)
    A_out = np.zeros(M_MAX)
    Q_out = np.zeros(M_MAX)
    A_out[:M_fit_final] = A_fit
    Q_out[:M_fit_final] = Q_fit
    
    theta_params_out = np.hstack([m0, amp, A_out, Q_out])
    
    flag = 0
    N, sig, rms = np.zeros(n_bands, dtype='int'), np.zeros(n_bands), np.zeros(n_bands)
    phi_rise = np.nan
    
    for i, ib in enumerate(activated_bands):
        mask = bmask[ib]
        if not np.any(mask):
            N[i] = 0
            flag = 1 # no data
            continue
            
        t_ft, mag_ft, emag_ft = t[mask], mag[mask], emag[mask]
        N[i] = len(t_ft)
        
        sig[i] = np.sqrt(np.average(mag_ft, weights=1 / emag_ft**2))
    
        theta_ft = [m0[i], amp[i], A_fit, Q_fit, P, E] 
        rms[i] = np.sqrt(chisq_single(theta_ft, t_ft, mag_ft, emag_ft, M_fit_final, unpack=False) / N[i])

        # parameter boundary excession check
        if (amp[i] > Amax) or (amp[i] < Amin): flag = 1
        m_lb, m_ub = min(mag_ft), max(mag_ft)
        if (m0[i] < m_lb) or (m0[i] > m_ub): flag = 1

        # Calculate rising time
        if filters[ib] == 'I':
            m_pk = np.percentile(mag_ft,[99,1])
            phase_pk = (t_ft[np.argmin(abs(mag_ft-m_pk[:,None]),axis=1)]/P0)%1 
            delta_phi = np.diff(phase_pk)[0]
            phi_rise = np.min([1-delta_phi, delta_phi])
            
    if verbose:
        print(f'ID = {sid} / Final M_fit = {M_fit_final} / CHI2 = {chi2_opt_final:.2f} / rrms = {rms[0]/sig[0]:.4f} / P = {P:.6f} days')

     #return m0, amp, A, Q, P, E, M_fit_final
    row = [sid, pulsation, *N, *sig, *rms, Zmax, P0, chi2_opt_final, P, E, phi_rise, M_fit_final, *theta_params_out, flag]
    return row