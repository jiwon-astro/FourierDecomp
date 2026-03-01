import numpy as np

from scipy.signal import find_peaks
from scipy.optimize import minimize

from astropy.stats import sigma_clip

from gatspy import periodic # multiband lomb-scargle

from .params import M_MAX, M_MIN, Amin, Amax, pmin, pmax, n0, K, harmonics, snr, opt_method, lam0, lam_min, lam_max, w_spike, THRESHOLD, ERR_FLOOR
from .LC import phase_gap_score
from .IO import epoch_arrays, get_data_config 
from .LSQ import F, LSQ_fit, chisq, chisq_single, unpack_theta
from .period_finder import robust_period_search

import warnings
warnings.simplefilter("ignore")

def bic(reduced_chi2, N, k):
    dof = max(N - k, 1)
    chi2 = reduced_chi2 * dof
    return chi2 + k * np.log(max(N, 1))

def eval_on_grid(theta, n_bands, M_fit, n_grid = 200, unpack=True):
    phi_grid = np.arange(0, 1, 1/n_grid)
    _, amp, A, Q, P, E = unpack_theta(theta, n_bands, M_fit=M_fit, include_amp=True)
    theta_rev = np.array([0, amp[0], A, Q, 1., (E/P)%1]) # unit period + phase offset, enough to use single band
    fval  = F(theta_rev, phi_grid, M_fit)
    return phi_grid, fval

def spike_penalty(fval, ratio = 0.05):
    # model value at uniform grid
    d2 = np.diff(fval, n=2)
    pk_max = np.max(np.abs(d2)) # spike
    pk_tot = np.sum(np.abs(d2)) # total variance
    return pk_max + ratio * pk_tot

def adjust_lambda(lam0, gmax, M_fit, N, lam_min = 1e-5, lam_max = 1e-1):
    # Heuristic function to adjust regularization weight
    lam = lam0 * (max(gmax, 0.05) / 0.12)**3 * (M_fit / 5.0)**2 * (30/max(N, 10))
    return float(np.clip(lam, lam_min, lam_max))   

def _fit_wrapper(P0, args, M_fit, bounds_full, activated_bands, phase_gaps, 
                period_fit=False, use_optim=False, adaptive_lam=True, verbose=False):
    """
    Auxilary function to minimize objective function for given (P, M_fit)
    """
    t, mag, emag, bmask = args
    n_bands = len(activated_bands)
    n_dim = 2 * n_bands + 2 * M_fit + 2 # n_dim 
    
    # initial parameter
    phase_flag = (phase_gaps > 0.05)
    lam = lam0
    if adaptive_lam:
        lam = adjust_lambda(lam0, phase_gaps[0], M_fit, np.sum(bmask[0]), lam_min=lam_min, lam_max=lam_max) # set first band as pivotal band
        if verbose: print(f'phase_gap_max = {phase_gaps} / N_ft = {np.sum(bmask, axis = 1)} / lam = {lam:.2e}')
    theta0 = LSQ_fit(P0, args, M_fit, activated_bands, phase_flag=phase_flag, 
                        opt_method = opt_method, lam = lam)

    P_init, E_init = theta0[-2], theta0[-1]
    if period_fit:
        # P와 E의 bound를 theta0
        bounds_full[-2] = (max(P_init * 0.9, pmin), min(P_init * 1.1, pmax)) # P 범위
    else: bounds_full[-2] = (P_init, P_init)
    bounds_full[-1] = (E_init - P_init, E_init + P_init)     # E 범위

    # 2. Global minimization
    if use_optim:
        res = minimize(chisq, theta0,
                    args=(t, mag, emag, bmask, M_fit, n_dim, activated_bands),
                    method='L-BFGS-B',
                    bounds=bounds_full)
        if res.success:
            return res.x, res.fun #, M_fit, n_dim
        
    # optimization failure
    chi2_init = chisq(theta0, *args, M_fit=M_fit, n_dim=n_dim, activated_bands=activated_bands)
    return theta0, chi2_init #, M_fit, n_dim


def _build_bounds(n_bands, M_fit):
    m_bounds = [(-np.inf, np.inf)] * n_bands
    a_bounds = [(Amin, Amax)] * n_bands
    A_bounds = [(Amin, Amax)] * M_fit
    Q_bounds = [(0, 2 * np.pi)] * M_fit
    P_bounds = [(pmin, pmax)] # (임시)
    E_bounds = [(-np.inf, np.inf)] # (임시)
    return m_bounds + a_bounds + A_bounds + Q_bounds + P_bounds + E_bounds

def calculate_m0_amp(args, sigma = 3.0, maxiter = 5):
    # calculate 1) mean magnitude and 2) peak-to-peak amplitude from multi-band epoch photometry
    # using for initial guess 
    t, mag, emag, bmask = args
    n_bands = len(bmask)
    m0s = np.zeros(n_bands); A0s = np.zeros(n_bands)
    resmasks = []
    for i, m in enumerate(bmask):
        mag_ft, emag_ft = mag[m], emag[m]
        w_ft = 1/np.maximum(emag_ft, ERR_FLOOR)**2
        n_prev = len(mag_ft)
        m0_ft = np.average(mag_ft, weights = w_ft)
        for _ in range(maxiter):
            resmask = sigma_clip(mag_ft - m0_ft, sigma=sigma, masked=True).mask
            m0_ft = np.average(mag_ft[~resmask], weights = w_ft[~resmask])
            Amp_ft = np.diff(np.percentile(mag_ft[~resmask], [5, 95])) 
            #Amp_ft = max(mag_ft[~resmask]) - min(mag_ft[~resmask])
            n_curr = (~resmask).sum()
            if n_curr<n_prev: n_prev = n_curr
            else: break
        m0s[i] = m0_ft; A0s[i] = Amp_ft
        resmasks.append(resmask)
    return m0s, A0s, resmask 

# === Main Function ===
def fourier_decomp(sid, period_fit=False, use_optim=False, adaptive_lam=True,
                   verbose=False, plot_LS=False, K=K, harmonics=harmonics, w_spike=w_spike,
                   mode='ogle', init='lsq'):
    # Load data
    if mode is None: mode = get_data_config().mode
    cfg = get_data_config(mode)
    filters = cfg.filters; activated_bands = cfg.activated_bands; n_bands = cfg.n_bands

    if mode=='ogle':
        sid_mask = (df_ident['ID'] == sid)
        pulsation = df_ident[sid_mask]['pulsation'].values[0]
    elif mode=='gaia':
        sid_mask = (df_ident['SOURCE_ID'] == sid)
        cep_type = df_ident['type_best_classification'][sid_mask][0]
        osc_type = df_ident['mode_best_classification'][sid_mask][0]
        pulsation = f'{cep_type}_{osc_type}'

    t, mag, emag, bands = epoch_arrays(ls_data, sid, mode=mode)
    #t, mag, emag, bands = [data[key].values for key in [*phot_names, 'band']]
    bmask = [(bands == band) for band in filters]
    args = (t, mag, emag, bmask)

    m0_data, amp_data, _ = calculate_m0_amp(args) # mean / peak-to-peak amplitude
    if verbose:
        print(f'mean mags = {m0_data}')
        print(f'amplitude = {amp_data}')
    # ======================================
    # 1) initial period
    # =====================================
    theta0_rrfit = None
    if init == 'rrfit':
        if df_rrfit is None:
            raise ValueError("init='rrfit' requires rrfit result file.")
        if templates is None:
            raise ValueError("init='rrfit' requires templates dict (A/Q).")
        sid_mask = (df_rrfit['ID'] == sid)
        fit_row = df_rrfit[sid_mask]
        P0 = float(fit_row['P'])
        E0 = float(fit_row['EPOCH'])
        T_idx = int(fit_row['T'])
        tmpl = templates[f'T{T_idx}']

        A_tmp = np.zeros(M_MAX); Q_tmp = np.zeros(M_MAX)
        A_RRFIT = np.array(tmpl.A, dtype=float)
        Q_RRFIT = np.array(tmpl.Q, dtype=float)
        mlen = min(len(A_RRFIT), M_MAX)
        A_tmp[:mlen] = A_RRFIT[:mlen]
        Q_tmp[:mlen] = Q_RRFIT[:mlen]

        #theta0_rrfit = np.array([*m0_data, *A0_data, *A_tmp, *Q_tmp, P0, E0], dtype=float)
        
        P0s = np.array([P0])
        Zs = np.array([np.nan])
        Zmax = np.nan

        if verbose: print(f'RRFit period = {P0:.4f}d / E = {E0:.4f}')
    else:
        P0s, Zs = robust_period_search(t, mag, emag, bands, 
                                    n0 = n0, K = K, harmonics = harmonics, 
                                    snr = snr, plot = plot_LS)
        Zmax = Zs.max()
        if verbose: print(f'Lomb-Scargle Period = {P0s} / Z = {Zs}')
    
    # =====================================
    # 2) Perform fitting
    # =====================================

    # --- 1st fit (M_MAX) ---
    M_fit_1 = M_MAX
    bounds_1 = _build_bounds(n_bands, M_fit_1)
    
    # _fit_wrapper: return = (theta0, chi2)
    chi2_opt_1 = np.inf
    P0 = pmax # initialize (best period)
    for Pi, Zi in zip(P0s, Zs):
        #if Zi<0.2*Zmax: continue # non significant component
        # phase filling check
        phase_gaps_i = np.array([phase_gap_score(t[mask], Pi, M_fit=M_MAX) for mask in bmask]) # maximum gap in phase domain
        
        """
        theta_init = None
        if init == 'rrfit': theta_init = theta0_rrfit
        """
        theta_1_tmp, chi2_1_tmp = _fit_wrapper(Pi, args, M_fit_1, bounds_1, activated_bands, phase_gaps = phase_gaps_i, 
                                               period_fit=period_fit, use_optim=use_optim, adaptive_lam=adaptive_lam, verbose=verbose)
                                               #theta0=theta_init)
        if verbose:
            print(f"{Pi:.4f} days / chi2 = {chi2_1_tmp:.4f}")
        if np.isfinite(chi2_1_tmp) and chi2_opt_1 > chi2_1_tmp: 
            if not np.isclose(Pi, P0, rtol = 0.01) and (chi2_opt_1 - chi2_1_tmp) < 5: # compare degree of improvement
                continue
            theta_opt_1 = theta_1_tmp
            chi2_opt_1 = chi2_1_tmp
            P0 = Pi; Zmax = Zi 
            phase_gaps = phase_gaps_i
            
    if not np.isfinite(chi2_opt_1):
        print(f'ID = {sid} / M_MAX fit failed.')
        return None # 1차 피팅 실패 시 중단
    
    if verbose: print(f'P0 = {P0:.4f} days')

    # --- truncation ---
    _, _, A_vec_1, _, _, _ = unpack_theta(theta_opt_1, n_bands, M_fit=M_fit_1, include_amp=True)

    M_trunc = M_MAX
    # To Do: Dealing the M_MIN?
    # currently, M_MIN = 3  (fixed)
    # for 1O/2O? - better to use M_MIN<3?
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
    bounds_2 = _build_bounds(n_bands, M_fit_2)
    theta_opt_2, chi2_opt_2 = _fit_wrapper(P0, args, M_fit_2, bounds_2, activated_bands, phase_gaps = phase_gaps, 
                                           period_fit= period_fit, use_optim=use_optim, adaptive_lam=adaptive_lam, verbose=verbose)
                                           #theta0=(theta0_rrfit if init=='rrfit' else None))

    # 2nd fitting is better than 1st fitting
    if verbose:
        print(f'chi1 = {chi2_opt_1:.4e} / chi2 = {chi2_opt_2:.4e}')
    
    bic_1 = bic(chi2_opt_1, len(t), len(theta_opt_1))
    bic_2 = bic(chi2_opt_2, len(t),len(theta_opt_2))

    _, fval_grid_1 = eval_on_grid(theta_opt_1, n_bands, M_fit_1)
    _, fval_grid_2 = eval_on_grid(theta_opt_2, n_bands, M_fit_2)
    spike_1 = spike_penalty(fval_grid_1)
    spike_2 = spike_penalty(fval_grid_2)
    bic_1 += w_spike * spike_1
    bic_2 += w_spike * spike_2

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
        
        w_ft = 1 / np.maximum(emag_ft, ERR_FLOOR)**2
        # photometric error (w/o genuine pulsation)
        dm_ft = mag_ft-m0_data[ib]
        sig[i] = np.sqrt(np.average(dm_ft**2, weights=w_ft))
    
        theta_ft = [m0[i], amp[i], A_fit, Q_fit, P, E] 
        fval = F(theta_ft, t_ft, M_fit_final)

        # residual 
        resid_ft = mag_ft - fval
        rms[i] = np.sqrt(np.average(resid_ft**2, weights=w_ft))

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