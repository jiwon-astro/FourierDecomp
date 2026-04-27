import numpy as np

from scipy.optimize import minimize
from astropy.stats import sigma_clip

from . import params
from .LC import phase_gap_score
from .IO import epoch_arrays, get_data_config 
from .LSQ import F, H, LSQ_fit, refit_m0_amp, chisq, unpack_theta, peak_to_peak_amplitude, _coef_mode, _harmonic_bounds, theta_to_AQ
from .period_finder import robust_period_search

import warnings
warnings.simplefilter("ignore")

def bic(reduced_chi2, N, k):
    dof = max(N - k, 1)
    chi2 = reduced_chi2 * dof
    return chi2 + k * np.log(max(N, 1))

def eval_on_grid(theta, n_bands, M_fit, n_grid = 200, coef_mode=None):
    phi_grid = np.arange(0, 1, 1/n_grid)
    # unpack theta
    _, amp, c1, c2, P, E = unpack_theta(theta, n_bands, M_fit=M_fit, include_amp=True, coef_mode=coef_mode)
    theta_rev = np.array([0, 1.0, c1, c2, 1., (E/P)%1], dtype=object) # unit period/amplitude + phase offset, enough to use single band
    fval  = F(theta_rev, phi_grid, M_fit, coef_mode=coef_mode)
    return phi_grid, fval

def spike_penalty(theta, n_bands, M_fit, coef_mode=None, n_grid=50, ratio = 0.05):
    # model value at uniform grid
    _, fval = eval_on_grid(theta, n_bands, M_fit, n_grid=n_grid, coef_mode=coef_mode)
    d2 = np.diff(fval, n=2)
    return np.percentile(np.abs(d2), 99)**2 + ratio * np.mean(d2**2)
    #pk_max = np.max(np.abs(d2)) # spike
    #pk_tot = np.sum(np.abs(d2)) # total variance
    #return pk_max + ratio * pk_tot

def harmonics_penalty(theta, n_bands, M_fit, coef_mode=None):
    _, _, c1, c2, _, _ = unpack_theta(theta, n_bands, M_fit=M_fit, include_amp=True, coef_mode=coef_mode)
    orders2 = (1.0 + np.arange(M_fit))**4
    return np.sum(orders2 * (c1**2 + c2**2))

def adjust_lambda(lam0, gmax, M_fit, N, lam_min = 1e-5, lam_max = 1e-1):
    # Heuristic function to adjust regularization weight
    lam = lam0 * (max(gmax, 0.05) / 0.12)**3 * (M_fit / 5.0)**2 * (30/max(N, 10))
    return np.clip(lam, lam_min, lam_max)

def fit_objective(theta, t, mag, emag, bmask, M_fit, n_dim, activated_bands,
                  lam_spike=0.0, lam_h=0.0, n_grid=50, coef_mode=None):
    n_bands = len(activated_bands)
    chi2_red = chisq(theta, t, mag, emag, bmask, M_fit, n_dim, activated_bands, coef_mode=coef_mode)
    pen_spike = spike_penalty(theta, n_bands, M_fit, n_grid=n_grid, coef_mode=coef_mode) # spike penalty
    pen_h = harmonics_penalty(theta, n_bands, M_fit, coef_mode=coef_mode) # harmonics penalty
    return chi2_red + lam_spike * pen_spike + lam_h * pen_h 

def _fit_wrapper(P0, args, M_fit, bounds_full, activated_bands, phase_gaps, 
                period_fit=False, use_optim=False, adaptive_lam=True, verbose=False):
    """
    Auxilary function to minimize objective function for given (P, M_fit)
    """
    t, mag, emag, bmask = args
    n_bands = len(activated_bands)
    coef_mode = _coef_mode() # use default
    n_dim = 2 * n_bands + 2 * M_fit + 2 # n_dim 
    
    # initial parameter - band별 계산 후 평균
    phase_flag = (phase_gaps > 0.05)
    N_fts = np.sum(bmask, axis = 1); N_eff = N_fts.max()
    lam = params.lam0
    if adaptive_lam:
        gmax = phase_gaps[activated_bands].max()
        lam = adjust_lambda(params.lam0, gmax, M_fit, N_eff, lam_min=params.lam_min, lam_max=params.lam_max) # set first band as pivotal band
        if verbose: print(f'phase_gap_max = {phase_gaps} / N_ft = {N_fts} / lam = {lam:.2e}')
    theta0 = LSQ_fit(P0, args, M_fit, activated_bands, phase_flag=phase_flag, 
                        opt_method = params.opt_method, lam = lam, quality_weight=params.quality_weight)

    P_init, E_init = theta0[-2], theta0[-1]
    if period_fit:
        # P와 E의 bound를 theta0
        bounds_full[-2] = (max(P_init * 0.9, params.pmin), min(P_init * 1.1, params.pmax)) # P 범위
    else: bounds_full[-2] = (P_init, P_init)
    bounds_full[-1] = (E_init - P_init, E_init + P_init)     # E 범위

    # 2. Global minimization
    if use_optim:
        res = minimize(fit_objective, theta0,
                    args=(t, mag, emag, bmask, M_fit, n_dim, activated_bands,
                          params.lam_spike, params.lam_h, params.n_grid, coef_mode),
                    method='L-BFGS-B',
                    bounds=bounds_full)
        if res.success:
            return res.x, res.fun #, M_fit, n_dim
        
    # optimization failure
    chi2_init = fit_objective(theta0, *args, M_fit=M_fit, n_dim=n_dim, activated_bands=activated_bands,
                              lam_spike=params.lam_spike, lam_h=params.lam_h, 
                              n_grid=params.n_grid, coef_mode=coef_mode)
    return theta0, chi2_init #, M_fit, n_dim

def _build_bounds(n_bands, M_fit, coef_mode=None):
    m_bounds = [(-np.inf, np.inf)] * n_bands
    a_bounds = [(params.Amin, params.Amax)] * n_bands
    coef_bounds = _harmonic_bounds(M_fit, coef_mode=coef_mode)
    P_bounds = [(params.pmin, params.pmax)] # (temporary)
    E_bounds = [(-np.inf, np.inf)] # (temporary)
    return m_bounds + a_bounds + coef_bounds + P_bounds + E_bounds

def calculate_m0_amp(args, sigma = 3.0, maxiter = 5):
    # calculate 1) mean magnitude and 2) peak-to-peak amplitude from multi-band epoch photometry
    # using for initial guess 
    t, mag, emag, bmask = args
    n_bands = len(bmask)
    m0s = np.zeros(n_bands); A0s = np.zeros(n_bands)
    resmasks = []
    for i, m in enumerate(bmask):
        mag_ft, emag_ft = mag[m], emag[m]
        if len(mag_ft)==0: 
            m0_ft = np.nan; Amp_ft = np.nan
            resmask = np.zeros_like(m, dtype=bool)
        else:
            w_ft = 1/np.maximum(emag_ft, params.ERR_FLOOR)**2
            n_prev = len(mag_ft)
            #m0_ft = np.average(mag_ft, weights = w_ft) # average: vulnerable for outlier
            m0_ft = np.median(mag_ft)
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
    return m0s, A0s, resmasks

def select_order(P0, args, activated_bands, phase_gaps, M_trunc,
                 period_fit=False, use_optim=False, adaptive_lam=False, verbose=False):
    t = args[0]
    n_bands = len(activated_bands)
    coef_mode = _coef_mode() # use default
    candidates = []
    M_ub = min([params.M_MAX, M_trunc + params.M_PAD])
    M_list = np.arange(params.M_MIN, M_ub+1)
    for M_fit in M_list:
        bounds = _build_bounds(n_bands, M_fit, coef_mode=coef_mode)
        theta_opt, obj_opt = _fit_wrapper(P0, args, M_fit, bounds, activated_bands, phase_gaps,
                                          period_fit=period_fit, use_optim=use_optim,
                                          adaptive_lam=adaptive_lam, verbose=False) # obj_opt != chi2
        _, fval_grid = eval_on_grid(theta_opt, n_bands, M_fit, coef_mode=coef_mode)
        chi2_red_opt = chisq(theta_opt, *args, M_fit, len(theta_opt),
                              activated_bands, coef_mode=coef_mode)
        score = bic(chi2_red_opt, len(t), len(theta_opt)) # BIC score
        candidates.append((M_fit, score, chi2_red_opt, obj_opt, theta_opt))
        if verbose:
            print(f"[Order scanning] M = {M_fit:2d} / obj = {obj_opt:.2f} / score = {score:.2f}")
    scores = np.array([x[1] for x in candidates])
    best_idx = int(np.argmin(scores))
    best_score = scores[best_idx]
    near = [x for x in candidates if (x[1] - best_score) <= getattr(params, 'ORDER_BIC_TOL', 2.0)]
    # tie-breaker: closest to M_trunc, then smaller order
    near.sort(key=lambda x: (abs(x[0] - M_trunc), x[0])) # x[0]: M_fit
    M_fit, score, chi2_red_opt, obj_opt, theta_opt = near[0]
    return M_fit, theta_opt, chi2_red_opt, obj_opt, score

# === Main Function ===
def fourier_decomp(sid, mode='ogle', init='lasso',
                   period_fit=False, use_optim=False, adaptive_lam=False, use_refit=False,
                   verbose=False, plot_LS=False, K=None, harmonics=None):
    # Load data
    if mode is None: mode = get_data_config().mode
    cfg = get_data_config(mode)
    filters = cfg.filters; activated_bands = cfg.activated_bands
    n_bands_full = cfg.n_bands_full; n_bands = cfg.n_bands # number of activated bands

    M_MAX, M_MIN = params.M_MAX, params.M_MIN

    if K is None: K = params.K
    if harmonics is None: harmonics = params.harmonics

    if mode=='ogle':
        sid_mask = (df_ident['ID'] == sid)
        pulsation = df_ident[sid_mask]['pulsation'].values[0]
    elif mode=='gaia':
        sid_mask = (df_ident['SOURCE_ID'] == sid)
        cep_type = df_ident['type_best_classification'][sid_mask][0]
        osc_type = df_ident['mode_best_classification'][sid_mask][0]
        pulsation = f'{cep_type}_{osc_type}'
    elif mode=='ztf':
        sid_mask = (df_ident['ID'] == sid)
        pulsation = 'NaN'

    t, mag, emag, bands = epoch_arrays(ls_data, sid, mode=mode)
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
        if verbose: print(f'RRFit best template:\n{tmpl}')

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
                                    n0 = params.n0, K = params.K, harmonics = params.harmonics, 
                                    snr = params.snr, plot = plot_LS)
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
    P0 = params.pmax # initialize (best period)
    for Pi, Zi in zip(P0s, Zs):
        #if Zi<0.2*Zmax: continue # non significant component
        # phase filling check
        phase_gaps_i = np.array([phase_gap_score(t[mask], Pi, M_fit=M_fit_1) for mask in bmask]) # maximum gap in phase domain
        
        """
        theta_init = None
        if init == 'rrfit': theta_init = theta0_rrfit
        """
        theta_1_tmp, chi2_1_tmp = _fit_wrapper(Pi, args, M_fit_1, bounds_1, activated_bands, phase_gaps = phase_gaps_i, 
                                               period_fit=period_fit, use_optim=use_optim, adaptive_lam=adaptive_lam, verbose=verbose)
                                               #theta0=theta_init)
        if verbose:
            print(f"{Pi:.4f} days / gmax = {phase_gaps_i} / chi2 = {chi2_1_tmp:.4f}")
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
    _, _, A_vec_1, _, _, _ = theta_to_AQ(theta_opt_1, n_bands, M_fit=M_fit_1, include_amp=True)

    M_trunc = M_MAX
    # To Do: Dealing the M_MIN?
    # currently, M_MIN = 3  (fixed)
    # for 1O/2O? - better to use M_MIN<3?
    if A_vec_1[0] > 1e-5: # not close to 0
        power = A_vec_1**2
        cum = np.cumsum(power) / np.sum(power)
        M_trunc = np.searchsorted(cum, 0.98) + 1
        M_trunc = np.clip(M_trunc, M_MIN, M_MAX)
        """
        dA = np.diff(A_vec_1)
        significant = np.where(dA/A_vec_1[:-1]>params.THRESHOLD)[0]
        if len(significant) > 0:
            M_trunc = 1 + np.min(significant)
            M_trunc = np.max([M_trunc, M_MIN])
        """
    else:
        M_trunc = M_MIN # M_min

    # --- secondary fitting ---
    if verbose:
        print(f'ID = {sid} / M_MAX fit done. Found M_trunc = {M_trunc}. Refitting...')
            
    M_fit_2 = M_trunc

    M_fit_final, theta_opt_final, chi2_opt_final, obj_opt_final, score_final = select_order(
        P0, args, activated_bands, phase_gaps, M_fit_2,
        period_fit=period_fit, use_optim=use_optim,
        adaptive_lam=adaptive_lam, verbose=verbose)

    m0, amp, A_fit, Q_fit, P, E = theta_to_AQ(theta_opt_final, n_bands, M_fit=M_fit_final,
                                              include_amp=True, coef_mode=_coef_mode())
    tmpl_amplitude = peak_to_peak_amplitude(A_fit, Q_fit, M_fit=M_fit_final, coef_mode='AQ')
    A_fit /= tmpl_amplitude
    amp *= tmpl_amplitude

    # =====================================
    # 3) summary statistics
    # =====================================
    # zero-padding (to fit M_MAX)
    A_out = np.zeros(M_MAX)
    Q_out = np.zeros(M_MAX)
    A_out[:M_fit_final] = A_fit
    Q_out[:M_fit_final] = Q_fit
    
    flag = 0
    N, sig, rms = np.zeros(n_bands_full, dtype='int'), np.zeros(n_bands_full), np.zeros(n_bands_full)
    phi_rise = np.nan
    
    m0_out, amp_out = np.zeros(n_bands_full), np.zeros(n_bands_full)
    theta_fit_tmpl = (A_fit, Q_fit) 
    for i, flt in enumerate(filters):
        mask = bmask[i]
        if not np.any(mask):
            N[i] = 0
            flag = 1 # no data
            continue
            
        t_ft, mag_ft, emag_ft = t[mask], mag[mask], emag[mask]
        N[i] = len(t_ft)
        
        w_ft = 1 / np.maximum(emag_ft, params.ERR_FLOOR)**2
        # photometric error (w/o genuine pulsation)
        dm_ft = mag_ft-m0_data[i]
        sig[i] = np.sqrt(np.average(dm_ft**2, weights=w_ft))

        # amplitude refitting`
        phi_ft = ((t_ft - E)/P)%1
        h_ft = H(theta_fit_tmpl, phi_ft, M_fit_final, coef_mode='AQ')
        if i in activated_bands: 
            ib = np.where(activated_bands == i)[0][0]
            m0_init = m0[ib]; amp_init = amp[ib]
        else: m0_init = m0_data[i]; amp_init = amp_data[i] # not used for Fourier coefficient calculations
        m0_out[i] = m0_init; amp_out[i] = amp_init
        if use_refit and (len(h_ft)>5):
            good = np.ones(len(h_ft),dtype=bool)
            for _ in range(params.REFIT_MAXITER):
                # refit (m0, amp)
                # these chi2_ampl values are evaluated only with "good" values.
                m0_out[i], amp_out[i], chi2_ampl = refit_m0_amp(h_ft, mag_ft, w_ft, opt_method='lsq',
                                                               good=good)
                amp_ratio = amp_out[i] / amp_data[i]
                if (0.7 > amp_ratio) or (1.3 < amp_ratio):
                    amp_lb, amp_ub = params.Amin, params.Amax # default (naive amplitude boundary)
                    m0_out[i], amp_out[i], chi2_ampl = refit_m0_amp(h_ft, mag_ft, w_ft, opt_method='optim',
                                                                    m0_init=m0_data[i], amp_init=amp_data[i], good=good,
                                                                    amp_bounds = (amp_lb, amp_ub))
                # sigma clip
                f_ft = m0_out[i] + amp_out[i] * h_ft
                resid_ft = mag_ft - f_ft
                resmask = sigma_clip(resid_ft, sigma=params.REFIT_SIGMA, masked=True).mask
                n_curr = (~resmask).sum()

                if np.array_equal(~resmask, good) or n_curr < 5: break
                good = ~resmask

            if verbose:
                print(f"[Refinement / {flt}] used = {n_curr}/{len(h_ft)} | m0 = {m0_init:.4f} -> {m0_out[i]:.4f} | amp = {amp_init:.4f} -> {amp_out[i]:.4f} (ratio(data) = {amp_ratio:.2f}) | CHI2 = {chi2_ampl:.2f}")
            
        else:
            f_ft = m0_out[i] + amp_out[i] * h_ft
            resid_ft = mag_ft - f_ft

        # residual 
        rms[i] = np.sqrt(np.average(resid_ft**2, weights=w_ft))

        # parameter boundary excession check (peak-to-peak amplitude only)
        amp_lb, amp_ub = params.Amin, params.Amax
        if (amp_out[i] > amp_ub) or (amp_out[i] < amp_lb): flag = 1

        m_lb, m_ub = min(mag_ft), max(mag_ft)
        if (m0_out[i] < m_lb) or (m0_out[i] > m_ub): flag = 1

        # Calculate rising time
        """
        if filters[ib] == 'I':
            m_pk = np.percentile(mag_ft,[99,1])
            phase_pk = (t_ft[np.argmin(abs(mag_ft-m_pk[:,None]),axis=1)]/P0)%1 
            delta_phi = np.diff(phase_pk)[0]
            phi_rise = np.min([1-delta_phi, delta_phi])
        """
    # ToDo:  after amplitude refinement, chi2 value should be reevaluated!!
    if verbose:
        print(f'ID = {sid} / Final M_fit = {M_fit_final} / CHI2 = {chi2_opt_final:.2f} / F_obj = {obj_opt_final:.2f} / rrms = {rms[0]/sig[0]:.4f} / P = {P:.6f} days')

    theta_params_out = np.hstack([m0_out, amp_out, A_out, Q_out])
     #return m0, amp, A, Q, P, E, M_fit_final
    row = [sid, pulsation, *N, *sig, *rms, *phase_gaps, Zmax, P0, chi2_opt_final, obj_opt_final, 
           P, E, phi_rise, M_fit_final, *theta_params_out, flag]
    return row

"""
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

    m0, amp, A_fit, Q_fit, P, E = unpack_theta(theta_opt_final, n_bands, M_fit=M_fit_final, include_amp=True)
    """