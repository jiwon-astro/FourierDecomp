import numpy as np

from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.linalg import lstsq, norm
from sklearn.linear_model import Lasso

from . import params

# === helper functions ===
def _coef_mode(coef_mode=None):
    if coef_mode is None: # (A,Q) or (alpha, beta)
        coef_mode = getattr(params, 'coef_mode', 'ab')
    return str(coef_mode)

def _harmonic_bounds(M_fit, coef_mode=None):
    coef_mode =_coef_mode(coef_mode)
    Amax = getattr(params, "Amax_harmonic", getattr(params, "Amax", 1.0))
    Amax = np.broadcast_to(np.asarray(Amax, dtype=float), (M_fit,)).copy()
    if coef_mode=='AQ': 
        A_bounds = list(zip(np.zeros(M_fit), Amax))
        Q_bounds =  [(0, 2 * np.pi)] * M_fit
        return A_bounds + Q_bounds
    else: return [(-u, u) for u in Amax]*2 

def AQ_to_ab(A,Q):
    alpha = A * np.cos(Q)
    beta  = A * np.sin(Q)
    return alpha, beta

def ab_to_AQ(alpha, beta):
    A = np.hypot(alpha, beta) # (alpha^2 + beta^2)^1/2
    Q = np.arctan2(beta, alpha) %  (2*np.pi) # atan2(sin, cos)
    return A, Q

# === Design matrix for Fourier series ===
def cs_matrix(t, P, E, M_fit):
    N = len(t)
    X = np.ones((1 + 2*M_fit, N)) # updated : fourier series order is adjusted by M_fit
    orders = 1 + np.arange(M_fit).reshape(-1, 1) # Fourier series order
    phi = 2 * np.pi * ((t - E) / P % 1.0) * orders 
    X[1::2] = np.cos(phi)
    X[2::2] = np.sin(phi)
    return X.T

# === Penalized least squares for linear parameters ===
def reg_lsq(X, y, w, return_sol = False):
    """
    Solve (X^T W X + lam*I) sol = X^T W y
    Return m0, A_vec, Q_vec
    """
    W = np.sqrt(w)
    Xw = X * W[:,None]
    yw = y * W
    A_mat = Xw.T.dot(Xw)
    #pen = np.concatenate(([0], np.repeat(orders**2, 2))) # k^2 penalty -> worse performance
    #A_mat += lam * np.diag(pen)
    b = Xw.T.dot(yw)
    
    try:
        sol = np.linalg.solve(A_mat, b)
    except np.linalg.LinAlgError:
        # Fallback to slower lstsq if solve fails
        sol = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        
    if return_sol: return sol
    else:
        # conversion
        m0 = sol[0]
        coeffs = sol[1:]
        A_vec, Q_vec = ab_to_AQ(coeffs[0::2], coeffs[1::2])
        return m0, A_vec, Q_vec

def reg_lsq_lasso(X, y, w, lam, beta_init, return_sol = False):
    """
    Solve L1-penalized (LASSO) weighted least squares.
    Objective: sum(w * (y - X.dot(beta))**2) + lam * sum(|beta[1:]|)
    
    Uses 'beta_init' (the WLS solution from reg_lsq) as the starting point.
    sklearn.linear_model.Lasso (more bareable to dealing the sharp corner of the L1 penalty
    """
    X_body = X[:, 1:]
    X_intercept = X[:, 0] # 1-vector
    
    if lam < 1e-9: sol = beta_init
    else: 
        #*intercept -> m0
        model = Lasso(alpha=lam, fit_intercept=True, 
                      warm_start=True, tol=1e-5, max_iter=3000)
        # initialization
        model.coef_ = beta_init[1:]
        model.intercept_ = beta_init[0]
        
        try:
            model.fit(X_body, y, sample_weight=w)
            
            sol = np.zeros(X.shape[1])
            sol[0] = model.intercept_
            sol[1:] = model.coef_
            
            # --- Sparsity ---
            # small coefficients = 0  (hard thresholding).
            sol[1:][np.abs(sol[1:]) < 1e-7] = 0.0 # 
        except:
            # Fallback or warning
            print(f"Warning: L1 minimization in reg_lsq failed (lam={lam}). Using initial LSQ guess.")
            sol = beta_init
    
    # --- Conversion ---
    if return_sol: return sol
    else: 
        m0 = sol[0]
        coeffs = sol[1:]
        A_vec, Q_vec = ab_to_AQ(coeffs[0::2], coeffs[1::2])
        return m0, A_vec, Q_vec

# === Vectorized Fourier Series === 
def H(theta, t, M_fit = None, coef_mode=None):
    # unit template function
    coef_mode = _coef_mode(coef_mode)
    if M_fit is None: M_fit = params.M_MAX
    # c1, c2: fourier coefficients (size M vector) 
    if len(theta) != 2:
        raise ValueError(f"F(): theta length={len(theta)} (expected 2)")
    c1, c2 = theta
    P = 1.0  # unit period
    phi = t / P % 1.0  # to make numerically stable
    orders = 1 + np.arange(M_fit).reshape(-1, 1) # Fourier series order
    phase = 2 * np.pi * phi * orders
    C, S = np.cos(phase), np.sin(phase)
    # Fourier Coefficients: (alpha, beta)
    if coef_mode == 'AQ':
        alpha, beta = AQ_to_ab(c1, c2) 
    else: # already (alpha, beta)
        alpha, beta = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)    
    return alpha @ C + beta @ S 

def F(theta, t, M_fit, coef_mode=None): 
    # m0 : mean magnitude
    # amp0 : peak-to-peak amplitude
    # c1, c2 : fourier coefficients (size M vector, (A,Q) or (alpha, beta)) 
    # P : period
    # E : epoch 
    coef_mode = _coef_mode(coef_mode)
    if len(theta) != 6:
        raise ValueError(f"F(): theta length={len(theta)} (expected 6)")
    m0, amp0, c1, c2, P, E = theta
    phi = (t - E) / P % 1.0  # to make numerically stable
    orders = 1 + np.arange(M_fit).reshape(-1, 1) # Fourier series order
    phase = 2 * np.pi * phi * orders
    C, S = np.cos(phase), np.sin(phase)

    # Fourier Coefficients: (alpha, beta)
    if coef_mode == 'AQ':
        alpha, beta = AQ_to_ab(c1, c2) 
    else: # already (alpha, beta)
        alpha, beta = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)

    return m0 + amp0 * (alpha @ C + beta @ S)

def unpack_theta(theta, n_bands, M_fit, P=False, include_amp=True,
                 coef_mode=None): # M_fit
    # return (m0, amp, c1, c2, P, E) tuple (len=6)
    idx = 0
    m0 = theta[idx:idx+n_bands]; idx += n_bands
    if include_amp: amp = theta[idx:idx+n_bands]; idx += n_bands
    else: amp = np.ones(n_bands)
    c1 = theta[idx:idx+M_fit]; idx += M_fit # M_fit
    c2 = theta[idx:idx+M_fit]; idx += M_fit # M_fit
    E = theta[-1]
    if (P is False) or (P is None): P = theta[-2]
    return m0, amp, c1, c2, P, E

def theta_to_AQ(theta, n_bands, M_fit, include_amp=True, coef_mode=None):
    # ensure coefficient type as (A, Q)
    coef_mode = _coef_mode(coef_mode)
    m0, amp, c1, c2, P, E = unpack_theta(theta, n_bands, M_fit, 
                                         include_amp=include_amp, coef_mode=coef_mode)
    if coef_mode == 'AQ': A, Q = c1, c2
    else: A, Q = ab_to_AQ(c1, c2)
    return m0, amp, A, Q, P, E

def peak_to_peak_amplitude(c1, c2, Nx=1001, M_fit=None, coef_mode=None): # M_fit
    coef_mode = _coef_mode(coef_mode)
    if M_fit is None: M_fit = len(c1) 
    t = np.linspace(0, 1, Nx)
    theta = (0.0, 1.0, c1, c2, 1.0, 0.0)
    fval = F(theta, t, M_fit, coef_mode=coef_mode)
    amp = np.diff(np.percentile(fval,[1, 99]))[0] # 1-99 percentile
    #amp = np.max(fval) - np.min(fval) # P-t-P
    return amp

# === chisq function - fitness evaluation ===
def chisq_single(theta, t, mag, emag, M_fit, P0=False, 
                 include_amp=True, unpack=True, coef_mode=None): 
    coef_mode = _coef_mode(coef_mode)
    if unpack:
        # convert into [m0, amp, A, Q, P, E] (len=6)
        m0, amp, c1, c2, P, E = unpack_theta(theta, n_bands=1, M_fit=M_fit,
                                             P=P0, include_amp=include_amp, coef_mode=coef_mode)
        theta = np.array([m0[0], amp[0], c1, c2, P, E], dtype=object)# assuming m0, A0, single value
    fval = F(theta, t, M_fit, coef_mode=coef_mode)
    resid = (mag - fval) / (np.maximum(emag, params.ERR_FLOOR))
    return np.sum(resid**2)

def chisq(theta, t, mag, emag, bmask, M_fit, n_dim, activated_bands, coef_mode=None): # M_fit, n_dim 
    summ = 0; Nx = 0
    coef_mode = _coef_mode(coef_mode)
    n_bands = len(activated_bands)
    m0, amp0, c1, c2, P, E = unpack_theta(theta, n_bands, M_fit=M_fit, 
                                          include_amp=True, coef_mode=coef_mode)
    for i, ib in enumerate(activated_bands):
        theta_b = [m0[i], amp0[i], c1, c2, P, E]
        mask = bmask[ib]; Nx += np.sum(mask)
        summ += chisq_single(theta_b, t[mask], mag[mask], emag[mask],
                             M_fit=M_fit, unpack=False, coef_mode=coef_mode)
    dof = Nx - n_dim
    if dof <= 0: return np.inf #insufficient data point
    return summ / dof # Reduced chi-square

# === LSQ fits ===
def LSQ_fit(P0, args, M_fit, activated_bands, opt_method='lsq', quality_weight=False,
             bounds=None, phase_flag=None, Nmin=50, lam=1e-3, coef_mode=None): # M_fit
    # phase_flag: having a large phase gap in the phase-folded light curve
    coef_mode = _coef_mode(coef_mode)
    n_bands = len(activated_bands)

    t, mag, emag, bmask = args
    m0 = np.zeros(n_bands)
    amp0 = np.zeros(n_bands)
    alpha_accum = np.zeros(M_fit) # M_fit, AcosQ
    beta_accum = np.zeros(M_fit) # M_fit,  AsinQ
    count = np.zeros(M_fit)   # M_fit
    E0 = None
    eps = 1e-6

    for i, ib in enumerate(activated_bands):
        mask = bmask[ib]
        if not np.any(mask): continue
        t_ft, mag_ft, emag_ft = t[mask], mag[mask], emag[mask]
        
        if E0 is None:
            t0 = t_ft[0]
            E0 = t0 + (t_ft[np.argmin(mag_ft)] - t0) % P0
            
        X_ft = cs_matrix(t_ft, P0, E0, M_fit)
        w_ft = 1.0 / np.maximum(emag_ft, params.ERR_FLOOR)**2

        beta_init = reg_lsq(X_ft, mag_ft, w_ft, return_sol=True)
        if opt_method == 'lsq': sol = beta_init
        else: sol = reg_lsq_lasso(X_ft, mag_ft, w_ft, lam=lam, beta_init=beta_init, return_sol=True)

        m0_ft = sol[0]
        alpha_ft = sol[1::2].copy()
        beta_ft = sol[2::2].copy()
            
        res_success = False # minimization success flag
        if phase_flag is not None:
            if phase_flag[ib]:
                if coef_mode == 'AQ':
                    A_ft, Q_ft = ab_to_AQ(alpha_ft, beta_ft)
                    theta_ft = np.hstack([m0_ft, A_ft, Q_ft, E0]) # amp=1 (False) 기준
                else: theta_ft = np.hstack([m0_ft, alpha_ft, beta_ft, E0])

                # bounds_single: [m0, A1..AM, Q1..QM, E]
                hb = _harmonic_bounds(M_fit, coef_mode=coef_mode)
                bounds_single = [(-np.inf, np.inf)] + hb + [(E0 - P0, E0 + P0)]

                # dimensionality check
                if len(bounds_single) != len(theta_ft):
                    raise ValueError(
                        f"bounds/theta mismatch: theta_ft: dim = {len(theta_ft)}, "
                        f"bounds_single: dim = {len(bounds_single)}, "
                        f"coef_mode = {coef_mode}, M_fit = {M_fit}, hb = {len(hb)}"
                    )

                res = minimize(chisq_single, theta_ft,
                               args=(t_ft, mag_ft, emag_ft, M_fit, P0, False, True), 
                               method='L-BFGS-B', bounds=bounds_single)
                
                if res.success:
                    m0_ft_res, _, c1_ft, c2_ft, P0, E2 = unpack_theta(res.x, n_bands=1, M_fit=M_fit, 
                                                                    P=P0, include_amp=False)
                    m0_ft = m0_ft_res[0]
                    if coef_mode == 'AQ': alpha_ft, beta_ft = AQ_to_ab(c1_ft, c2_ft)
                    else: alpha_ft, beta_ft = c1_ft, c2_ft
                    E0 = E2
                    res_success = True

        scale = peak_to_peak_amplitude(alpha_ft, beta_ft, M_fit=M_fit, coef_mode='ab')
        if scale < eps: scale = eps
        
        m0[i] = m0_ft
        amp0[i] = scale


        w_band = 1.0
        if quality_weight:
            # evaluate residual 
            theta_ft = [m0_ft, 1, alpha_ft, beta_ft, P0, E0] 
            fval = F(theta_ft, t_ft, M_fit, coef_mode='ab')
            resid_ft = mag_ft - fval
            rms = np.sqrt(np.average(resid_ft**2, weights=w_ft))
            w_band = 1/rms**2
        
        if (not phase_flag[ib]) or (phase_flag[ib] and res_success):
            # phase_flag = False -> LSQ fit only (its sufficient)
            # phase_flag = True -> secondary fitting with minimization -> reliable peak-to-peak amplitude from best fit solution
            alpha_accum += alpha_ft / scale
            beta_accum += beta_ft / scale
            count += w_band

    valid = count > 0
    alpha0 = np.zeros(M_fit)
    beta0 = np.zeros(M_fit)
    alpha0[valid] = alpha_accum[valid] / count[valid]
    beta0[valid]  = beta_accum[valid] / count[valid]

    if coef_mode == 'AQ':
            A0, Q0 = ab_to_AQ(alpha0, beta0)
            theta0 = np.hstack([m0, amp0, A0, Q0, P0, E0])
    else:
        theta0 = np.hstack([m0, amp0, alpha0, beta0, P0, E0])
    return theta0