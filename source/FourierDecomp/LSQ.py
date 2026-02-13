import numpy as np

from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.linalg import lstsq, norm

from .params import ORDERS_MAX, ERR_FLOOR, Amin, Amax, activated_bands, n_bands

# === Design matrix for Fourier series ===
def cs_matrix(t, P, E, M_fit):
    N = len(t)
    X = np.ones((1 + 2*M_fit, N)) # updated : fourier series order is adjusted by M_fit
    phi = 2 * np.pi * ((t - E) / P % 1.0) * ORDERS_MAX[:M_fit] 
    X[1::2] = np.cos(phi)
    X[2::2] = np.sin(phi)
    return X.T

# === Penalized least squares for linear parameters ===
def reg_lsq(X, y, w, lam):
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
    sol = np.linalg.solve(A_mat, b)
    # conversion
    m0 = sol[0]
    coeffs = sol[1:]
    A_vec = np.hypot(coeffs[0::2], coeffs[1::2])
    Q_vec = (np.arctan2(coeffs[1::2], coeffs[0::2]) % (2*np.pi))
    return m0, A_vec, Q_vec

# === Vectorized Fourier Series === 
def F(theta, t, M_fit): 
    # m0 : mean magnitude
    # P : period
    # E : epoch 
    # A, Q : fourier coefficients (size M vector) 
    m0, amp0, A, Q, P, E = theta
    phi = (t - E) / P % 1.0  # to make numerically stable
    phase = 2 * np.pi * phi * ORDERS_MAX[:M_fit] 
    C, S = np.cos(phase), np.sin(phase)
    alpha, beta = A * np.cos(Q), A * np.sin(Q) # Fourier Coefficients
    return m0 + amp0 * (alpha @ C + beta @ S)

def unpack_theta(theta, n_bands, M_fit, P=False, include_amp=True): # M_fit
    idx = 0
    m0 = theta[idx:idx+n_bands]; idx += n_bands
    if include_amp: amp = theta[idx:idx+n_bands]; idx += n_bands
    else: amp = np.ones(n_bands)
    A = theta[idx:idx+M_fit]; idx += M_fit # M_fit
    Q = theta[idx:idx+M_fit]; idx += M_fit # M_fit
    E = theta[-1]
    if (P is False) or (P is None): P = theta[-2]
    return m0, amp, A, Q, P, E

def peak_to_peak_amplitude(A, Q, Nx=1001, M_fit=None): # M_fit
    if M_fit is None: M_fit = len(A) 
    t = np.linspace(0, 1, Nx)
    theta = (0, 1, A, Q, 1, 0)
    fval = F(theta, t, M_fit)
    amp = np.diff(np.percentile(fval,[1, 99])) # 1-99 percentile
    #amp = np.max(fval) - np.min(fval) # P-t-P
    return amp

# === chisq function - fitness evaluation ===
def chisq_single(theta, t, mag, emag, M_fit, P0=False, include_amp=True, unpack=True): 
    if unpack:
        # convert into [m0, amp, A, Q, P, E]
        m0, amp, A, Q, P, E = unpack_theta(theta, n_bands=1, M_fit=M_fit,
                                           P=P0, include_amp=include_amp)
        theta = np.array([*m0, *amp, *A, *Q, P, E])
    fval = F(theta, t, M_fit)
    resid = (mag - fval) / (np.maximum(emag, ERR_FLOOR))
    return np.sum(resid**2)

def chisq(theta, t, mag, emag, bmask, M_fit, n_dim): # M_fit, n_dim 
    summ = 0
    Nx = 0
    m0, amp0, A, Q, P, E = unpack_theta(theta, n_bands, M_fit=M_fit, include_amp=True)
    for i, ib in enumerate(activated_bands):
        theta_b = [m0[i], amp0[i], A, Q, P, E]
        mask = bmask[ib]; Nx += np.sum(mask)
        summ += chisq_single(theta_b, t[mask], mag[mask], emag[mask],
                             M_fit=M_fit, unpack=False)
    
    dof = Nx - n_dim
    if dof <= 0: return np.inf #insufficient data point
    return summ / dof # Reduced chi-square

# === LSQ fits ===
def LSQ_fit(P0, args, M_fit, bounds=None,
                phase_flag=None, Nmin=50, lam=1e-3): # M_fit
    t, mag, emag, bmask = args
    m0 = np.zeros(n_bands)
    amp0 = np.zeros(n_bands)
    A_accum = np.zeros(M_fit) # M_fit
    Q_accum = np.zeros(M_fit) # M_fit
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
        w_ft = 1.0 / np.maximum(emag_ft, ERR_FLOOR)**2
        
        m0_ft, A_ft, Q_ft = reg_lsq(X_ft, mag_ft, w_ft, lam)
        
        res_success = False # minimization success flag
        if phase_flag is not None:
            if phase_flag[ib]:
                theta_ft = np.hstack([m0_ft, A_ft, Q_ft, E0]) # amp=1 (False) 기준
                
                # bounds_single: [m0, A1..AM, Q1..QM, E]
                bounds_single = [(-np.inf, np.inf)] + [(Amin, Amax)] * M_fit + [(0, 2 * np.pi)] * M_fit + [(E0 - P0, E0 + P0)]
                
                res = minimize(chisq_single, theta_ft,
                               args=(t_ft, mag_ft, emag_ft, M_fit, P0, False, True), 
                               method='L-BFGS-B', bounds=bounds_single)
                
                if res.success:
                    m0_ft_res, _, A_ft, Q_ft, P0, E2 = unpack_theta(res.x, n_bands=1, M_fit=M_fit, 
                                                                    P=P0, include_amp=False)
                    m0_ft = m0_ft_res[0]
                    E0 = E2
                    res_success = True

        scale = peak_to_peak_amplitude(A_ft, Q_ft, M_fit=M_fit)
        if scale < eps: scale = eps
        
        m0[i] = m0_ft
        amp0[i] = scale
        
        if (not phase_flag[ib]) or (phase_flag[ib] and res_success):
            A_accum += A_ft / scale
            Q_accum += Q_ft
            count += 1

    valid = count > 0
    A0 = np.zeros_like(A_accum)
    Q0 = np.zeros_like(Q_accum)
    A0[valid] = A_accum[valid] / count[valid]
    Q0[valid] = Q_accum[valid] / count[valid]

    theta0 = np.hstack([m0, amp0, A0, Q0, P0, E0])
    return theta0