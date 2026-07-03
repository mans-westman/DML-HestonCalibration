
import numpy as np
import pandas as pd



def cumulant_truncation(L,T,r,q,theta,v0,sigma,rho,kappa,x):

    # 1. First cumulant (c1): drift

    ekt = np.exp(-kappa * T)

    # Expected integrated variance
    avg_var_integral = (v0 - theta)*(1 - ekt)/kappa + theta*T

    c_1 = x + (r - q) * T - 0.5 * avg_var_integral

    # 2. Second cumulant (c2). Fallback: c2 can be approximated by v0*T
    # (as done in Oosterlee & Fang), but the full expression is used here.
    c2_term1 = sigma * ekt * (v0 - theta) * (8*kappa*rho - 4*sigma)

    c2_term2 = kappa * rho * sigma * (1 - ekt) * (16*theta - 8*v0)

    c2_term3 = 2 * theta * kappa * T * (-4*kappa*rho*sigma + sigma**2 + 4*kappa**2)

    c2_term4 = sigma**2 * ( (theta - 2*v0) * np.exp(-2*kappa*T) + \
                            theta * (6*ekt - 7) + 2*v0 )

    c2_term5 = 8 * kappa**2 * (v0 - theta) * (1 - ekt)

    c2_scalar = 1 / (8 * kappa**3)
    c_2 = c2_scalar * (c2_term1 + c2_term2 + c2_term3 + c2_term4 + c2_term5)

    # 3. Truncation range.
    # Floor c_2 to a small positive value: it is mathematically positive but
    # can turn slightly negative near T -> 0 from floating-point error.
    c_2 = np.maximum(c_2, 1e-6)

    # Fourth cumulant, Gatheral (2006).
    c_4 = ( 12 * (sigma**4) * theta * T ) /  ( kappa ** 3 )

    # Widen the interval by the correlation and vol-of-vol to cover fat tails.
    tail_factor = np.sqrt(c_2 + np.sqrt(c_4)) + abs(rho) + abs(sigma)

    a = c_1 - L * tail_factor
    b = c_1 + L * tail_factor

    return a, b



def chi_k(k,a,b,c,d):

    """
    Cosine-series coefficient chi_k: the analytical integral of
    e^y * cos(k*pi*(y-a)/(b-a)) over [c, d]. Used for the exponential part
    of the option payoff in the COS expansion.

    Inputs
    ------
    k : int or array
        Cosine series index.
    a : float
        Lower bound of the truncation interval.
    b : float
        Upper bound of the truncation interval.
    c : float
        Lower bound of the payoff support.
    d : float
        Upper bound of the payoff support.

    Returns
    -------
    float or array
        Value of chi_k.
    """

    omega_k = k * np.pi / (b - a)
    scaling = 1.0 / (1.0 + omega_k**2)

    cos_d = np.cos(omega_k*(d-a))
    sin_d = np.sin(omega_k*(d-a))
    cos_c = np.cos(omega_k*(c-a))
    sin_c = np.sin(omega_k*(c-a))

    exp_d = np.exp(d)
    exp_c = np.exp(c)

    term_d = exp_d * (cos_d + omega_k * sin_d)
    term_c = exp_c * (cos_c + omega_k * sin_c)

    return scaling * (term_d - term_c)


def psi_k(k,a,b,c,d):

    """
    Cosine-series coefficient psi_k: the analytical integral of
    cos(k*pi*(y-a)/(b-a)) over [c, d]. Used for the constant part of the
    option payoff in the COS expansion. Expressed via numpy.sinc.

    Inputs
    ------
    k : int or array
        Summation index.
    a : float
        Lower bound of truncation.
    b : float
        Upper bound of truncation.
    c : float
        Lower bound of integral.
    d : float
        Upper bound of integral.

    Returns
    -------
    float or array
        Value of psi_k.
    """

    arg_d = k * (d - a) / (b - a)
    arg_c = k * (c - a) / (b - a)

    term_d = (d - a) * np.sinc(arg_d)
    term_c = (c - a) * np.sinc(arg_c)

    return term_d - term_c

def charact_func(omega,x,r,q,rho,sigma,kappa,v0,theta,T):
    """
    The characteristic function of the Heston model, using the numerically
    stable "little trap" form (Albrecher et al.) to avoid branch-cut issues.

    Inputs
    ------
    omega : array
        Argument of the characteristic function.
    x : float
        Log-moneyness log(S/K).
    r : float
        Interest rate.
    q : float
        Dividend rate.
    rho : float
        Correlation between stock and volatility.
    sigma : float
        Vol of vol.
    kappa : float
        Rate of mean-reversion.
    v0 : float
        Initial variance.
    theta : float
        Long-term variance.
    T : float
        Time to maturity.

    Returns
    -------
    complex or array
        Value of the characteristic function.
    """

    W = kappa - 1j * rho * sigma * omega
    d = np.sqrt(W**2 + (omega**2 + 1j * omega) * (sigma**2))

    # Stability fix (Albrecher form)
    c = (W - d) / (W + d)


    comp = 1j * omega * x # The shift for log(S/K)
    drift_term = 1j * omega * (r - q) * T

    exp_DT = np.exp(-d * T)
    div_term = 1 - c * exp_DT

    log_part = np.log(div_term / (1 - c))

    D = ((W-d) / sigma**2) * (1 - exp_DT) / div_term

    # theta part (stable form)
    part_theta = (kappa * theta / sigma**2) * (
        (W - d) * T - 2 * log_part
    )

    C = part_theta + drift_term

    return np.exp(C + D*v0 + comp)


def V_k(K,k,a,b,flag):

    # Cosine-series coefficients of the payoff (COS article).
    # This formulation preserves vectorization.
    M = 2.0 / (b - a)

    # Only puts are priced here; call prices are recovered via put-call parity
    # by the caller. The call term is kept for reference.
    # term_call = chi_k(k, a, b, 0, b) - psi_k(k, a, b, 0, b)
    term_put  = -chi_k(k, a, b, a, 0) + psi_k(k, a, b, a, 0)

    return M * K * term_put



def heston_cos_vectorized(x,K,T,kappa,v0,theta,rho,sigma,r,q,N,L,flag):

    """
    Vectorized Heston option pricer via the COS method.

    Inputs
    ------
    x : array
        Log-moneyness log(S/K).
    K : array
        Strike.
    T : array
        Time to maturity.
    kappa, v0, theta, rho, sigma : array
        Heston parameters.
    r : array
        Interest rate.
    q : array
        Dividend rate.
    N : int
        Number of integration (cosine series) terms.
    L : float
        Truncation-range width factor.
    flag : array
        1 for call, 0 for put.

    Returns
    -------
    (array, array)
        Option prices and a per-sample numerical-stability mask.
    """
    flag = flag.astype(np.float64)

    batch_inputs = [x, K, T, kappa, v0, theta, rho, sigma, r, q, flag]

    batch_inputs = [np.atleast_1d(i).reshape(-1, 1) for i in batch_inputs] # column form

    x, K, T, kappa, v0, theta, rho, sigma, r, q, flag = batch_inputs

    a,b = cumulant_truncation(L,T,r,q,theta,v0,sigma,rho,kappa,x)

    # Summation index
    k = np.arange(N).reshape(1, -1)
    omega = k*np.pi / (b-a)
    character_func = charact_func(omega, x, r, q, rho, sigma, kappa, v0, theta, T)

    # Alternative: for stability, always price puts and convert to calls via
    # put-call parity (Call = Put + S0*exp(-qT) - K*exp(-rT), with S0 = K*exp(x)).
    # The active path below prices puts directly.

    Vk = V_k(K,k,a,b,flag)

    integrand = character_func * Vk * np.exp(-1j*omega*a)

    v = np.exp(-r*T) * np.real( 0.5*integrand[:,0:1] + np.sum(integrand[:,1:],axis=1,keepdims=True))
    v = v.squeeze()

    # --- Stability test ---

    tail_energy = np.sum(np.abs(integrand[:, -8:]), axis=1)

    scaled_tail = tail_energy / (np.abs(v) + 1e-8)

    # Stability criteria
    stable_mask = (tail_energy < 1e-6) | (scaled_tail < 1e-7)

    if v.size == 1:
        return v.item(), stable_mask.item()

    # stable_mask is auxiliary output
    return v, stable_mask
