"""
JAX implementation of the Heston COS pricer, providing exact parameter
gradients (Greeks) via automatic differentiation.

Requires JAX: pip install --upgrade "jax[cpu]" jaxlib
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

# Use 64-bit for better numerical accuracy
jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd

def smooth_maximum(x, epsilon=1e-8, beta=1e8):
    """
    A continuously differentiable alternative to jnp.maximum(x, epsilon).
    Requires jax_enable_x64=True for high beta values.
    """
    # jax.nn.softplus(z) is numerically stable for large z
    return epsilon + jax.nn.softplus(beta * (x - epsilon)) / beta

def cumulant_truncation_jax(L, T, r, q, theta, v0, sigma, rho, kappa, x):
    # Same structure as COS_nojax, using jax.numpy (jnp).
    ekt = jnp.exp(-kappa * T)
    avg_var_integral = (v0 - theta)*(1 - ekt)/kappa + theta*T
    c1 = x + (r - q) * T - 0.5 * avg_var_integral

    c2_term1 = sigma * ekt * (v0 - theta) * (8 * kappa * rho - 4 * sigma)
    c2_term2 = kappa * rho * sigma * (1 - ekt) * (16 * theta - 8 * v0)
    c2_term3 = 2 * theta * kappa * T * (-4 * kappa * rho * sigma + sigma**2 + 4 * kappa**2)
    c2_term4 = sigma**2 * ((theta - 2 * v0) * jnp.exp(-2 * kappa * T) + theta * (6 * ekt - 7) + 2 * v0)
    c2_term5 = 8 * kappa**2 * (v0 - theta) * (1 - ekt)
    c2_scalar = 1.0 / (8.0 * kappa**3)
    c_2 = c2_scalar * (c2_term1 + c2_term2 + c2_term3 + c2_term4 + c2_term5)

    # Use a smooth floor rather than jnp.maximum, which would zero the gradient.
    c_2 = smooth_maximum(c_2, epsilon=1e-8, beta=1e8)

    c4 = (12.0 * (sigma**4) * theta * T) / (kappa ** 3)
    tail_factor = jnp.sqrt(c_2 + jnp.sqrt(jnp.maximum(c4, 0.0))) + jnp.abs(rho) + jnp.abs(sigma)

    a = c1 - L * tail_factor
    b = c1 + L * tail_factor

    # Truncation bounds are kept wide (no aggressive clipping) to match the
    # NumPy COS behavior; clipping here would alter the expansion.
    return a, b


def chi_k_jax(k, a, b, c, d):
    # k is an array of indices
    bma = b - a
    omega_k = k * jnp.pi / bma
    scaling = 1.0 / (1.0 + omega_k**2)

    cos_d = jnp.cos(omega_k*(d-a))
    sin_d = jnp.sin(omega_k*(d-a))
    cos_c = jnp.cos(omega_k*(c-a))
    sin_c = jnp.sin(omega_k*(c-a))

    # Clip exponent arguments to a safe range to avoid overflow.
    exp_d = jnp.exp(jnp.clip(d, -700.0, 700.0))
    exp_c = jnp.exp(jnp.clip(c, -700.0, 700.0))

    term_d = exp_d * (cos_d + omega_k * sin_d)
    term_c = exp_c * (cos_c + omega_k * sin_c)
    return scaling * (term_d - term_c)


def psi_k_jax(k, a, b, c, d):
    bma = b - a
    arg_d = k * (d - a) / bma
    arg_c = k * (c - a) / bma
    term_d = (d - a) * jnp.sinc(arg_d)
    term_c = (c - a) * jnp.sinc(arg_c)
    return term_d - term_c


def V_k_jax(K, k, a, b, flag):
    M = 2.0 / (b - a)
    term_put = -chi_k_jax(k, a, b, a, 0.0) + psi_k_jax(k, a, b, a, 0.0)
    return M * K * term_put


def charact_func_jax(omega, x, r, q, rho, sigma, kappa, v0, theta, T):
    W = kappa - 1j * rho * sigma * omega
    # Numerically stable Albrecher form; d is complex and jnp.sqrt handles
    # the branch cut.
    d = jnp.sqrt(W ** 2 + (omega ** 2 + 1j * omega) * (sigma ** 2))
    c = (W - d) / (W + d)

    comp = 1j * omega * x
    drift_term = 1j * omega * (r - q) * T

    exp_DT = jnp.exp(-d * T)

    div_term = 1.0 - c * exp_DT

    log_part = jnp.log(div_term/(1-c))

    D = ((W - d) / (sigma ** 2)) * (1.0 - exp_DT) / div_term

    part_theta = (kappa * theta / (sigma ** 2)) * ((W - d) * T - 2.0 * log_part)
    C = part_theta + drift_term
    return jnp.exp(C + D*v0 + comp)


def price_single_jax(params, x, T, r=0.0, q=0.0, N=256, L=12):
    """params: array-like [theta, v0, sigma, rho, kappa]
       x: log-moneyness ln(S/K) (scalar)
       T, r: scalars
    """
    theta, v0, sigma, rho, kappa = params

    a, b = cumulant_truncation_jax(L, T, r, q, theta, v0, sigma, rho, kappa, x)
    bma = b - a
    k = jnp.arange(N, dtype=jnp.float64)
    omega = k * jnp.pi / bma

    # Characteristic function
    phi = charact_func_jax(omega, x, r, q, rho, sigma, kappa, v0, theta, T)

    # Cosine-series payoff coefficients. K = exp(-x) corresponds to S = 1.0.
    K = jnp.exp(-x)
    Vk = V_k_jax(K, k, a, b, 0.0)

    integrand = phi * Vk * jnp.exp(-1j*omega*a)

    # Compute the real, discounted price (first term weighted by 1/2).
    first_term = 0.5 * integrand[0]
    rest = jnp.sum(integrand[1:])
    v = jnp.exp(-r * T) * jnp.real(first_term + rest)

    return jnp.real(v)


# JIT and gradient functions.
# N and L must be static (known at trace time), so their argument positions
# are marked static_argnums. Signature: (params, x, T, r, q, N, L).
price_jit = jax.jit(price_single_jax, static_argnums=(5, 6))
grad_price = jax.jit(jax.grad(price_single_jax), static_argnums=(5, 6))
