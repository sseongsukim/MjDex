"""Noise-schedule helpers for DDPM/DDIM diffusion policies.

Every schedule is returned as a dict of plain Python float tuples rather than
arrays, so that the coefficients can be carried inside an agent's `config`
(a `nonpytree_field`, which has to stay hashable for `jax.jit`).
"""

import numpy as np


def cosine_beta_schedule(denoising_steps, s=0.008):
    """Cosine beta schedule from https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = denoising_steps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0, 0.999)


def linear_beta_schedule(denoising_steps, beta_start=1e-4, beta_end=0.02):
    """Linear beta schedule from the original DDPM paper."""
    return np.linspace(beta_start, beta_end, denoising_steps)


def ddpm_schedule(denoising_steps, beta_schedule="cosine", cosine_s=0.008):
    """Precompute the DDPM coefficients as plain tuples."""
    if beta_schedule == "cosine":
        betas = cosine_beta_schedule(denoising_steps, s=cosine_s)
    elif beta_schedule == "linear":
        betas = linear_beta_schedule(denoising_steps)
    else:
        raise ValueError(f"Unknown beta schedule '{beta_schedule}'.")

    # αₜ = 1 - βₜ, α̅ₜ = ∏ᵗₛ₌₁ αₛ, α̅ₜ₋₁
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.concatenate([np.ones(1), alphas_cumprod[:-1]])

    # β̃ₜ = σₜ² = βₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
    ddpm_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    schedule = dict(
        betas=betas,
        alphas_cumprod=alphas_cumprod,
        # q(xₜ|x₀): xₜ = √ α̅ₜ x₀ + √ (1-α̅ₜ) ε
        sqrt_alphas_cumprod=np.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=np.sqrt(1.0 - alphas_cumprod),
        # x₀ = √ 1/α̅ₜ xₜ - √ (1/α̅ₜ - 1) ε
        sqrt_recip_alphas_cumprod=np.sqrt(1.0 / alphas_cumprod),
        sqrt_recipm1_alphas_cumprod=np.sqrt(1.0 / alphas_cumprod - 1.0),
        ddpm_logvar_clipped=np.log(np.clip(ddpm_var, 1e-20, None)),
        # μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ) x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ) xₜ
        ddpm_mu_coef1=betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        ddpm_mu_coef2=(1.0 - alphas_cumprod_prev)
        * np.sqrt(alphas)
        / (1.0 - alphas_cumprod),
    )
    return {k: tuple(float(x) for x in v) for k, v in schedule.items()}


def ddim_schedule(
    alphas_cumprod,
    denoising_steps,
    ddim_steps,
    ddim_discretize="uniform",
    ddim_eta=0.0,
):
    """Precompute the DDIM coefficients as plain tuples.

    In the DDIM paper alpha refers to DDPM's alphas_cumprod. All arrays are
    flipped so that index 0 is the first (noisiest) sampling step, matching the
    order the reverse loop walks them in.
    """
    alphas_cumprod = np.asarray(alphas_cumprod, dtype=np.float64)

    if ddim_discretize == "uniform":  # the HF "leading" style
        step_ratio = denoising_steps // ddim_steps
        ddim_t = np.arange(0, ddim_steps) * step_ratio
    else:
        raise ValueError(f"Unknown DDIM discretization '{ddim_discretize}'.")

    ddim_alphas = alphas_cumprod[ddim_t]
    ddim_alphas_prev = np.concatenate([np.ones(1), alphas_cumprod[ddim_t[:-1]]])
    ddim_sqrt_one_minus_alphas = np.sqrt(1.0 - ddim_alphas)
    ddim_sigmas = ddim_eta * np.sqrt(
        (1.0 - ddim_alphas_prev)
        / (1.0 - ddim_alphas)
        * (1.0 - ddim_alphas / ddim_alphas_prev)
    )

    schedule = dict(
        ddim_t=ddim_t,
        ddim_alphas=ddim_alphas,
        ddim_alphas_prev=ddim_alphas_prev,
        ddim_sqrt_one_minus_alphas=ddim_sqrt_one_minus_alphas,
        ddim_sigmas=ddim_sigmas,
    )
    return {k: tuple(float(x) for x in np.flip(v)) for k, v in schedule.items()}
