import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax

from utils.networks import ResMLPDiffusion, UNetActorVectorField
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.encoders import encoder_modules
from utils.diffusion_utils import ddim_schedule, ddpm_schedule


class DBCAgent(flax.struct.PyTreeNode):
    """Diffusion Policy Behavior Cloning (FBC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    lr_schedule: Any = nonpytree_field()

    def _coef(self, name, t, ndim):
        """Gather a schedule coefficient at timestep `t`, broadcast to `ndim`."""
        coefs = jnp.asarray(self.config[name])
        return coefs[t].reshape(*t.shape, *((1,) * (ndim - t.ndim)))

    def _predict_x_start(self, x, t, eps, alpha=None, sqrt_one_minus_alpha=None):
        """Recover x₀ from the network output at timestep `t`."""
        if not self.config["predict_epsilon"]:
            return eps  # the network directly predicts x₀

        if self.config["use_ddim"]:
            # x₀ = (xₜ - √ (1-αₜ) ε) / √ αₜ
            return (x - sqrt_one_minus_alpha * eps) / jnp.sqrt(alpha)

        # x₀ = √ 1/α̅ₜ xₜ - √ (1/α̅ₜ - 1) ε
        return (
            self._coef("sqrt_recip_alphas_cumprod", t, x.ndim) * x
            - self._coef("sqrt_recipm1_alphas_cumprod", t, x.ndim) * eps
        )

    def actor_loss(self, batch, grad_params, rng):
        """Compute the BC actor loss."""
        batch_size, horizon_steps, action_dim = batch["actions"].shape

        rng, noise_rng, t_rng = jax.random.split(rng, 3)
        x_start = batch["actions"]
        noise = jax.random.normal(noise_rng, (batch_size, horizon_steps, action_dim))
        t = jax.random.randint(
            t_rng, (batch_size,), 0, self.config["denoising_steps"]
        )

        # Forward process: xₜ = √ α̅ₜ x₀ + √ (1-α̅ₜ) ε
        x_noisy = (
            self._coef("sqrt_alphas_cumprod", t, x_start.ndim) * x_start
            + self._coef("sqrt_one_minus_alphas_cumprod", t, x_start.ndim) * noise
        )

        pred = self.network.select("actor")(
            batch["observations"],
            x_noisy,
            t.astype(jnp.float32),
            params=grad_params,
        )
        target = noise if self.config["predict_epsilon"] else x_start
        bc_loss = jnp.mean((pred - target) ** 2)

        return bc_loss, {"bc_loss": bc_loss}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng, 2)

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        info["lr"] = self.lr_schedule(self.network.step)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, rng=None, temperature=None):
        """
        Sample multi-step actions using diffusion / flow matching.

        Args:
            observations: Dict[str, jnp.ndarray], batched observations
            seed: PRNGKey
            temperature: Unused (kept for API compatibility)

        Returns:
            actions: (batch_size, horizon_steps, action_dim)
        """
        seed = self.rng if rng is None else rng

        if len(observations.shape) == 1:
            observations = jnp.expand_dims(observations, axis=0)

        batch_dims = observations.shape[: -len(self.config["ob_dims"])]
        noise_seed, denoise_seed = jax.random.split(seed)
        x = jax.random.normal(
            noise_seed,
            (*batch_dims, self.config["horizon_steps"], self.config["action_dim"]),
        )

        denoised_clip_value = self.config["denoised_clip_value"]
        randn_clip_value = self.config["randn_clip_value"]
        eps_clip_value = self.config["eps_clip_value"]

        def ddpm_iter(i, carry):
            x, iter_rng = carry
            iter_rng, step_rng = jax.random.split(iter_rng)

            # Walk t from denoising_steps - 1 down to 0.
            t_index = self.config["denoising_steps"] - 1 - i
            t = jnp.full(batch_dims, t_index, dtype=jnp.int32)

            eps = self.network.select("actor")(
                observations, x, t.astype(jnp.float32)
            )
            x_recon = self._predict_x_start(x, t, eps)
            if denoised_clip_value is not None:
                x_recon = jnp.clip(x_recon, -denoised_clip_value, denoised_clip_value)

            # μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ) x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ) xₜ
            mu = (
                self._coef("ddpm_mu_coef1", t, x.ndim) * x_recon
                + self._coef("ddpm_mu_coef2", t, x.ndim) * x
            )
            logvar = self._coef("ddpm_logvar_clipped", t, x.ndim)

            # The last step is deterministic; elsewhere keep the std off zero.
            std = jnp.where(t_index == 0, 0.0, jnp.clip(jnp.exp(0.5 * logvar), 1e-3))
            noise = jnp.clip(
                jax.random.normal(step_rng, x.shape), -randn_clip_value, randn_clip_value
            )
            return mu + std * noise, iter_rng

        def ddim_iter(i, carry):
            x, iter_rng = carry

            index = jnp.full((*batch_dims,), i, dtype=jnp.int32)
            t = jnp.asarray(self.config["ddim_t"])[index]
            alpha = self._coef("ddim_alphas", index, x.ndim)
            alpha_prev = self._coef("ddim_alphas_prev", index, x.ndim)
            sqrt_one_minus_alpha = self._coef(
                "ddim_sqrt_one_minus_alphas", index, x.ndim
            )
            sigma = self._coef("ddim_sigmas", index, x.ndim)

            eps = self.network.select("actor")(
                observations, x, t.astype(jnp.float32)
            )
            x_recon = self._predict_x_start(
                x, t, eps, alpha=alpha, sqrt_one_minus_alpha=sqrt_one_minus_alpha
            )
            if denoised_clip_value is not None:
                x_recon = jnp.clip(x_recon, -denoised_clip_value, denoised_clip_value)
                # Re-derive epsilon from the clamped x₀.
                eps = (x - jnp.sqrt(alpha) * x_recon) / sqrt_one_minus_alpha
            if eps_clip_value is not None:
                eps = jnp.clip(eps, -eps_clip_value, eps_clip_value)

            # μ = √ αₜ₋₁ x₀ + √ (1-αₜ₋₁-σₜ²) ε, deterministic for eta = 0.
            dir_xt = jnp.sqrt(jnp.clip(1.0 - alpha_prev - sigma**2, 0.0)) * eps
            return jnp.sqrt(alpha_prev) * x_recon + dir_xt, iter_rng

        if self.config["use_ddim"]:
            num_steps, denoise_iter = self.config["ddim_steps"], ddim_iter
        else:
            num_steps, denoise_iter = self.config["denoising_steps"], ddpm_iter

        x, _ = jax.lax.fori_loop(0, num_steps, denoise_iter, (x, denoise_seed))

        final_action_clip_value = self.config["final_action_clip_value"]
        if final_action_clip_value is not None:
            x = jnp.clip(x, -final_action_clip_value, final_action_clip_value)

        return x

    @classmethod
    def create(
        cls,
        seed,
        ex_transition,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = ex_transition["observations"]
        ex_actions = ex_transition["actions"]

        ob_dims = ex_observations.shape[1:]

        batch_size, horizon_steps, action_dim = ex_actions.shape
        ex_times = np.empty(shape=(batch_size,))

        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["actor"] = encoder_module()

        if config["network_type"] == "mlp":
            actor_def = ResMLPDiffusion(
                hidden_dims=config["hidden_dims"],
                time_step_embed_dim=config["time_step_embed_dim"],
                horizon_steps=config["horizon_steps"],
                action_dim=action_dim,
                layer_norm=config["layer_norm"],
                encoder=encoders.get("actor"),
            )
        elif config["network_type"] == "unet":
            actor_def = UNetActorVectorField(
                action_dim=action_dim,
                cond_dim=ob_dims[0],
                time_step_embed_dim=config["time_step_embed_dim"],
                dim=config["dim"],
                dim_mults=config["dim_mults"],
                kernel_size=config["kernel_size"],
                n_groups=config["n_groups"],
                encoder=encoders.get("actor"),
            )

        network_info = dict(
            actor=(actor_def, (ex_observations, ex_actions, ex_times)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config["lr"],
            warmup_steps=config["train_steps"] * 0.1,
            decay_steps=config["train_steps"],
            end_value=0.0,
        )

        if config["weight_decay"] > 0:
            network_tx = optax.adamw(
                learning_rate=lr_scheduler, weight_decay=config["weight_decay"]
            )
        else:
            network_tx = optax.adam(learning_rate=lr_scheduler)

        network_variables = network_def.init(init_rng, **network_args)
        network_params = network_variables["params"]

        network = TrainState.create(network_def, network_params, tx=network_tx)

        config["action_dim"] = action_dim
        config["ob_dims"] = ob_dims

        # DDPM/DDIM coefficients live in `config` (as hashable tuples) so that the
        # agent's PyTreeNode fields stay exactly as they are.
        ddpm_params = ddpm_schedule(
            config["denoising_steps"],
            beta_schedule=config["beta_schedule"],
            cosine_s=config["cosine_s"],
        )
        for key, value in ddpm_params.items():
            config[key] = value

        if config["use_ddim"]:
            assert config[
                "predict_epsilon"
            ], "DDIM requires predicting epsilon for now."
            ddim_steps = config["ddim_steps"]
            assert (
                ddim_steps is not None and 0 < ddim_steps <= config["denoising_steps"]
            ), "ddim_steps must be in (0, denoising_steps] when use_ddim is set."
            ddim_params = ddim_schedule(
                ddpm_params["alphas_cumprod"],
                denoising_steps=config["denoising_steps"],
                ddim_steps=ddim_steps,
                ddim_discretize=config["ddim_discretize"],
                ddim_eta=config["ddim_eta"],
            )
            for key, value in ddim_params.items():
                config[key] = value

        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            lr_schedule=lr_scheduler,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="dbc",  # Agent name.
            lr=1e-4,  # Learning rate for the optimizer.
            batch_size=256,  # Batch size used during training.
            hidden_dims=(1024, 1024, 1024, 1024, 1024),
            # Hidden layer dimensions of the actor MLP.
            layer_norm=True,  # Whether to apply layer normalization in the actor network.
            dataset_class="MultistepDataset",
            # Dataset class name (e.g., for multi-step trajectories).
            network_type="mlp",  # Network type
            horizon_steps=24,  # Number of action steps predicted in parallel.
            weight_decay=1e-6,
            flow_steps=30,  # Number of diffusion / flow integration steps.
            time_step_embed_dim=32,  # Dimensionality of the diffusion timestep embedding.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            inference_steps=4,  # Number of inference steps.
            # UNet
            dim=256,
            dim_mults=(1, 2, 4),
            kernel_size=5,
            n_groups=8,
            # DDPM parameters
            denoising_steps=100,  # Number of DDPM denoising steps.
            predict_epsilon=True,  # Predict the noise instead of x_0.
            beta_schedule="cosine",  # Beta schedule ('cosine' or 'linear').
            cosine_s=0.008,  # Offset of the cosine beta schedule.
            # Various clipping
            denoised_clip_value=1.0,  # Clip the predicted x_0 at each step.
            randn_clip_value=10.0,  # Clip the noise sampled at each step.
            final_action_clip_value=1.0,  # Clip the action returned by the last step.
            eps_clip_value=ml_collections.config_dict.placeholder(
                float
            ),  # Clip the predicted epsilon (DDIM only).
            # DDIM sampling
            use_ddim=False,  # Sample with DDIM instead of DDPM.
            ddim_discretize="uniform",  # DDIM timestep discretization.
            ddim_steps=10,  # Number of DDIM sampling steps.
            ddim_eta=0.0,  # DDIM noise scale (0 is deterministic).
        )
    )
    return config
