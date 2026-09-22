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


class FBCAgent(flax.struct.PyTreeNode):
    """Flow-matching Behavior Cloning (FBC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    lr_schedule: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params, rng):
        """Compute the BC actor loss."""
        batch_size, horizon_steps, action_dim = batch["actions"].shape

        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (batch_size, horizon_steps, action_dim))
        x_1 = batch["actions"]
        t = jax.random.uniform(t_rng, shape=(batch_size,))

        vel = x_1 - x_0
        x_t = x_0 + t[:, None, None] * vel

        pred = self.network.select("actor")(
            batch["observations"], x_t, t, params=grad_params
        )
        bc_loss = jnp.mean((pred - vel) ** 2)

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

        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                self.config["horizon_steps"],
                self.config["action_dim"],
            ),
        )

        actions = noises

        def flow_iter(i, noises):
            t = jnp.full((*observations.shape[:-1],), i / self.config["flow_steps"])

            vels = self.network.select("actor")(observations, noises, t)
            return noises + vels / self.config["flow_steps"]

        noises = jax.lax.fori_loop(0, self.config["flow_steps"], flow_iter, noises)
        actions = jnp.clip(noises, -1.0, 1.0)

        return actions

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

        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            lr_schedule=lr_scheduler,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="fbc",  # Agent name.
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
        )
    )
    return config
