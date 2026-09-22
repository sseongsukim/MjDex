from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={"params": 0, "intermediates": 0},
        split_rngs={"params": True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding module."""

    dim: int

    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class FourierFeatures(nn.Module):
    # used for timestep embedding
    output_size: int = 64
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param(
            "log_value", init_fn=lambda key: jnp.full((), jnp.log(self.init_value))
        )
        return jnp.exp(log_value)


class TanhNormalMLP(nn.Module):
    """Tanh-squashed Gaussian policy head backed by an MLP.

    QAM uses this as an optional edit policy over actions produced by the flow
    policy.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    def setup(self):
        self.net = MLP(
            self.hidden_dims, activate_final=True, layer_norm=self.layer_norm
        )
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init())
        self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init())

    def __call__(self, inputs):
        x = self.net(inputs)
        means = self.mean_net(x)
        log_stds = self.log_std_net(x)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        dist = distrax.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds),
        )
        return TransformedWithMode(dist, distrax.Block(distrax.Tanh(), ndims=1))


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(
            self.hidden_dims, activate_final=True, layer_norm=self.layer_norm
        )
        self.mean_net = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )
        else:
            if not self.const_std:
                self.log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class(
            (*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm
        )

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64

    def setup(self) -> None:
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v


class LatentActorVectorField(nn.Module):
    """Latent actor used by LPS to select a mean-flow latent/action candidate."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    latent_dist: str = "sphere"

    def setup(self):
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    @nn.compact
    def __call__(self, observations, actions=None, is_encoded=False, rng=None):
        inputs = []
        if self.encoder is not None:
            if isinstance(observations, (dict, flax.core.FrozenDict)):
                image = (
                    observations["image"]
                    if is_encoded
                    else self.encoder(observations["image"])
                )
                inputs.extend([image, observations["state"]])
            else:
                inputs.append(
                    observations if is_encoded else self.encoder(observations)
                )
        else:
            inputs.append(observations)

        if actions is not None:
            inputs.append(actions)

        v = self.mlp(jnp.concatenate(inputs, axis=-1))
        if self.latent_dist == "truncated_normal":
            return 2.0 * nn.tanh(v)
        if self.latent_dist == "uniform":
            return nn.tanh(v)
        if self.latent_dist == "sphere":
            norm = jnp.sqrt(jnp.sum(jnp.square(v), axis=-1, keepdims=True) + 1e-6)
            return v / norm * jnp.sqrt(self.action_dim)
        if self.latent_dist == "sphere_plus":
            norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
            return v / (norm + 1e-6) * jnp.sqrt(self.action_dim)
        return v


class ActorMeanFlowField(nn.Module):
    """Mean-flow actor field used by LPS."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 16
    mlp_class: Any = MLP

    def setup(self):
        self.mlp = self.mlp_class(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    def embed_time(self, time):
        return self.ff(time) if self.use_fourier_features else time

    @nn.compact
    def __call__(self, observations, actions, t, r=None, is_encoded=False):
        inputs = []
        if self.encoder is not None:
            if isinstance(observations, (dict, flax.core.FrozenDict)):
                image = (
                    observations["image"]
                    if is_encoded
                    else self.encoder(observations["image"])
                )
                inputs.extend([image, observations["state"]])
            else:
                inputs.append(
                    observations if is_encoded else self.encoder(observations)
                )
        else:
            inputs.append(observations)

        inputs.append(actions)
        if t is not None:
            inputs.append(self.embed_time(t))
        if r is not None:
            inputs.append(self.embed_time(r))

        return self.mlp(jnp.concatenate(inputs, axis=-1))


class ResMLP(nn.Module):
    """Residual MLP.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: If True, it works as an intermediate layer; if False, it works as a standalone neural network.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        assert self.layer_norm

        x = nn.Dense(self.hidden_dims[0], kernel_init=self.kernel_init)(x)
        x = nn.LayerNorm()(x)
        x = self.activations(x)
        num_res_blocks = (
            len(self.hidden_dims) if self.activate_final else len(self.hidden_dims) - 1
        )

        for i in range(num_res_blocks):
            size = self.hidden_dims[i]
            residual = x
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            x = nn.LayerNorm()(x)
            x = x + residual
        x = nn.LayerNorm()(x)

        if not self.activate_final:
            x = nn.Dense(self.hidden_dims[-1], kernel_init=self.kernel_init)(x)

        return x


class ResMLPDrift(nn.Module):
    """Residual MLP drift policy network.

    The module can be used in two modes:
    - Encoding mode: encode observations only (for encoder caching).
    - Diffusion mode: predict action velocities given (encoded) observations,
      noisy actions, and diffusion timesteps.

    Attributes:
        hidden_dims: Hidden layer dimensions of the residual MLP.
        horizon_steps: Number of action steps predicted in parallel.
        action_dim: Dimensionality of a single action.
        layer_norm: Whether to apply layer normalization in the MLP.
        activations: Activation function used in the network.
        kernel_init: Weight initialization for linear layers.
        encoder: Optional encoder module to encode observations before diffusion.
    """

    hidden_dims: Sequence[int]
    horizon_steps: int
    action_dim: int
    layer_norm: bool
    activations: Any = nn.gelu
    kernel_init: Any = default_init()
    encoder: nn.Module = None

    def setup(self):
        hidden_dims = self.hidden_dims + (self.action_dim * self.horizon_steps,)
        self.res_mlp = ResMLP(
            hidden_dims=hidden_dims,
            activations=self.activations,
            layer_norm=self.layer_norm,
            activate_final=False,
        )

    def __call__(self, ob, actions=None, is_encoded=None):
        if not is_encoded and self.encoder is not None:
            ob = self.encoder(ob)

        batch_size, horizon_steps, action_dim = actions.shape
        actions = actions.reshape(batch_size, -1)

        x = jnp.concatenate([actions, ob], axis=-1)
        x = self.res_mlp(x)
        return x.reshape(batch_size, horizon_steps, action_dim)


class ResMLPDiffusion(nn.Module):
    """Residual MLP diffusion policy network.

    The module can be used in two modes:
    - Encoding mode: encode observations only (for encoder caching).
    - Diffusion mode: predict action velocities given (encoded) observations,
      noisy actions, and diffusion timesteps.

    Attributes:
        hidden_dims: Hidden layer dimensions of the residual MLP.
        time_step_embed_dim: Dimensionality of the timestep embedding.
        horizon_steps: Number of action steps predicted in parallel.
        action_dim: Dimensionality of a single action.
        layer_norm: Whether to apply layer normalization in the MLP.
        activations: Activation function used in the network.
        kernel_init: Weight initialization for linear layers.
        encoder: Optional encoder module to encode observations before diffusion.
    """

    hidden_dims: Sequence[int]
    time_step_embed_dim: int
    horizon_steps: int
    action_dim: int
    layer_norm: bool
    activations: Any = nn.gelu
    kernel_init: Any = default_init()
    encoder: nn.Module = None

    def setup(self):
        self.time_mlp = nn.Sequential(
            [
                SinusoidalPosEmb(self.time_step_embed_dim),
                nn.Dense(self.time_step_embed_dim * 4, kernel_init=self.kernel_init),
                self.activations,
                nn.Dense(self.time_step_embed_dim, kernel_init=self.kernel_init),
            ]
        )

        hidden_dims = self.hidden_dims + (self.action_dim * self.horizon_steps,)
        self.res_mlp = ResMLP(
            hidden_dims=hidden_dims,
            activations=self.activations,
            layer_norm=self.layer_norm,
            activate_final=False,
        )

    def __call__(self, ob, actions=None, times=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            ob = self.encoder(ob)

        batch_size, horizon_steps, action_dim = actions.shape
        actions = actions.reshape(batch_size, -1)

        time_embed = self.time_mlp(times)
        x = jnp.concatenate([actions, time_embed, ob], axis=-1)
        x = self.res_mlp(x)
        return x.reshape(batch_size, horizon_steps, action_dim)


class ResMLPShortcutDiffusion(nn.Module):
    """Residual MLP shortcut flow policy network.

    This extends `ResMLPDiffusion` with a second conditioning signal for the
    requested shortcut step size. The step size is represented as `dt_base`,
    where a step of length `1 / 2**dt_base` corresponds to the convention used
    in shortcut models.
    """

    hidden_dims: Sequence[int]
    time_step_embed_dim: int
    horizon_steps: int
    action_dim: int
    layer_norm: bool
    activations: Any = nn.gelu
    kernel_init: Any = default_init()
    encoder: nn.Module = None

    def setup(self):
        def make_time_mlp():
            return nn.Sequential(
                [
                    SinusoidalPosEmb(self.time_step_embed_dim),
                    nn.Dense(
                        self.time_step_embed_dim * 4, kernel_init=self.kernel_init
                    ),
                    self.activations,
                    nn.Dense(self.time_step_embed_dim, kernel_init=self.kernel_init),
                ]
            )

        self.time_mlp = make_time_mlp()
        self.dt_mlp = make_time_mlp()

        hidden_dims = self.hidden_dims + (self.action_dim * self.horizon_steps,)
        self.res_mlp = ResMLP(
            hidden_dims=hidden_dims,
            activations=self.activations,
            layer_norm=self.layer_norm,
            activate_final=False,
        )

    def __call__(self, ob, actions=None, times=None, dt_base=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            ob = self.encoder(ob)

        batch_size, horizon_steps, action_dim = actions.shape
        actions = actions.reshape(batch_size, -1)

        time_embed = self.time_mlp(times)
        dt_embed = self.dt_mlp(dt_base)
        x = jnp.concatenate([actions, time_embed, dt_embed, ob], axis=-1)
        x = self.res_mlp(x)
        return x.reshape(batch_size, horizon_steps, action_dim)


class Upsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(
            features=self.dim,
            kernel_size=(4,),
            strides=(2,),
            padding="SAME",
        )(x.transpose(0, 2, 1))
        return x.transpose(0, 2, 1)


class Downsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.dim,
            kernel_size=(3,),
            strides=(2,),
            padding=((1, 1),),
        )(x.transpose(0, 2, 1))
        return x.transpose(0, 2, 1)


class TransposeModule(nn.Module):

    @nn.compact
    def __call__(self, x):
        return x.transpose(0, 2, 1)


class Conv1dBlock(nn.Module):
    input_channels: int
    output_channels: int
    kernel_size: int
    n_groups: int
    activation: Any = nn.gelu
    eps: float = 1e-5

    def setup(self):
        self.conv = nn.Conv(
            features=self.output_channels,
            kernel_size=(self.kernel_size,),
            padding="SAME",
        )
        self.groupnorm = nn.GroupNorm(
            num_groups=self.n_groups,
            epsilon=self.eps,
        )

    def __call__(self, x):
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        if self.n_groups is not None:
            x = self.groupnorm(x)
        x = self.activation(x)
        return x.transpose(0, 2, 1)


class ResidualBlock1d(nn.Module):
    in_channels: int
    out_channels: int
    cond_dim: int
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    eps: float = 1e-5
    activation: Any = nn.gelu
    kernel_init: Any = default_init()

    def setup(self):
        self.block1 = Conv1dBlock(
            input_channels=self.in_channels,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            eps=self.eps,
        )
        self.block2 = Conv1dBlock(
            input_channels=self.in_channels,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            eps=self.eps,
        )

        cond_out_features = (
            self.out_channels * 2 if self.cond_predict_scale else self.out_channels
        )
        self.cond_mlp = nn.Sequential(
            [
                self.activation,
                nn.Dense(features=cond_out_features, kernel_init=self.kernel_init),
            ]
        )

        if self.in_channels != self.out_channels:
            self.residual_conv = nn.Conv(
                features=self.out_channels,
                kernel_size=(1,),
                padding="SAME",
            )
        else:
            self.residual_conv = None

    def __call__(self, x, cond):
        out = self.block1(x)

        cond = self.cond_mlp(cond)
        cond = jnp.expand_dims(cond, axis=-1)

        if self.cond_predict_scale:
            cond = cond.reshape(cond.shape[0], 2, self.out_channels, 1)
            scale = cond[:, 0, ...]
            bias = cond[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + cond

        out = self.block2(out)
        if self.residual_conv is not None:
            residual = self.residual_conv(x.transpose(0, 2, 1))
            residual = residual.transpose(0, 2, 1)
        else:
            residual = x
        return out + residual


class UNet(nn.Module):
    action_dim: int
    cond_dim: int
    time_step_embed_dim: int
    dim: int
    dim_mults: tuple
    kernel_size: int = 5
    n_groups: int = 8
    activation: Any = nn.gelu
    cond_predict_scale: bool = True
    eps: float = 1e-5
    kernel_init: Any = default_init()

    def setup(self):

        dims = [self.action_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.time_mlp = nn.Sequential(
            [
                SinusoidalPosEmb(self.time_step_embed_dim),
                nn.Dense(
                    features=self.time_step_embed_dim * 4, kernel_init=self.kernel_init
                ),
                self.activation,
                nn.Dense(
                    features=self.time_step_embed_dim, kernel_init=self.kernel_init
                ),
            ]
        )

        cond_block_dim = self.time_step_embed_dim + self.cond_dim

        mid_dims = dims[-1]
        self.mid_modules = [
            ResidualBlock1d(
                in_channels=mid_dims,
                out_channels=mid_dims,
                cond_dim=cond_block_dim,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_predict_scale=self.cond_predict_scale,
                eps=self.eps,
            ),
            ResidualBlock1d(
                in_channels=mid_dims,
                out_channels=mid_dims,
                cond_dim=cond_block_dim,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_predict_scale=self.cond_predict_scale,
                eps=self.eps,
            ),
        ]
        down_modules = []
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (len(in_out) - 1)
            block = [
                ResidualBlock1d(
                    in_channels=dim_in,
                    out_channels=dim_out,
                    cond_dim=cond_block_dim,
                    kernel_size=self.kernel_size,
                    n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale,
                    eps=self.eps,
                ),
                ResidualBlock1d(
                    in_channels=dim_in,
                    out_channels=dim_out,
                    cond_dim=cond_block_dim,
                    kernel_size=self.kernel_size,
                    n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale,
                    eps=self.eps,
                ),
                Downsample1d(dim=dim_out) if not is_last else Identity(),
            ]
            down_modules.append(block)
        self.down_modules = down_modules

        up_modules = []
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= len(in_out) - 1
            block = [
                ResidualBlock1d(
                    in_channels=dim_out * 2,
                    out_channels=dim_in,
                    cond_dim=cond_block_dim,
                    kernel_size=self.kernel_size,
                    n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale,
                    eps=self.eps,
                ),
                ResidualBlock1d(
                    in_channels=dim_in,
                    out_channels=dim_in,
                    cond_dim=cond_block_dim,
                    kernel_size=self.kernel_size,
                    n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale,
                    eps=self.eps,
                ),
                Upsample1d(dim=dim_in) if not is_last else Identity(),
            ]
            up_modules.append(block)
        self.up_modules = up_modules

        self.final_conv = nn.Sequential(
            [
                Conv1dBlock(
                    input_channels=self.dim,
                    output_channels=self.dim,
                    kernel_size=self.kernel_size,
                    n_groups=self.n_groups,
                    eps=self.eps,
                ),
                TransposeModule(),
                nn.Conv(
                    features=self.action_dim,
                    kernel_size=(1,),
                    padding="SAME",
                ),
            ]
        )

    def __call__(self, x, time, cond):
        B = x.shape[0]
        x = x.transpose(0, 2, 1)

        time = jnp.broadcast_to(time, (B,))
        time = self.time_mlp(time)

        global_features = jnp.concatenate([time, cond], axis=-1)
        h = []
        h_local = list()
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_features)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_features)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_features)
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = jnp.concatenate((x, h.pop()), axis=1)
            x = resnet(x, global_features)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_features)
            x = upsample(x)
        x = self.final_conv(x)
        return x


class UNetActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    action_dim: int
    cond_dim: int
    time_step_embed_dim: int
    dim: int
    dim_mults: tuple
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    eps: float = 1e-5
    encoder: nn.Module = None

    def setup(self) -> None:
        self.unet = UNet(
            action_dim=self.action_dim,
            cond_dim=self.cond_dim,
            time_step_embed_dim=self.time_step_embed_dim,
            dim=self.dim,
            dim_mults=self.dim_mults,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            cond_predict_scale=self.cond_predict_scale,
            eps=self.eps,
        )

    @nn.compact
    def __call__(
        self,
        observations,
        actions=None,
        times=None,
        is_encoded=False,
    ):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        v = self.unet(x=actions, time=times, cond=observations)

        return v


class Dynamics(nn.Module):
    """Dynamics model.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        output_dim: Output dimension (set to None for scalar output).
        mlp_class: MLP class.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
    """

    hidden_dims: Sequence[int]
    output_dim: int = None
    mlp_class: Any = MLP
    layer_norm: bool = True
    num_ensembles: int = 1
    delta_pred: bool = True
    stochastic: bool = False

    def setup(self):
        mlp_class = self.mlp_class
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        output_dim = self.output_dim if self.output_dim is not None else 1
        if self.stochastic:
            dynamics_net = mlp_class(
                (*self.hidden_dims, 2 * output_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
            )
        else:
            dynamics_net = mlp_class(
                (*self.hidden_dims, output_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
            )

        self.dynamics_net = dynamics_net

    def __call__(self, observations, actions):
        """Return the predicted next states.

        Args:
            observations: Observations.
            actions: Actions.
        """
        inputs = []
        inputs.append(observations)
        inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        pred = self.dynamics_net(inputs)
        if self.stochastic:
            means, log_stds = pred[..., : self.output_dim], pred[..., self.output_dim :]
            min_logstd, max_logstd = -5.0, 1.0
            log_stds = jax.nn.sigmoid(log_stds) * (max_logstd - min_logstd) + min_logstd
            if self.delta_pred:
                means = observations + means
            distribution = distrax.MultivariateNormalDiag(
                loc=means, scale_diag=jnp.exp(log_stds)
            )
            return distribution

        if self.delta_pred:
            pred = observations + pred

        return pred
