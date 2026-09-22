# MjDex

MjDex is a MuJoCo-based repository for building dexterous manipulation tasks
and evaluating a variety of imitation learning and reinforcement learning
fine-tuning algorithms.

## Installation

Create the Conda environment named `mjdex`, install `uv`, and then install the
project in editable mode:

```bash
conda env create -f environment.yml
conda activate mjdex
python -m pip install uv
uv pip install -e .
```

Alternatively, use Python's built-in virtual environment support:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## MuJoCo Warp parallel environments

Install the optional backend in the same environment. The pinned MuJoCo/MJX/Warp
versions are intentional: the batched contact layout is version-specific.

```bash
conda activate mjdex
uv pip install -e '.[warp,test]'
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

All 15 existing registered environments are available through `make_warp_env`:
single/dual Relocate (cracker, mustard, meat, bleach), the four DishRack variants,
single MugRack (Cartesian and joint position), and dual Barcode.
The CPU `gym.make(...)` API and scene builders remain available unchanged.

```python
from mjdex.mujoco_warp import make_warp_env
import jax.numpy as jnp

env = make_warp_env("single-relocate-mustard-v0", num_envs=64, seed=0)
obs, info = env.reset(seed=0)

# Same absolute xyz + wxyz quaternion + hand joint positions as the CPU env.
# Every observation leaf and action has a leading num_envs dimension.
actions = jnp.concatenate([obs["ee_pose"], obs["hand_joint_position"]], axis=-1)
obs, reward, terminated, truncated, info = env.step(actions)

# Save the final transition before resetting completed worlds. No autoreset.
obs, reset_info = env.reset(mask=terminated | truncated)
env.close()
```

Observations retain the original Dict keys, ordering through Gymnasium's Dict
space, units and shapes. Rewards, flags and info values are batched JAX arrays
on the selected CUDA device. `single_action_space` and `single_observation_space`
describe one world; `action_space` and `observation_space` describe the full batch.
Set `return_numpy=True` for debugging (this synchronizes and transfers to CPU).
To flatten observations in the same order as `utils.env_utils.FlattenObsWrapper`:

```python
flat_obs = jnp.concatenate(
    [obs[k].reshape(env.num_envs, -1) for k in env.single_observation_space.spaces],
    axis=-1,
)
```

The original MJCF/assets, actuator gains, physics/control timestep, gravity
compensation, action layout, IK iteration/tolerance settings, task predicates
and effective CPU episode limits are reused. Physics runs through **MJX-Warp**;
isolated arm IK kinematics and batch task calculations run in JAX on the GPU.
Warp uses float32, so contact trajectories and IK near singularities can differ
numerically from CPU MuJoCo's float64 results.


### Run and verify

```bash
# Open the MuJoCo viewer and inspect the whole GPU batch on one grid.
python collect/view_warp_task.py \
  --task dual-hand-dishrack-v0 --num-envs 16

# Mirror a single world instead, switching between them with [ and ].
python collect/view_warp_task.py \
  --task dual-hand-dishrack-v0 --num-envs 16 --layout single --world-index 0
```


## SpaceMouse setup

Set up the [SpaceMouse](https://3dconnexion.com/us/product/spacemouse-wireless/)
for demonstration collection:

```bash
# Python packages needed for SpaceMouse support
uv pip install numpy termcolor atomics scipy
uv pip install git+https://github.com/cheng-chi/spnav

# System packages and daemon
sudo apt install libspnav-dev spacenavd
sudo systemctl start spacenavd
```