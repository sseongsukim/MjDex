"""GPU differential IK using the CPU environment's isolated arm models."""

import jax
import jax.numpy as jnp
from mujoco import mjx

from .math import mat_to_quat, quat_error


class BatchedDiffIK:
    def __init__(self, controller, device):
        # Only kinematics/Jacobians run in JAX; scene physics runs in Warp.
        self.model = mjx.put_model(controller._model, impl="jax", device=device)
        self.template = mjx.make_data(controller._model, impl="jax", device=device)
        self.site = int(controller._site_ids[0])
        self.body = int(controller._model.site_bodyid[self.site])
        self.damping = float(controller._damping[0, 0])
        self.max_change = controller._max_angle_change

    def solve(self, qpos, position, quaternion):
        def one(q, pos, quat):
            def iteration(_, state):
                q, done = state
                d = self.template.replace(qpos=q)
                d = mjx.com_pos(self.model, mjx.kinematics(self.model, d))
                error = jnp.concatenate(
                    (
                        pos - d.site_xpos[self.site],
                        quat_error(quat, mat_to_quat(d.site_xmat[self.site])),
                    )
                )
                done = done | (
                    (jnp.linalg.norm(error[:3]) <= 1e-4)
                    & (jnp.linalg.norm(error[3:]) <= 1e-4)
                )
                jp, jr = mjx.jac(
                    self.model, d, d.site_xpos[self.site], jnp.asarray(self.body)
                )
                jac = jnp.concatenate((jp.T, jr.T), axis=0)
                update = jac.T @ jnp.linalg.solve(
                    jac @ jac.T + self.damping * jnp.eye(6), error
                )
                scale = jnp.minimum(
                    1.0, self.max_change / jnp.maximum(jnp.max(jnp.abs(update)), 1e-8)
                )
                # All joints in the isolated FR3/UR5e models are hinges.
                return jnp.where(done, q, q + scale * update), done

            return jax.lax.fori_loop(0, 20, iteration, (q, jnp.array(False)))[0]

        return jax.vmap(one)(qpos, position, quaternion)
