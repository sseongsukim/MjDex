"""Batched wxyz quaternion operations, following MuJoCo's conventions."""

import jax.numpy as jnp


def normalize_quat(q):
    norm = jnp.linalg.norm(q, axis=-1, keepdims=True)
    return jnp.where(
        norm < 1e-8, jnp.array([1.0, 0.0, 0.0, 0.0]), q / jnp.maximum(norm, 1e-8)
    )


def quat_mul(a, b):
    aw, ax, ay, az = jnp.moveaxis(a, -1, 0)
    bw, bx, by, bz = jnp.moveaxis(b, -1, 0)
    return jnp.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quat_error(goal, current):
    q = normalize_quat(quat_mul(goal, current * jnp.array([1.0, -1.0, -1.0, -1.0])))
    q = jnp.where(q[..., :1] < 0, -q, q)
    norm = jnp.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2 * jnp.arctan2(norm, q[..., :1])
    return q[..., 1:] * angle / jnp.maximum(norm, 1e-8)


def quat_distance(a, b):
    dot = jnp.abs(jnp.sum(normalize_quat(a) * normalize_quat(b), axis=-1))
    return 2 * jnp.arccos(jnp.clip(dot, -1.0, 1.0))


def mat_to_quat(m):
    # Select the same branch as mju_mat2Quat, including its quaternion sign.
    a, b, c = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    diag = jnp.stack(
        (1 + a + b + c, 1 + a - b - c, 1 - a + b - c, 1 - a - b + c), axis=-1
    )
    yz = m[..., 2, 1] - m[..., 1, 2]
    xz = m[..., 0, 2] - m[..., 2, 0]
    xy = m[..., 1, 0] - m[..., 0, 1]
    sxy = m[..., 0, 1] + m[..., 1, 0]
    sxz = m[..., 0, 2] + m[..., 2, 0]
    syz = m[..., 1, 2] + m[..., 2, 1]
    rows = jnp.stack(
        (
            jnp.stack((diag[..., 0], yz, xz, xy), -1),
            jnp.stack((yz, diag[..., 1], sxy, sxz), -1),
            jnp.stack((xz, sxy, diag[..., 2], syz), -1),
            jnp.stack((xy, sxz, syz, diag[..., 3]), -1),
        ),
        -2,
    )
    index = jnp.where(a + b + c > 0, 0, 1 + jnp.argmax(diag[..., 1:], axis=-1))
    return normalize_quat(
        jnp.take_along_axis(rows, index[..., None, None], axis=-2)[..., 0, :]
    )
