"""RACE utilities: TOPP-RA interpolation and Best-of-N chunk selection.

Reference: "Time Optimal Execution of Action Chunk Policies Beyond Demonstration
           Speed", Sunwoo Kim et al., ICLR 2026.

Design notes
------------
* No external toppra library is used. The algorithm is implemented directly
  with scipy.interpolate.CubicSpline (already a MjDex dependency).
* Path parameterization: cubic spline q(s) where s is the waypoint index
  (0 → N). Physical time is NOT baked into s; TOPP-RA assigns time later.
* TOPP-RA state variable: x = (ds/dt)^2  (squared path speed, units: (idx/s)^2)
  Derivative:             u = dx/ds        (path acceleration proxy)
* Joint velocity constraint:  |dq_j/ds| * sqrt(x) <= vel_j
  → x <= (vel_j / |dq_j/ds|)^2
* Joint acceleration constraint: q̈_j = dq_j/ds * (u/2) + d²q_j/ds² * x
  → a_j * u + b_j * x ∈ [-acc_j, acc_j]  where a_j = dq_j/ds / 2, b_j = d²q_j/ds²
* Backward pass: compute controllable set K[i] = (0, x_max[i]) by propagating
  from the end; x_max[i] is the largest feasible x at grid point i.
* Forward pass: greedily maximise x at each step, clamped to K.
* Integration: t(s) = ∫ ds / sqrt(x(s))  → dense trajectory at physics_dt.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Per-robot joint limits
#
# Velocity (rad/s): official manufacturer spec.
# Acceleration (rad/s^2): not always published; values here are conservative
#   practical figures consistent with the reference RACE implementation.
# ---------------------------------------------------------------------------

# Universal Robots UR5e — 6 DOF, all joints identical velocity limit
UR5E_VEL_LIMITS = np.array([3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159])
UR5E_ACC_LIMITS = np.array([15.0, 15.0, 15.0, 20.0, 20.0, 20.0])

# Franka Research 3 (FR3) — 7 DOF
# Velocity: joints 1-4 slower (2.62), wrist joints 5-7 faster (5.26 / 4.18 / 4.18)
FR3_VEL_LIMITS = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 4.18])
FR3_ACC_LIMITS = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

# Registry keyed by the robot name used in MjDex envs (mjdex/robots.py)
_ROBOT_VEL_LIMITS: dict[str, np.ndarray] = {
    "ur5e": UR5E_VEL_LIMITS,
    "fr3": FR3_VEL_LIMITS,
}
_ROBOT_ACC_LIMITS: dict[str, np.ndarray] = {
    "ur5e": UR5E_ACC_LIMITS,
    "fr3": FR3_ACC_LIMITS,
}


class TOPPRAInterpolator:
    """Minimum-time trajectory planner for joint-space waypoints.

    Given a chunk of N waypoints [q_1, ..., q_N] from a policy (each at
    policy_dt intervals in the demo), this class fits a cubic spline through
    the waypoints and solves for the fastest time-parameterization that stays
    within joint velocity and acceleration limits.

    Usage::

        interp = TOPPRAInterpolator(
            vel_limits=UR5E_VEL_LIMITS,
            acc_limits=UR5E_ACC_LIMITS,
            physics_dt=env.physics_timestep,    # e.g. 0.002
            policy_dt=env.control_timestep,     # e.g. 0.05  (20 Hz)
        )
        dense, s_query = interp.compute(chunk, q0=obs["arm_joint_position"], qd0=qd)
    """

    def __init__(
        self,
        vel_limits: np.ndarray,
        acc_limits: np.ndarray,
        physics_dt: float,
        policy_dt: float,
        n_grid: int = 100,
        vel_safety: float = 0.9,
        acc_safety: float = 0.9,
    ) -> None:
        """
        Args:
            vel_limits:  (dof,) max joint velocity  [rad/s]
            acc_limits:  (dof,) max joint acceleration [rad/s^2]
            physics_dt:  MuJoCo physics timestep [s]
            policy_dt:   policy control timestep [s]  (= 1 / policy_hz)
            n_grid:      number of TOPP-RA discretization points along path
            vel_safety:  multiplicative safety margin on vel_limits  (< 1.0)
            acc_safety:  multiplicative safety margin on acc_limits  (< 1.0)
        """
        self.vel_limits = np.asarray(vel_limits, dtype=np.float64) * vel_safety
        self.acc_limits = np.asarray(acc_limits, dtype=np.float64) * acc_safety
        self.physics_dt = float(physics_dt)
        self.policy_dt = float(policy_dt)
        self.n_grid = int(n_grid)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        waypoints: np.ndarray,
        q0: np.ndarray,
        qd0: np.ndarray,
        open_loop_steps: int | None = None,
    ) -> np.ndarray:
        """Compute a dense trajectory from a policy chunk via TOPP-RA.

        Args:
            waypoints:       (N, dof) joint positions predicted by the policy.
                             All N waypoints are used as the planning path.
            q0:              (dof,)   current joint position
            qd0:             (dof,)   current joint velocity  [rad/s]
            open_loop_steps: if given, only return the dense steps up to the
                             time at which waypoint ``open_loop_steps`` is
                             reached.  Useful when horizon_steps > inference_steps:
                             plan over the full horizon for a better speed profile,
                             but execute only the first ``open_loop_steps`` portion.

        Returns:
            (dense_traj, s_query):
                dense_traj (M, dof)  — positions sampled every policy_dt.
                    M = ceil(optimal_duration / policy_dt), or truncated when
                    open_loop_steps is set.  Falls back to nominal timing if
                    TOPP-RA is infeasible.
                s_query (M,) — path parameter at each control step.
                    s=0 is q0, s=k is waypoint k.  Used to map gripper commands
                    to the correct path position in step_chunk().
        """
        q0 = np.asarray(q0, dtype=np.float64)
        qd0 = np.asarray(qd0, dtype=np.float64)
        waypoints = np.asarray(waypoints, dtype=np.float64)

        path_pts = np.concatenate([q0[None], waypoints], axis=0)  # (N+1, dof)
        n_pts = len(path_pts)
        s_pts = np.arange(n_pts, dtype=np.float64)

        # Spline boundary: initial velocity in "index units" (dq/ds at s=0)
        # q̇ = dq/ds * ds/dt  →  dq/ds = q̇ / (ds/dt) = q̇ * policy_dt
        qd0_per_idx = qd0 * self.policy_dt
        cs = CubicSpline(
            s_pts,
            path_pts,
            bc_type=((1, qd0_per_idx), "not-a-knot"),
        )

        # Evaluate path derivatives on dense grid
        s_grid = np.linspace(0.0, float(n_pts - 1), self.n_grid)
        dq_ds = cs(s_grid, 1)  # (n_grid, dof)
        ddq_ds = cs(s_grid, 2)  # (n_grid, dof)

        x_max_vel = self._velocity_x_max(dq_ds, ddq_ds)
        K = self._backward_pass(s_grid, dq_ds, ddq_ds, x_max_vel)

        # Initial x: nominal path speed is 1 idx/policy_dt  → x = (1/policy_dt)^2
        x0_nominal = (1.0 / self.policy_dt) ** 2
        x0 = float(np.clip(x0_nominal, 0.0, K[0, 1]))

        if K[0, 1] < 1e-9:
            # TOPP-RA found no feasible plan; fall back to nominal timing
            return self._nominal_trajectory(cs, n_pts)

        x_opt = self._forward_pass(s_grid, dq_ds, ddq_ds, K, x0)
        t_grid = self._integrate_time(s_grid, x_opt)

        # Optionally truncate: plan over all N waypoints but execute only
        # up to waypoint open_loop_steps (waypoint k ↔ s = k in path coords).
        t_max = None
        if open_loop_steps is not None and open_loop_steps < len(waypoints):
            t_max = float(np.interp(float(open_loop_steps), s_grid, t_grid))

        return self._sample_trajectory(cs, s_grid, t_grid, t_max=t_max)

    # ------------------------------------------------------------------
    # TOPP-RA internals
    # ------------------------------------------------------------------

    def _velocity_x_max(
        self,
        dq_ds: np.ndarray,
        ddq_ds: np.ndarray,
    ) -> np.ndarray:
        """Upper bound on x from velocity constraints (and zero-derivative acc).

        Fully vectorised over both grid points and joints — no Python loops.
        """
        active_vel = np.abs(dq_ds) > 1e-9  # (n, dof)
        x_vel = np.where(
            active_vel,
            (self.vel_limits / np.maximum(np.abs(dq_ds), 1e-10)) ** 2,
            np.inf,
        )  # (n, dof)

        active_acc = (~active_vel) & (np.abs(ddq_ds) > 1e-9)
        x_acc = np.where(
            active_acc,
            self.acc_limits / np.maximum(np.abs(ddq_ds), 1e-10),
            np.inf,
        )  # (n, dof)

        x_max = np.minimum(x_vel, x_acc).min(axis=1)  # (n,)
        fallback = (self.vel_limits.max() * 10.0) ** 2
        x_max = np.where(np.isinf(x_max), fallback, x_max)
        return np.maximum(x_max, 0.0)

    def _u_bounds(
        self,
        dq_ds_i: np.ndarray,
        ddq_ds_i: np.ndarray,
        x: float,
    ) -> tuple[float, float]:
        """Feasible u = dx/ds interval at given x from acceleration constraints.

        q̈_j = dq_j/ds * (u/2) + d²q_j/ds² * x  must satisfy |q̈_j| ≤ a_max_j.
        """
        residual = ddq_ds_i * x
        a_lo = -self.acc_limits - residual
        a_hi = self.acc_limits - residual
        active = np.abs(dq_ds_i) > 1e-9
        if not np.any(active):
            return -np.inf, np.inf
        dq = dq_ds_i[active]
        # When dq > 0: lo = 2*a_lo/dq, hi = 2*a_hi/dq
        # When dq < 0: swap (dividing by negative flips the inequality)
        lo = 2.0 * np.where(dq > 0, a_lo[active], a_hi[active]) / dq
        hi = 2.0 * np.where(dq > 0, a_hi[active], a_lo[active]) / dq
        return float(lo.max()), float(hi.min())

    def _backward_pass(
        self,
        s_grid: np.ndarray,
        dq_ds: np.ndarray,
        ddq_ds: np.ndarray,
        x_max_vel: np.ndarray,
    ) -> np.ndarray:
        """Compute controllable set K[i] = [0, x_max[i]] via backward propagation.

        K[i] is the set of x values at s_grid[i] from which the robot can safely
        reach the end of the path while satisfying all constraints.
        """
        n = self.n_grid
        K = np.zeros((n, 2))
        K[-1, 1] = x_max_vel[-1]

        for i in range(n - 2, -1, -1):
            ds = s_grid[i + 1] - s_grid[i]
            x_next_hi = K[i + 1, 1]

            # Find the largest x_i ≤ x_max_vel[i] that can reach K[i+1].
            # x_{i+1} = x_i + u * ds  must fall in [0, x_next_hi].
            # Starting from x_candidate, check feasibility.
            x_candidate = x_max_vel[i]

            if not self._is_forward_feasible(
                x_candidate, x_next_hi, ds, dq_ds[i], ddq_ds[i]
            ):
                # Binary search for the largest feasible x_i
                lo, hi = 0.0, x_candidate
                for _ in range(24):
                    mid = (lo + hi) / 2.0
                    if self._is_forward_feasible(
                        mid, x_next_hi, ds, dq_ds[i], ddq_ds[i]
                    ):
                        lo = mid
                    else:
                        hi = mid
                x_candidate = (
                    lo
                    if self._is_forward_feasible(lo, x_next_hi, ds, dq_ds[i], ddq_ds[i])
                    else 0.0
                )

            K[i, 1] = x_candidate

        return K

    def _is_forward_feasible(
        self,
        x_i: float,
        x_next_hi: float,
        ds: float,
        dq_ds_i: np.ndarray,
        ddq_ds_i: np.ndarray,
    ) -> bool:
        """Return True if x_i can reach [0, x_next_hi] in one step."""
        if x_i < 0:
            return False
        u_lo, u_hi = self._u_bounds(dq_ds_i, ddq_ds_i, x_i)
        if u_lo > u_hi:
            return False
        x_next_lo_reach = x_i + u_lo * ds
        x_next_hi_reach = x_i + u_hi * ds
        # Intersection of achievable interval and [0, x_next_hi] must be non-empty
        return x_next_hi_reach >= 0.0 and x_next_lo_reach <= x_next_hi

    def _forward_pass(
        self,
        s_grid: np.ndarray,
        dq_ds: np.ndarray,
        ddq_ds: np.ndarray,
        K: np.ndarray,
        x0: float,
    ) -> np.ndarray:
        """Greedily maximise x at each step (fastest feasible traversal)."""
        n = self.n_grid
        x_opt = np.zeros(n)
        x_opt[0] = x0

        for i in range(n - 1):
            ds = s_grid[i + 1] - s_grid[i]
            u_lo_i, u_hi_i = self._u_bounds(dq_ds[i], ddq_ds[i], x_opt[i])
            x_next_max = x_opt[i] + u_hi_i * ds
            x_opt[i + 1] = float(np.clip(x_next_max, 0.0, K[i + 1, 1]))

        return x_opt

    def _integrate_time(
        self,
        s_grid: np.ndarray,
        x_opt: np.ndarray,
    ) -> np.ndarray:
        """Integrate dt = ds / sqrt(x) to get cumulative time t[i] at each s_grid[i]."""
        ds = np.diff(s_grid)
        x_avg = np.maximum((x_opt[:-1] + x_opt[1:]) / 2.0, 1e-8)
        dt = ds / np.sqrt(x_avg)
        return np.concatenate([[0.0], np.cumsum(dt)])

    def _sample_trajectory(
        self,
        cs: CubicSpline,
        s_grid: np.ndarray,
        t_grid: np.ndarray,
        t_max: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample trajectory at policy_dt intervals by inverting t(s) → s(t).

        Sampling at policy_dt (not physics_dt) so that each returned waypoint
        is executed via env.step(), which runs the full n_steps physics
        sub-steps and properly settles the PD controller.

        Args:
            t_max: if given, only sample up to this time (open-loop truncation).

        Returns:
            (positions, s_query): positions (M, dof) and path parameters (M,).
            s_query[i] ∈ [0, N] is the path position at control step i
            (s=0 is q0, s=k is waypoint k), enabling correct gripper remapping.
        """
        total_duration = t_grid[-1] if t_max is None else min(t_max, t_grid[-1])
        n_dense = max(1, int(np.ceil(total_duration / self.policy_dt)))
        t_query = np.linspace(0.0, total_duration, n_dense + 1)[1:]  # skip t=0
        s_query = np.interp(t_query, t_grid, s_grid)
        return cs(s_query), s_query  # (n_dense, dof), (n_dense,)

    def _nominal_trajectory(
        self, cs: CubicSpline, n_pts: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback: sample path at original policy_dt timing (no speed-up)."""
        n_dense = max(1, n_pts - 1)
        s_query = np.linspace(0.0, float(n_pts - 1), n_dense + 1)[1:]
        return cs(s_query), s_query


# ---------------------------------------------------------------------------
# Best-of-N chunk selection
# ---------------------------------------------------------------------------


def best_of_n_select(
    chunks: np.ndarray,
    q0: np.ndarray,
    qd0: np.ndarray,
    policy_dt: float,
    n_eval_pts: int = 50,
) -> tuple[np.ndarray, int]:
    """Select the smoothest chunk from N diffusion policy samples.

    Smoothness criterion: minimise mean squared second derivative of the
    cubic spline (= integral of path curvature). A smoother path has smaller
    |d²q/ds²|, which keeps the TOPP-RA acceleration terms (b_j * x) small and
    enlarges the controllable set — making a high-speed plan easier to find.

    Args:
        chunks:      (N, T, dof)  N candidate action chunks from diffusion policy
        q0:          (dof,)       current joint position
        qd0:         (dof,)       current joint velocity [rad/s]
        policy_dt:   float        policy control timestep [s]
        n_eval_pts:  int          number of evaluation points along spline

    Returns:
        best_chunk:  (T, dof)  the selected chunk
        best_idx:    int       index into chunks
    """
    q0 = np.asarray(q0, dtype=np.float64)
    qd0 = np.asarray(qd0, dtype=np.float64)
    n_samples = chunks.shape[0]
    scores = np.empty(n_samples)
    qd0_per_idx = qd0 * policy_dt

    for k in range(n_samples):
        path_pts = np.concatenate([q0[None], chunks[k]], axis=0)
        s_pts = np.arange(len(path_pts), dtype=np.float64)
        cs = CubicSpline(s_pts, path_pts, bc_type=((1, qd0_per_idx), "not-a-knot"))
        s_eval = np.linspace(0.0, s_pts[-1], n_eval_pts)
        ddq = cs(s_eval, 2)  # (n_eval_pts, dof)
        # Negative mean squared curvature; higher = smoother = preferred
        scores[k] = -float(np.mean(np.sum(ddq**2, axis=1)))

    best_idx = int(np.argmax(scores))
    return chunks[best_idx], best_idx


# ---------------------------------------------------------------------------
# Convenience: build a TOPPRAInterpolator from a MjDexEnv instance
# ---------------------------------------------------------------------------


def make_toppra_for_env(
    env,
    vel_limits: np.ndarray | None = None,
    acc_limits: np.ndarray | None = None,
    **kwargs,
) -> TOPPRAInterpolator:
    """Build a TOPPRAInterpolator using timestep settings from a MjDexEnv.

    Args:
        env:         a MjDexEnv (or subclass) instance after reset()
        vel_limits:  override joint velocity limits; defaults to UR5E_VEL_LIMITS
        acc_limits:  override joint acceleration limits; defaults to UR5E_ACC_LIMITS
        **kwargs:    forwarded to TOPPRAInterpolator (n_grid, vel_safety, …)

    Returns:
        TOPPRAInterpolator ready to use with step_chunk().
    """
    if vel_limits is None:
        robot_name = getattr(env, "robot_name", None)
        vel_limits = _ROBOT_VEL_LIMITS.get(robot_name, UR5E_VEL_LIMITS)

    if acc_limits is None:
        robot_name = getattr(env, "robot_name", None)
        acc_limits = _ROBOT_ACC_LIMITS.get(robot_name, UR5E_ACC_LIMITS)

    return TOPPRAInterpolator(
        vel_limits=vel_limits,
        acc_limits=acc_limits,
        physics_dt=env.physics_timestep,
        policy_dt=env.control_timestep,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# RACEEnvWrapper
# ---------------------------------------------------------------------------


class RACEEnvWrapper(gym.Wrapper):
    """Gymnasium wrapper that executes policy chunks via TOPP-RA.

    Expected wrapper stack (innermost → outermost):

        MjDexEnv  →  FlattenObsWrapper  →  ActionNormalizeEnv  →  RACEEnvWrapper

    i.e. wrap the env returned by ``create_env()`` directly::

        env, eval_env = create_env(env_name)
        env = RACEEnvWrapper(env)

    Interfaces
    ----------
    * ``step(action)``       – unchanged pass-through (single action, normalized).
                               Keeps compatibility with existing evaluation loops.
    * ``step_chunk(chunk)``  – RACE interface.  Accepts a policy chunk of shape
                               ``(T, action_dim)`` or ``(N, T, action_dim)``.
                               Internally runs Best-of-N selection (if N>1),
                               TOPP-RA interpolation on the arm joints, and
                               ``env.step()`` for every control-rate dense step.
                               Returns the standard Gymnasium 5-tuple with the
                               same obs/reward format as ``step()``.
    """

    def __init__(
        self,
        env: gym.Env,
        vel_limits: np.ndarray | None = None,
        acc_limits: np.ndarray | None = None,
        **toppra_kwargs: Any,
    ) -> None:
        """
        Args:
            env:            ActionNormalizeEnv wrapping FlattenObsWrapper wrapping MjDexEnv.
            vel_limits:     Joint velocity limits [rad/s]. Defaults to UR5E values.
            acc_limits:     Joint acceleration limits [rad/s²]. Defaults to UR5E values.
            **toppra_kwargs: Forwarded to TOPPRAInterpolator (n_grid, vel_safety, …).
        """
        super().__init__(env)
        self._vel_limits = vel_limits
        self._acc_limits = acc_limits
        self._toppra_kwargs = toppra_kwargs
        self._toppra: TOPPRAInterpolator | None = None  # built after first reset
        self._last_renders: list = []  # frames captured during last step_chunk call

    # ------------------------------------------------------------------
    # Gymnasium overrides
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Build TOPP-RA interpolator now that the model is compiled
        mjdex_env = self.unwrapped
        self._toppra = make_toppra_for_env(
            mjdex_env,
            vel_limits=self._vel_limits,
            acc_limits=self._acc_limits,
            **self._toppra_kwargs,
        )
        return obs, info

    # step() is inherited from gym.Wrapper → passes through to ActionNormalizeEnv
    # unchanged, keeping full backward-compatibility.

    # ------------------------------------------------------------------
    # RACE interface
    # ------------------------------------------------------------------

    def step_chunk(
        self,
        chunk: np.ndarray,
        n_eval_pts: int = 50,
        open_loop_steps: int | None = None,
        video_frame_skip: int = 0,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute a policy chunk with TOPP-RA time-optimal interpolation.

        Args:
            chunk:            ``(T, action_dim)`` or ``(N, T, action_dim)`` normalized
                              actions from the policy.  If 3-D, Best-of-N is applied
                              first (arm-joints only; gripper ignored for scoring).
            n_eval_pts:       Spline evaluation points for Best-of-N curvature score.
            open_loop_steps:  If set, TOPP-RA plans over all T waypoints but only
                              executes the dense steps up to waypoint
                              ``open_loop_steps``.  Set this to ``inference_steps``
                              when the policy horizon is longer than the re-query
                              interval (horizon_steps > inference_steps).

        Returns:
            (obs, reward, terminated, truncated, info) – same format as ``step()``.
        """
        if self._toppra is None:
            raise RuntimeError("Call reset() before step_chunk().")

        chunk = np.asarray(chunk, dtype=np.float64)
        mjdex_env = self.unwrapped
        arm_dof: int = mjdex_env._arm_dof

        # ---- Best-of-N selection (only when N > 1) -------------------------
        if chunk.ndim == 3:
            # (N, T, action_dim) → unnormalize in one broadcast call → score arm joints
            unnorm_all = self.env.unnormalize_action(chunk)  # (N, T, action_dim)
            q0 = mjdex_env.data.qpos[mjdex_env._arm_qpos_ids].copy()
            qd0 = mjdex_env.data.qvel[mjdex_env._arm_dof_ids].copy()
            _, best_idx = best_of_n_select(
                unnorm_all[:, :, :arm_dof],
                q0,
                qd0,
                policy_dt=self._toppra.policy_dt,
                n_eval_pts=n_eval_pts,
            )
            chunk = chunk[best_idx]  # → (T, action_dim)

        # ---- Unnormalize single chunk --------------------------------------
        unnorm = self.env.unnormalize_action(chunk)  # (T, action_dim)
        arm_chunk = unnorm[:, :arm_dof]  # (T, arm_dof)
        gripper_chunk = unnorm[:, arm_dof:]  # (T, gripper_dof)

        # ---- TOPP-RA: plan at policy_dt resolution -------------------------
        q0 = mjdex_env.data.qpos[mjdex_env._arm_qpos_ids].copy()
        qd0 = mjdex_env.data.qvel[mjdex_env._arm_dof_ids].copy()
        dense_arm, s_query = self._toppra.compute(
            arm_chunk, q0, qd0, open_loop_steps=open_loop_steps
        )  # (M, arm_dof), (M,)

        # ---- Gripper: index by path parameter, not wall clock --------------
        # s_query[i] ∈ [0, T]: 0 = q0, k = waypoint k.
        # Waypoint k corresponds to gripper_chunk[k-1] (0-indexed).
        # floor(s_query - 1) maps s ∈ [k, k+1) → gripper index k-1.
        M = len(dense_arm)
        T = len(gripper_chunk)
        gripper_idx = np.clip(np.floor(s_query - 1.0).astype(int), 0, T - 1)
        gripper_schedule = gripper_chunk[gripper_idx]  # (M, gripper_dof)

        # ---- Normalize and execute via env.step() --------------------------
        # env.step() runs the full n_steps physics sub-steps per control step,
        # properly settles the PD controller, and handles obs normalization and
        # episode bookkeeping automatically through the wrapper chain.
        self._last_dense_len = M
        self._last_renders = []
        full_traj = np.concatenate([dense_arm, gripper_schedule], axis=1)
        full_traj_norm = self.env.normalize_action(full_traj)  # (M, action_dim)

        obs, reward, terminated, truncated, info = None, 0.0, False, False, {}
        for i in range(M):
            obs, reward, terminated, truncated, info = self.env.step(full_traj_norm[i])
            if video_frame_skip > 0 and i % video_frame_skip == 0:
                frame = self.render()
                if frame is not None:
                    self._last_renders.append(frame.copy())
            if terminated or truncated:
                break

        return obs, float(reward), terminated, truncated, info


# ---------------------------------------------------------------------------
# create_race_env — drop-in replacement for create_env
# ---------------------------------------------------------------------------


def create_race_env(
    env_name: str,
    dataset_dir: str | None = None,
    control_hz: float = 20.0,
    vel_limits: np.ndarray | None = None,
    acc_limits: np.ndarray | None = None,
    **toppra_kwargs: Any,
) -> "RACEEnvWrapper":
    """Create a single RACEEnvWrapper for deployment / evaluation.

    Args:
        env_name:    Gymnasium environment ID.
        dataset_dir: Root dataset directory (e.g. "data/").  When given,
                     obs normalization stats and action bounds are loaded from
                     ``<dataset_dir>/<env_name>.pkl`` — required for the policy
                     to receive correctly scaled observations at inference time.
        control_hz:  Policy control frequency [Hz].
    """
    import os
    from utils.env_utils import create_env  # local import avoids circular deps

    if dataset_dir is not None:
        ds_path = os.path.join(dataset_dir, env_name)
        env, _, _, _ = create_env(env_name, dataset_dir=ds_path, control_hz=control_hz)
    else:
        env, _ = create_env(env_name, dataset_dir=None, control_hz=control_hz)

    return RACEEnvWrapper(
        env,
        vel_limits=vel_limits,
        acc_limits=acc_limits,
        **toppra_kwargs,
    )


# ---------------------------------------------------------------------------
# race_evaluate
# ---------------------------------------------------------------------------


def race_evaluate(
    agent,
    env: "RACEEnvWrapper",
    config,
    num_eval_episodes: int = 50,
    num_video_episodes: int = 0,
    video_frame_skip: int = 3,
    action_dim: int | None = None,
    num_samples: int = 1,
    open_loop_horizon: int = 8,
):
    """Evaluate a policy in a RACEEnvWrapper using chunk-level execution.

    Differences from ``evaluate()``:

    * One policy call produces a full action chunk ``(T, action_dim)``.
      The chunk is passed to ``env.step_chunk()`` which runs TOPP-RA
      internally — no manual action queue needed.
    * If ``num_samples > 1``, the policy is called ``num_samples`` times per
      step and the resulting ``(N, T, action_dim)`` tensor is passed to
      ``step_chunk()``, which applies Best-of-N selection automatically.

    Args:
        agent:               Trained policy agent (must have ``sample_actions``).
        env:                 A :class:`RACEEnvWrapper` instance.
        config:              Agent config dict (must contain ``"action_dim"``
                             and either ``"inference_steps"`` or
                             ``"horizon_steps"``).
        num_eval_episodes:   Episodes used for statistics.
        num_video_episodes:  Extra episodes rendered to video (appended after
                             eval episodes, not counted in stats).
        video_frame_skip:    Render every N-th physics step.
        action_dim:          Override ``config["action_dim"]`` if needed.
        num_samples:         Number of policy samples per chunk for Best-of-N.
                             1 = disabled (single deterministic chunk).

    Returns:
        stats  (dict):            Per-episode metrics averaged over all eval
                                  episodes (same keys as ``evaluate()``).
        trajs  (list[dict]):      Per-episode trajectory dicts.
        renders (list[ndarray]):  Video frames for ``num_video_episodes``.
    """
    from collections import defaultdict

    import jax
    from tqdm import trange

    from utils.evaluation import supply_rng, flatten, add_to

    if action_dim is None:
        action_dim = config["action_dim"]

    # horizon_steps: full chunk predicted by the policy (used as TOPP-RA waypoints)
    # inference_steps: how many waypoints to actually execute before re-querying
    # When horizon_steps == inference_steps (or only one is set), they are the same.
    if "horizon_steps" in config:
        chunk_len = config["horizon_steps"]
    else:
        chunk_len = config["inference_steps"]

    # open_loop_horizon: how many waypoints to execute before re-querying the policy.
    # Matches the reference `open_loop_horizon` parameter (separate from inference_steps).
    open_loop_steps = open_loop_horizon if open_loop_horizon < chunk_len else None

    actor_fn = supply_rng(
        agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))
    )

    trajs = []
    stats = defaultdict(list)
    renders = []

    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        done = False
        total_rewards = 0.0
        render = []

        while not done:
            # --- Sample chunk(s) from policy --------------------------------
            if num_samples == 1:
                raw = actor_fn(observations=observation)
                chunk = np.clip(
                    np.array(raw).reshape(chunk_len, action_dim),
                    -1.0,
                    1.0,
                )  # (T, action_dim)
            else:
                # Tile observation into a batch and call the policy once.
                # This is equivalent to N sequential calls but uses a single
                # batched JAX forward pass — much faster for diffusion policies.
                obs_batch = np.tile(observation[None], (num_samples, 1))  # (N, obs_dim)
                raw = actor_fn(observations=obs_batch)  # (N, T, action_dim)
                chunk = np.clip(
                    np.array(raw).reshape(num_samples, chunk_len, action_dim),
                    -1.0,
                    1.0,
                )  # (N, T, action_dim)

            # --- Execute via TOPP-RA ----------------------------------------
            next_observation, reward, terminated, truncated, info = env.step_chunk(
                chunk,
                open_loop_steps=open_loop_steps,
                video_frame_skip=video_frame_skip if should_render else 0,
            )
            done = terminated or truncated
            total_rewards += reward

            # --- Collect frames captured during execution -------------------
            if should_render:
                render.extend(getattr(env, "_last_renders", []))

            add_to(
                traj,
                dict(
                    observation=observation,
                    next_observation=next_observation,
                    chunk=chunk if chunk.ndim == 2 else chunk[0],
                    reward=reward,
                    done=done,
                    info=info,
                ),
            )
            observation = next_observation

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            stats["total_rewards"].append(total_rewards)
            trajs.append(traj)
        else:
            renders.append(np.array(render) if render else np.empty((0,)))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
