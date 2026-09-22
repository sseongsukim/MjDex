"""Key-vector-based retargeting optimizer.

Reference: dex-retargeting VectorOptimizer (arxiv 2506.11916)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_optimizer import BaseOptimizer
from .utils import M_TO_CM, TimingStats, huber_loss_np, huber_loss_grad_np


class KeyVectorOptimizer(BaseOptimizer):
    """Key-vector-based retargeting optimizer.

    Each key vector is defined by an (origin_link, task_link) pair on the robot,
    matched to a (origin_kp, task_kp) MediaPipe keypoint pair.
    The optimizer minimizes the mean Huber distance between robot vectors and
    scaled human vectors.

    Loss:
        L(q) = (1/N) * Σᵢ Huber(‖[FK(task_i)-FK(origin_i)] - scale_i*(mp[task_kp_i]-mp[origin_kp_i])‖)
             + norm_delta * ‖q - q_prev‖²

    Config (retarget.key_vectors, required):
        - origin: robot origin link name
        - task:   robot task link name
        - origin_kp: MediaPipe origin keypoint index
        - task_kp:   MediaPipe task keypoint index
        - scale:     scale applied to human vector (default 1.0)
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self._timing = TimingStats()
        self._enable_timing = True

        retarget_config = config.get('retarget', {})
        robot_config = config.get('robot', {})
        robot_type = robot_config.get('type', 'shadow_hand')

        kv_config = retarget_config.get('key_vectors')
        if not kv_config:
            raise ValueError(
                "retarget.key_vectors is required for KeyVectorOptimizer. "
                "Define a list of {origin, task, origin_kp, task_kp, scale} entries."
            )

        origin_names_raw = [kv['origin'] for kv in kv_config]
        task_names_raw   = [kv['task']   for kv in kv_config]

        # Apply same left-hand name replacement as BaseOptimizer
        if robot_type == 'shadow_hand' and self.hand_side == 'left':
            def _replace(name):
                return name.replace('rh_', 'lh_')
            origin_names = [_replace(n) for n in origin_names_raw]
            task_names   = [_replace(n) for n in task_names_raw]
        elif robot_type == 'unitree_dex5_hand' and self.hand_side == 'left':
            def _replace(name):
                if name == 'base_link00':
                    return 'base_link00L'
                return f"{name[:-1]}L" if name.endswith('R') else name
            origin_names = [_replace(n) for n in origin_names_raw]
            task_names   = [_replace(n) for n in task_names_raw]
        else:
            origin_names = origin_names_raw
            task_names   = task_names_raw

        # Deduplicate links to avoid redundant FK/Jacobian computation.
        # The origin link (e.g. palm) typically appears for all N vectors.
        all_names = list(dict.fromkeys(origin_names + task_names))
        self._kv_computed_link_names   = all_names
        self._kv_computed_link_indices = [self.robot.get_link_index(n) for n in all_names]
        self._kv_origin_indices = np.array([all_names.index(n) for n in origin_names], dtype=int)
        self._kv_task_indices   = np.array([all_names.index(n) for n in task_names],   dtype=int)

        self._origin_kp_indices = np.array([kv['origin_kp'] for kv in kv_config], dtype=int)
        self._task_kp_indices   = np.array([kv['task_kp']   for kv in kv_config], dtype=int)
        self._vector_scalings   = np.array([kv.get('scale', 1.0) for kv in kv_config], dtype=np.float64)
        self.num_vectors        = len(kv_config)

        # Pre-allocated zero offsets (key vectors use link frame origins, no local offset)
        n_unique = len(all_names)
        self._zero_offsets = np.zeros((n_unique, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    # Target vector computation
    # ------------------------------------------------------------------

    def _compute_target_vectors(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute scaled target vectors from MediaPipe keypoints.

        Args:
            keypoints: (21, 3) MediaPipe keypoints in wrist frame (meters)

        Returns:
            (N, 3) target vectors in cm
        """
        origin_kp = keypoints[self._origin_kp_indices]  # (N, 3)
        task_kp   = keypoints[self._task_kp_indices]    # (N, 3)
        vecs = (task_kp - origin_kp) * self._vector_scalings[:, None] * M_TO_CM
        return vecs.astype(np.float64)

    # ------------------------------------------------------------------
    # Loss and analytical gradient
    # ------------------------------------------------------------------

    def _loss_and_grad(
        self,
        qpos: np.ndarray,
        target_vectors: np.ndarray,
        last_qpos: Optional[np.ndarray],
    ) -> tuple[float, np.ndarray]:
        """Compute loss and analytical gradient.

        Args:
            qpos: (num_joints,) full joint positions
            target_vectors: (N, 3) scaled target vectors in cm
            last_qpos: (num_joints,) previous qpos for regularization, or None

        Returns:
            (loss, full_grad) where full_grad has shape (num_joints,)
        """
        qpos = np.asarray(qpos, dtype=np.float64)

        # FK: world positions for all unique links
        positions = self.robot.compute_points_batch(
            qpos,
            self._kv_computed_link_indices,
            self._zero_offsets,
        ) * M_TO_CM  # (num_unique_links, 3)

        # Jacobians: (num_unique_links, 3, nq)
        Js = self.robot.compute_all_jacobians_batch_with_offsets(
            qpos,
            self._kv_computed_link_indices,
            self._zero_offsets,
        ) * M_TO_CM

        # Per-vector positions and Jacobians
        origin_pos = positions[self._kv_origin_indices]  # (N, 3)
        task_pos   = positions[self._kv_task_indices]    # (N, 3)
        J_origin   = Js[self._kv_origin_indices]         # (N, 3, nq)
        J_task     = Js[self._kv_task_indices]           # (N, 3, nq)

        # Residuals
        robot_vec = task_pos - origin_pos                # (N, 3)
        diff      = robot_vec - target_vectors           # (N, 3)
        dist      = np.linalg.norm(diff, axis=1)         # (N,)

        # Mean Huber loss
        total_loss = float(np.mean(huber_loss_np(dist, self.huber_delta)))

        # Analytical gradient via chain rule
        huber_grad  = huber_loss_grad_np(dist, self.huber_delta)  # (N,)
        diff_normed = diff / (dist[:, None] + 1e-8)               # (N, 3)

        total_grad = np.zeros(self.num_joints, dtype=np.float64)
        N = self.num_vectors
        for i in range(N):
            J_diff = J_task[i] - J_origin[i]                      # (3, nq)
            total_grad += (huber_grad[i] / N) * (diff_normed[i] @ J_diff)

        # Regularization: penalize large joint velocity
        if last_qpos is not None:
            delta = qpos - last_qpos
            total_loss += self.norm_delta * float(np.sum(delta ** 2))
            total_grad += 2.0 * self.norm_delta * delta

        return total_loss, total_grad

    # ------------------------------------------------------------------
    # NLopt objective factory
    # ------------------------------------------------------------------

    def _get_objective(
        self,
        target_vectors: np.ndarray,
        last_qpos: Optional[np.ndarray],
    ):
        """Create NLopt objective function with analytical gradient.

        The objective operates on independent joints only (num_opt_vars).
        Mimic joints are expanded internally before FK, and the gradient is
        mapped back via chain rule.
        """
        target_vectors = np.asarray(target_vectors, dtype=np.float64)
        if last_qpos is not None:
            last_qpos = np.asarray(last_qpos, dtype=np.float64)

        def objective(x: np.ndarray, grad_out: np.ndarray) -> float:
            opt_vars  = np.asarray(x, dtype=np.float64)
            full_qpos = self.expand_to_full_qpos(opt_vars)

            loss, full_grad = self._loss_and_grad(full_qpos, target_vectors, last_qpos)

            if grad_out.size > 0:
                grad_out[:] = self.map_gradient_to_independent(full_grad)

            if self._enable_timing:
                self._timing.record_iter_loss(float(loss))

            return float(loss)

        return objective

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def solve(
        self,
        mediapipe_keypoints: np.ndarray,
        last_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve for joint angles given MediaPipe keypoints."""
        if self._enable_timing:
            self._timing.start_frame()

        mediapipe_keypoints = np.asarray(mediapipe_keypoints, dtype=np.float64)
        if mediapipe_keypoints.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {mediapipe_keypoints.shape}")

        target_vectors = self._compute_target_vectors(mediapipe_keypoints)
        reg_qpos  = self._get_reg_qpos(last_qpos)
        init_qpos = self._get_init_qpos(last_qpos)

        objective_fn = self._get_objective(target_vectors, reg_qpos)
        result = self._run_optimization(objective_fn, init_qpos)

        if self._enable_timing:
            self._timing.end_frame(self.opt.get_numevals())

        return result

    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        """Compute loss for given joint angles."""
        target_vectors = self._compute_target_vectors(
            np.asarray(mediapipe_keypoints, dtype=np.float64)
        )
        loss, _ = self._loss_and_grad(
            np.asarray(qpos, dtype=np.float64), target_vectors, None
        )
        return float(loss)

    # ------------------------------------------------------------------
    # Timing (required by benchmark_frames.py)
    # ------------------------------------------------------------------

    def get_timing_stats(self) -> TimingStats:
        """Get timing statistics."""
        return self._timing

    def reset_timing_stats(self):
        """Reset timing statistics."""
        self._timing.reset()

    def set_timing_enabled(self, enabled: bool):
        """Enable or disable timing instrumentation."""
        self._enable_timing = enabled
