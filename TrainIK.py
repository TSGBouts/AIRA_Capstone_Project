import argparse
import gc
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from numpy.random import Generator
from torch import nn
from torch.cuda import amp

from Robot import Robot
from IKModel import IKNet, decode_angles
from Circle import CircularTrajectory3D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = torch.cuda.is_available()


def resolve_device(preferred: str | None) -> str:
    """Return a valid device string, falling back to CPU if necessary."""
    if preferred is not None:
        requested = preferred.lower()
        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            print("[Config] Requested CUDA but no GPU detected. Falling back to CPU.", flush=True)
            return "cpu"
        if requested == "cpu":
            return "cpu"
        print(f"[Config] Unknown device '{preferred}', defaulting to auto-detect.", flush=True)
    return "cuda" if torch.cuda.is_available() else "cpu"


def free_cuda_cache() -> None:
    """Release cached GPU memory and run a garbage collection pass."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move ``tensor`` to the active device if required."""
    if tensor.device.type == DEVICE:
        return tensor
    return tensor.to(DEVICE, non_blocking=True)


class Tee:
    """Tee stdout to both console and an optional log file."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_stdout(log_path: str | None):
    """Context manager that mirrors stdout to ``log_path`` when provided."""
    if not log_path:
        yield
        return

    log_file = open(log_path, "a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

# ---------------- Hyperparameters ----------------
N_SAMPLES     = 30000         # total dataset size after deduplication
TRAIN_SPLIT   = 0.8
EPOCHS        = 1000
BATCH         = 128
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
LR_FACTOR     = 0.5   # multiplicative LR drop when validation stalls
LR_PATIENCE   = 25     # epochs of no improvement before lowering LR
LR_MIN        = 1e-5   # clamp LR to this floor
SEED          = 0

# Early stopping
EARLY_STOP_PATIENCE = 100              # epochs of no blended improvement
BLEND_WEIGHT = 0.70                    # blended early-stop metric: 0.7*val_RMSE + 0.3*circle_RMSE

# Loss mixing weights
W_CART = 0.8   # weight on Cartesian position loss
W_ANG  = 0.2   # weight on angle (sincos) loss

# Model architecture hyperparameters
HIDDEN       = 2048     # hidden layer size (increased for capacity)
N_BLOCKS     = 10       # number of gated residual blocks (deeper network)
DROP_OUT     = 0.15     # dropout probability for regularization

# Joint-space sampling ranges
Q1_RANGE = (-np.pi, np.pi)
Q2_RANGE = (np.deg2rad(25), np.deg2rad(135))
Q3_RANGE = (np.deg2rad(-60), np.deg2rad(60))
MIN_RADIUS_M = 0.14     # minimum radius in XY to avoid singularity (covers focus circle perigee)

# Circle parameters for validation diagnostics.
#
# ``eval_val_sets`` uses a single canonical focus circle for held-out metrics.
# Training samples, however, are now drawn from a distribution of tilted circles
# (see ``sample_training_focus_configs`` below).  The canonical configuration is
# reserved exclusively for validation / test diagnostics so that reported metrics
# remain comparable across experiments.
FOCUS_TILT_THETA = 0.79  # ~45° canonical tilt polar angle
FOCUS_TILT_PHI   = 0.52  # ~30° canonical tilt azimuth
FOCUS_OFFSET     = 1.0   # canonical offset along the workspace normal
_workspace_center = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_workspace_radius = np.sqrt(2.0)


def _focus_normal_from_angles(tilt_theta: float, tilt_phi: float) -> np.ndarray:
    normal = np.array([
        np.sin(tilt_theta) * np.cos(tilt_phi),
        np.sin(tilt_theta) * np.sin(tilt_phi),
        np.cos(tilt_theta),
    ], dtype=np.float64)
    return normal / (np.linalg.norm(normal) + 1e-12)


def _focus_circle_geometry(tilt_theta: float,
                           tilt_phi: float,
                           offset: float) -> tuple[np.ndarray, float, np.ndarray]:
    if offset >= _workspace_radius:
        raise ValueError("Focus circle offset must lie within the reachable workspace sphere")
    normal = _focus_normal_from_angles(tilt_theta, tilt_phi)
    center = _workspace_center + offset * normal
    radius = float(np.sqrt(max(1e-12, _workspace_radius**2 - offset**2)))
    return center, radius, normal


FOCUS_CENTER, FOCUS_RADIUS, _ = _focus_circle_geometry(FOCUS_TILT_THETA,
                                                        FOCUS_TILT_PHI,
                                                        FOCUS_OFFSET)
FOCUS_CENTER = tuple(FOCUS_CENTER.tolist())
FOCUS_FRACTION   = 0.30  # target fraction of training points near the focus circle
PLANE_BAND       = 0.020 # +/- 2 cm tolerance perpendicular to the circle plane
RING_BAND        = 0.025 # +/- 2.5 cm tolerance radially within the circle plane

# Training focus-circle distribution.  ``sample_training_focus_configs`` draws
# randomized tilts/offsets from these ranges to diversify the augmented samples.
FOCUS_TRAIN_TILT_THETA_RANGE = (np.deg2rad(35.0), np.deg2rad(70.0))
FOCUS_TRAIN_TILT_PHI_RANGE   = (0.0, 2.0 * np.pi)
FOCUS_TRAIN_OFFSET_RANGE     = (0.85, 1.10)
FOCUS_TRAIN_CONFIGS          = 3

# EMA (Exponential Moving Average) of weights for stabler eval/saving
EMA_ENABLED   = True
EMA_BETA      = 0.999
EMA_USE_FOR_EVAL = True  # evaluate/snapshot using EMA weights

# Error-driven hard example mining (H.E.M.)
MINING_ENABLED         = True
MINING_PERIOD_EPOCHS   = 50     # also mine on fixed cadence (e.g. every 50 epochs)
MINING_ON_STALL        = True
MINING_STALL_EPOCHS    = max(1, EARLY_STOP_PATIENCE // 4)  # e.g., after 25 epochs no improvement
MINING_POOL_CIRCLE_PTS = 1000   # dense circle points in the pool
MINING_POOL_RANDOM     = 4000   # random workspace points in the pool
MINING_TOP_FRAC        = 0.15   # mine top 15% hardest pool points
MINING_SEEDS_PER_POINT = 3      # DLS local solves per hard point
MINING_MAX_APPEND      = 5000   # cap how many newly mined samples we add per mining round
MINING_REFINE_STEPS    = 6      # iterative DLS steps per refinement attempt
MINING_ACCEPT_TOLERANCE_M = 0.002  # accept solutions within 2 mm residual error

# ---------------- Utility Functions ----------------
def iterate_minibatches_with_idx(X, Y, batch_size):
    """Shuffle and yield mini-batches with indices."""
    idx = torch.randperm(X.shape[0], device=X.device)
    for i in range(0, X.shape[0], batch_size):
        j = idx[i:i+batch_size]
        yield X[j], Y[j], j

def cosine_pair_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance loss for (sin, cos) pairs:
    1 - sum_over_pairs(sin_pred*sin_true + cos_pred*cos_true) averaged.
    """
    p = pred.view(-1, 3, 2)
    t = target.view(-1, 3, 2)
    return (1.0 - (p * t).sum(dim=2)).mean()

def build_features(P_xyz: np.ndarray,
                   xyz_mean: np.ndarray,
                   xyz_std: np.ndarray,
                   r_mean: float,
                   r_std: float) -> np.ndarray:
    """
    Build input features [x_n, y_n, z_n, r_n, sin(phi), cos(phi)] from 3D positions.
    - Normalize XYZ by mean/std.
    - Normalize radial distance r by mean/std.
    - Compute azimuth phi and take sin, cos.
    """
    P = P_xyz.astype(np.float32)
    # Normalize x, y, z coordinates
    Xn = (P - xyz_mean) / xyz_std
    # Radial distance in XY-plane and normalize
    r  = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
    rn = (r - r_mean) / (r_std + 1e-6)
    # Angle in XY-plane
    phi = np.arctan2(P[:, 1], P[:, 0])
    feats = np.column_stack([
        Xn,
        rn.astype(np.float32),
        np.sin(phi).astype(np.float32),
        np.cos(phi).astype(np.float32)
    ]).astype(np.float32)
    return feats


def compute_normalization_stats(*position_sets: np.ndarray) -> dict:
    """Compute normalization statistics from one or more position arrays."""
    stacks = [p.astype(np.float32) for p in position_sets if p.size > 0]
    if len(stacks) == 0:
        raise ValueError("At least one non-empty position array is required for normalization stats")
    P = np.vstack(stacks)
    xyz_mean = P.mean(axis=0, keepdims=True).astype(np.float32)
    xyz_std = (P.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    r = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
    r_mean = np.float32(r.mean())
    r_std = np.float32(r.std() + 1e-6)
    return {
        "mean": xyz_mean,
        "std": xyz_std,
        "r_mean": r_mean,
        "r_std": r_std,
    }


def rebuild_dataset_tensors(train_P_np: np.ndarray,
                             train_Y_np: np.ndarray,
                             test_P_np: np.ndarray,
                             test_Y_np: np.ndarray):
    """Recompute normalization stats and rebuild tensors on the target device."""
    stats = compute_normalization_stats(train_P_np, test_P_np)
    train_X_np = build_features(train_P_np, stats["mean"], stats["std"], stats["r_mean"], stats["r_std"])
    test_X_np = build_features(test_P_np, stats["mean"], stats["std"], stats["r_mean"], stats["r_std"])
    train_X = torch.from_numpy(train_X_np)
    test_X = torch.from_numpy(test_X_np)
    train_Y = torch.from_numpy(train_Y_np.astype(np.float32))
    test_Y = torch.from_numpy(test_Y_np.astype(np.float32))
    train_P = torch.from_numpy(train_P_np.astype(np.float32))
    test_P = torch.from_numpy(test_P_np.astype(np.float32))

    if DEVICE == "cuda":
        train_X = train_X.pin_memory()
        test_X = test_X.pin_memory()
        train_Y = train_Y.pin_memory()
        test_Y = test_Y.pin_memory()
        train_P = train_P.pin_memory()
        test_P = test_P.pin_memory()
    return stats, train_X, test_X, train_Y, test_Y, train_P, test_P

def sample_fk_dataset(rob: Robot, n_joint: int, q_ranges, rng: Generator | None = None):
    """
    Sample random joint configurations and filter out those too close to base axis.
    """
    lo = np.array([q_ranges[0][0], q_ranges[1][0], q_ranges[2][0]], dtype=np.float64)
    hi = np.array([q_ranges[0][1], q_ranges[1][1], q_ranges[2][1]], dtype=np.float64)
    rng_local = rng if rng is not None else np.random.default_rng()
    Q, P = [], []
    while len(Q) < n_joint:
        q = lo + (hi - lo) * rng_local.random(3)
        T, _ = rob.fk(q)
        p = T[:3, 3]
        if np.hypot(p[0], p[1]) < MIN_RADIUS_M:
            continue
        Q.append(q.astype(np.float32))
        P.append(p.astype(np.float32))
    return np.asarray(Q, dtype=np.float32), np.asarray(P, dtype=np.float32)

def dedup_by_voxel(P: np.ndarray, Q: np.ndarray, voxel=0.015):
    """
    Deduplicate by 3D voxel: keep one sample per small voxel of size ~1.5cm.
    Returns Q_kept, P_kept with consistent indexing.
    """
    vox = np.floor(P / voxel).astype(np.int32)
    keys = [tuple(v) for v in vox]
    seen = {}
    keep_indices = []
    order = np.random.permutation(len(P))
    for i in order:
        k = keys[i]
        if k in seen:
            continue
        seen[k] = True
        keep_indices.append(i)
    keep_indices = np.array(keep_indices, dtype=np.int32)
    return Q[keep_indices], P[keep_indices]

def compute_focus_frame(circle: CircularTrajectory3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the (normal, u_axis, v_axis) basis for the tilted circle plane.
    """
    normal = circle.normal.astype(np.float32)
    normal /= np.linalg.norm(normal) + 1e-9
    u = circle.u_axis.astype(np.float32)
    u /= np.linalg.norm(u) + 1e-9
    v = circle.v_axis.astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return normal, u, v

def compute_focus_mask(P: np.ndarray,
                       center: np.ndarray,
                       normal: np.ndarray,
                       u_axis: np.ndarray,
                       v_axis: np.ndarray,
                       radius: float,
                       plane_band: float,
                       ring_band: float) -> np.ndarray:
    """
    Boolean mask selecting points within the focus bands of the tilted circle.
    """
    d_plane = np.abs((P - center) @ normal)
    Pu = (P - center) @ u_axis
    Pv = (P - center) @ v_axis
    r_in_plane = np.sqrt(Pu**2 + Pv**2)
    d_ring = np.abs(r_in_plane - radius)
    return (d_plane < plane_band) & (d_ring < ring_band)


@dataclass
class FocusCircleConfig:
    center: np.ndarray
    radius: float
    tilt_theta: float
    tilt_phi: float
    offset: float
    normal: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray


def _validate_circle_on_workspace(circle: CircularTrajectory3D, tol: float = 1e-3) -> None:
    """Ensure a circle trajectory lies on the reachable workspace sphere."""

    x_probe, y_probe, z_probe = circle.generate_trajectory(points=16)
    probe_pts = np.column_stack((x_probe, y_probe, z_probe)).astype(np.float64)
    workspace_residual = (
        probe_pts[:, 0] ** 2
        + probe_pts[:, 1] ** 2
        + (probe_pts[:, 2] - 1.0) ** 2
        - 2.0
    )
    if not np.all(np.abs(workspace_residual) < tol):
        raise ValueError(
            "Focus circle no longer lies on reachable workspace sphere; adjust parameters."
        )


def build_focus_circle_config(tilt_theta: float,
                              tilt_phi: float,
                              offset: float) -> FocusCircleConfig:
    """Create a ``FocusCircleConfig`` for the given tilt/origin parameters."""

    center, radius, normal = _focus_circle_geometry(tilt_theta, tilt_phi, offset)
    circle = CircularTrajectory3D(center_x=center[0], center_y=center[1], center_z=center[2],
                                  radius=radius, tilt_theta=tilt_theta, tilt_phi=tilt_phi)
    _validate_circle_on_workspace(circle)
    basis_normal, u_axis, v_axis = compute_focus_frame(circle)
    return FocusCircleConfig(center=center.astype(np.float32),
                             radius=float(radius),
                             tilt_theta=float(tilt_theta),
                             tilt_phi=float(tilt_phi),
                             offset=float(offset),
                             normal=basis_normal,
                             u_axis=u_axis,
                             v_axis=v_axis)


def sample_training_focus_configs(rng: Generator,
                                  n_configs: int) -> list[FocusCircleConfig]:
    """Draw randomized focus-circle configurations for training augmentation."""

    if n_configs <= 0:
        return []

    configs: list[FocusCircleConfig] = []
    canonical_center = np.asarray(FOCUS_CENTER, dtype=np.float32)
    max_attempts = max(10 * n_configs, 10)
    attempts = 0
    while len(configs) < n_configs and attempts < max_attempts:
        attempts += 1
        tilt_theta = float(rng.uniform(*FOCUS_TRAIN_TILT_THETA_RANGE))
        tilt_phi = float(rng.uniform(*FOCUS_TRAIN_TILT_PHI_RANGE))
        offset = float(rng.uniform(*FOCUS_TRAIN_OFFSET_RANGE))
        try:
            cfg = build_focus_circle_config(tilt_theta, tilt_phi, offset)
        except ValueError:
            continue
        if (np.allclose(cfg.center, canonical_center, atol=1e-4)
                and abs(cfg.radius - FOCUS_RADIUS) < 1e-4):
            # Skip the canonical configuration; it is reserved for evaluation.
            continue
        configs.append(cfg)
    return configs


def compute_focus_union_mask(P: np.ndarray,
                             configs: list[FocusCircleConfig],
                             plane_band: float,
                             ring_band: float) -> np.ndarray:
    """Union of focus masks across multiple ``FocusCircleConfig`` instances."""

    if not configs:
        return np.zeros(P.shape[0], dtype=bool)

    union_mask = np.zeros(P.shape[0], dtype=bool)
    for cfg in configs:
        union_mask |= compute_focus_mask(P,
                                         cfg.center,
                                         cfg.normal,
                                         cfg.u_axis,
                                         cfg.v_axis,
                                         cfg.radius,
                                         plane_band,
                                         ring_band)
    return union_mask

def _newton_adjust_focus(q1: float,
                         q2: float,
                         plane_target: float,
                         ring_target: float,
                         center: np.ndarray,
                         normal: np.ndarray,
                         u_axis: np.ndarray,
                         v_axis: np.ndarray,
                         radius: float,
                         plane_tol: float,
                         ring_tol: float,
                         max_iters: int = 15) -> tuple[float, float] | None:
    """Solve for (q1, q2) whose FK point hits the requested focus offsets."""

    q1_curr = float(q1)
    q2_curr = float(q2)
    damping = 1e-4

    for _ in range(max_iters):
        s1, c1 = np.sin(q1_curr), np.cos(q1_curr)
        s2, c2 = np.sin(q2_curr), np.cos(q2_curr)

        reach = c2 + s2
        x = c1 * reach
        y = s1 * reach
        z = 1.0 + s2 - c2

        px = x - center[0]
        py = y - center[1]
        pz = z - center[2]

        plane_val = px * normal[0] + py * normal[1] + pz * normal[2]
        pu = px * u_axis[0] + py * u_axis[1] + pz * u_axis[2]
        pv = px * v_axis[0] + py * v_axis[1] + pz * v_axis[2]
        r_plane = np.hypot(pu, pv)
        ring_val = r_plane - radius

        res_plane = plane_val - plane_target
        res_ring = ring_val - ring_target

        if abs(res_plane) <= plane_tol and abs(res_ring) <= ring_tol:
            return q1_curr, q2_curr

        dreach_dq2 = c2 - s2
        dx_dq1 = -s1 * reach
        dy_dq1 = c1 * reach
        dz_dq1 = 0.0
        dx_dq2 = c1 * dreach_dq2
        dy_dq2 = s1 * dreach_dq2
        dz_dq2 = c2 + s2

        df_plane_dq1 = normal[0] * dx_dq1 + normal[1] * dy_dq1 + normal[2] * dz_dq1
        df_plane_dq2 = normal[0] * dx_dq2 + normal[1] * dy_dq2 + normal[2] * dz_dq2

        denom = max(r_plane, 1e-9)
        dpu_dq1 = u_axis[0] * dx_dq1 + u_axis[1] * dy_dq1 + u_axis[2] * dz_dq1
        dpu_dq2 = u_axis[0] * dx_dq2 + u_axis[1] * dy_dq2 + u_axis[2] * dz_dq2
        dpv_dq1 = v_axis[0] * dx_dq1 + v_axis[1] * dy_dq1 + v_axis[2] * dz_dq1
        dpv_dq2 = v_axis[0] * dx_dq2 + v_axis[1] * dy_dq2 + v_axis[2] * dz_dq2
        df_ring_dq1 = (pu * dpu_dq1 + pv * dpv_dq1) / denom
        df_ring_dq2 = (pu * dpu_dq2 + pv * dpv_dq2) / denom

        J = np.array([[df_plane_dq1, df_plane_dq2],
                      [df_ring_dq1, df_ring_dq2]], dtype=np.float64)
        residual = np.array([res_plane, res_ring], dtype=np.float64)

        JTJ = J.T @ J
        rhs = -J.T @ residual
        JTJ[0, 0] += damping * damping
        JTJ[1, 1] += damping * damping

        try:
            delta = np.linalg.solve(JTJ, rhs)
        except np.linalg.LinAlgError:
            return None

        step_norm = np.linalg.norm(delta)
        if step_norm > np.deg2rad(10.0):
            delta *= np.deg2rad(10.0) / step_norm

        q1_curr = ((q1_curr + delta[0] + np.pi) % (2.0 * np.pi)) - np.pi
        q2_curr = np.clip(q2_curr + delta[1], Q2_RANGE[0], Q2_RANGE[1])

    return None


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return ((angle + np.pi) % (2.0 * np.pi)) - np.pi


def _coarse_seed(robot: Robot,
                 target: np.ndarray,
                 q1_nominal: float,
                 q2_grid: np.ndarray,
                 q3_grid: np.ndarray) -> np.ndarray:
    """Brute-force search over a coarse joint grid to obtain a robust IK seed."""
    best_q = None
    best_err = np.inf
    for dq1 in (-0.35, 0.0, 0.35):  # +/-20° nudges to escape wrap issues
        q1 = _wrap_angle(q1_nominal + dq1)
        for q2 in q2_grid:
            for q3 in q3_grid:
                q = np.array([q1, q2, q3], dtype=np.float64)
                if not robot.within_limits(q):
                    continue
                p = robot.ee_pos(q)
                err = np.linalg.norm(p - target)
                if err < best_err:
                    best_err = err
                    best_q = q
    if best_q is None:
        # Fallback: clamp nominal guess inside limits
        best_q = robot.clamp(np.array([q1_nominal, q2_grid[len(q2_grid)//2], q3_grid[len(q3_grid)//2]],
                              dtype=np.float64))
    return best_q


def sample_focus_ik(robot: Robot,
                    n_samples: int,
                    circle_center: np.ndarray,
                    normal: np.ndarray,
                    u_axis: np.ndarray,
                    v_axis: np.ndarray,
                    radius: float,
                    rng: Generator,
                    seeds_per_target: int = 5,
                    max_attempt_factor: int = 10,
                    accept_tolerance_m: float = 0.010,
                    log_every_attempts: int | None = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Generate training samples that lie close to the focus circle.

    Args:
        seeds_per_target:  Number of random joint seeds to try per Cartesian
            target; increasing this improves robustness at the cost of compute.
        accept_tolerance_m: Maximum Cartesian miss distance to accept a refined
            joint configuration.
    """

    if n_samples <= 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32))

    rng_local = rng if rng is not None else np.random.default_rng()

    center = np.asarray(circle_center, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    u_axis = np.asarray(u_axis, dtype=np.float64)
    u_axis /= np.linalg.norm(u_axis) + 1e-12
    v_axis = np.asarray(v_axis, dtype=np.float64)
    v_axis /= np.linalg.norm(v_axis) + 1e-12
    radius = float(radius)

    plane_tol = PLANE_BAND
    ring_tol = RING_BAND

    q2_grid = np.linspace(Q2_RANGE[0], Q2_RANGE[1], 9, dtype=np.float64)
    q3_grid = np.linspace(Q3_RANGE[0], Q3_RANGE[1], 9, dtype=np.float64)

    q_lo = np.array([Q1_RANGE[0], Q2_RANGE[0], Q3_RANGE[0]], dtype=np.float64)
    q_hi = np.array([Q1_RANGE[1], Q2_RANGE[1], Q3_RANGE[1]], dtype=np.float64)

    accepted_q: list[np.ndarray] = []
    accepted_p: list[np.ndarray] = []
    accepted_total = 0
    attempts = 0
    max_attempts = max(n_samples * max_attempt_factor, n_samples * 50)
    log_interval = max(1, log_every_attempts) if log_every_attempts else None

    while accepted_total < n_samples and attempts < max_attempts:
        attempts += 1

        angle = rng_local.uniform(0.0, 2.0 * np.pi)
        radial_offset = rng_local.uniform(-RING_BAND, RING_BAND)
        plane_offset = rng_local.uniform(-PLANE_BAND, PLANE_BAND)

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        in_plane_dir = cos_a * u_axis + sin_a * v_axis

        target = (center +
                  (radius + radial_offset) * in_plane_dir +
                  plane_offset * normal)

        success = False

        q1_nominal = np.arctan2(target[1], target[0])
        coarse_seed = _coarse_seed(robot, target, q1_nominal, q2_grid, q3_grid)

        candidate_pool = [coarse_seed]
        for _ in range(max(0, int(seeds_per_target) - 1)):
            q_rand = q_lo + (q_hi - q_lo) * rng_local.random(3)
            candidate_pool.append(q_rand)

        best_local_q = None
        best_local_err = np.inf

        for q_seed in candidate_pool:
            q_candidate = refine_one(robot, q_seed, target,
                                     steps=25,
                                     lm_lambda=5e-3,
                                     step_clip=np.deg2rad(12.0))
            if not robot.within_limits(q_candidate):
                continue

            p_candidate = robot.ee_pos(q_candidate)
            miss = np.linalg.norm(p_candidate - target)
            if not np.isfinite(miss):
                continue

            if miss < best_local_err:
                best_local_err = miss
                best_local_q = q_candidate

            if miss > accept_tolerance_m:
                continue

            if np.hypot(p_candidate[0], p_candidate[1]) < MIN_RADIUS_M:
                continue

            px = p_candidate - center
            plane_dist = abs(px @ normal)
            pu = px @ u_axis
            pv = px @ v_axis
            ring_dist = abs(np.hypot(pu, pv) - radius)

            if plane_dist >= plane_tol or ring_dist >= ring_tol:
                continue

            accepted_q.append(q_candidate.astype(np.float32))
            accepted_p.append(p_candidate.astype(np.float32))
            accepted_total += 1
            success = True
            break

        if not success and best_local_q is not None and best_local_err <= accept_tolerance_m * 1.5:
            p_candidate = robot.ee_pos(best_local_q)
            if np.hypot(p_candidate[0], p_candidate[1]) >= MIN_RADIUS_M:
                px = p_candidate - center
                plane_dist = abs(px @ normal)
                pu = px @ u_axis
                pv = px @ v_axis
                ring_dist = abs(np.hypot(pu, pv) - radius)
                if plane_dist < plane_tol * 1.2 and ring_dist < ring_tol * 1.2:
                    accepted_q.append(best_local_q.astype(np.float32))
                    accepted_p.append(p_candidate.astype(np.float32))
                    accepted_total += 1
                    success = True

        if success and accepted_total % 512 == 0:
            plane_tol = max(plane_tol * 0.9, PLANE_BAND * 0.25)
            ring_tol = max(ring_tol * 0.9, RING_BAND * 0.25)

        if log_interval and attempts % log_interval == 0:
            print(f"[Focus]   attempts={attempts:,} | accepted={accepted_total:,}/{n_samples:,}")

    if accepted_total < n_samples:
        print(
            "[Focus] Warning: generated "
            f"{accepted_total} focus samples out of requested {n_samples} after "
            f"{attempts} attempts."
        )

    if accepted_total == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32))

    Q_new = np.stack(accepted_q, axis=0)
    P_new = np.stack(accepted_p, axis=0)

    if Q_new.shape[0] > n_samples:
        Q_new = Q_new[:n_samples]
        P_new = P_new[:n_samples]

    return Q_new, P_new

def fk_pos_torch(q: torch.Tensor) -> torch.Tensor:
    """
    Differentiable FK: maps (batch,3) joint angles to end-effector (batch,3) positions.

    For this specific manipulator the end-effector position depends only on q1 and q2,
    allowing a lightweight closed-form expression that avoids the expensive batched
    4x4 matrix products previously used.
    """
    q1 = q[:, 0]
    q2 = q[:, 1]
    c1 = torch.cos(q1)
    s1 = torch.sin(q1)
    c2 = torch.cos(q2)
    s2 = torch.sin(q2)

    reach = c2 + s2
    x = c1 * reach
    y = s1 * reach
    z = q1.new_tensor(1.0) + s2 - c2
    return torch.stack((x, y, z), dim=1)

# ---------------- Refinement (for mining) ----------------
def refine_one(robot: Robot, q_init: np.ndarray, p_target: np.ndarray,
               steps=6, lm_lambda=1e-2, step_clip=np.deg2rad(8.0)):
    """
    Damped least-squares refinement (few steps) around a seed q to reach p_target.
    """
    q = np.asarray(q_init, dtype=np.float64).copy().reshape(3,)
    for _ in range(int(max(0, steps))):
        p_current = robot.ee_pos(q)
        error = (np.asarray(p_target, dtype=np.float64) - p_current).reshape(3, 1)
        J = robot.jacobian_fd(q)
        JTJ = J.T @ J
        JTJ_reg = JTJ + (lm_lambda ** 2) * np.eye(3, dtype=np.float64)
        dq = np.linalg.solve(JTJ_reg, J.T @ error).reshape(-1)
        # step size clamp
        norm_dq = np.linalg.norm(dq)
        if norm_dq > step_clip:
            dq *= (step_clip / norm_dq)
        q = robot.clamp(q + dq)
    return q

# ---------------- EMA helper ----------------
class EMAHelper:
    def __init__(self, model: nn.Module, beta: float = 0.999):
        self.beta = beta
        # Keep EMA weights on the same device as the model parameters to avoid
        # host-device transfers every update when training on GPU.
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.beta).add_(v.detach(), alpha=1.0 - self.beta)

    def load_into(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

    def copy_state(self):
        # return a deep copy for snapshotting best weights
        return {k: v.clone() for k, v in self.shadow.items()}

# ---------------- Mining helpers ----------------
def build_circle_points(center, radius, tilt_theta, tilt_phi, n_points):
    circle = CircularTrajectory3D(center_x=center[0], center_y=center[1], center_z=center[2],
                                  radius=radius, tilt_theta=tilt_theta, tilt_phi=tilt_phi)
    x, y, z = circle.generate_trajectory(points=n_points)
    return np.stack([x, y, z], axis=1).astype(np.float32)

def mine_hard_examples(model: IKNet,
                       rob: Robot,
                       stats: dict,
                       pool_P: np.ndarray,
                       rng: Generator,
                       top_frac: float = 0.15,
                       seeds: int = 3,
                       max_append: int = 5000,
                       refine_steps: int = MINING_REFINE_STEPS,
                       accept_tol_m: float = MINING_ACCEPT_TOLERANCE_M) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate current model on a pool of Cartesian targets; select the hardest top_frac,
    and for each run a small local DLS solve from multiple random seeds to label (P->Q).
    Returns (Q_new, P_new). Refinement depth and acceptance tolerance are controlled by
    `refine_steps` and `accept_tol_m` so they can be tuned without modifying the
    implementation.

    Args:
        rng: numpy random Generator used for drawing initial joint seeds so
            successive mining rounds advance the generator state and explore
            different local solutions while remaining deterministic for a fixed
            seed.
    """
    model.eval()
    xyz_mean, xyz_std, r_mean, r_std = stats["mean"], stats["std"], float(stats["r_mean"]), float(stats["r_std"])

    # 1) Predict q for pool points and measure error
    X = build_features(pool_P, xyz_mean, xyz_std, r_mean, r_std)
    Xt = torch.from_numpy(X).to(DEVICE)
    pool_P_t = torch.from_numpy(pool_P).to(DEVICE)
    with torch.no_grad():
        with amp.autocast(enabled=AMP_ENABLED):
            sincos = model(Xt)
            q_pred = decode_angles(sincos)
            P_exec = fk_pos_torch(q_pred)
    P_exec = P_exec.float()
    pool_P_t = pool_P_t.float()
    err = torch.norm(P_exec - pool_P_t, dim=1).cpu().numpy()

    # 2) Select hardest indices
    k = max(1, int(top_frac * len(pool_P)))
    hard_idx = np.argpartition(err, -k)[-k:]
    # limit how many we will attempt to append (speed)
    if hard_idx.size > max_append:
        hard_idx = hard_idx[np.argsort(err[hard_idx])[-max_append:]]

    # 3) Local IK refinement from multiple random seeds
    Q_new, P_new = [], []
    q_lo = np.array([Q1_RANGE[0], Q2_RANGE[0], Q3_RANGE[0]], dtype=np.float64)
    q_hi = np.array([Q1_RANGE[1], Q2_RANGE[1], Q3_RANGE[1]], dtype=np.float64)
    for idx in hard_idx:
        p = pool_P[idx]
        best_q, best_e = None, np.inf
        for _ in range(seeds):
            q0 = q_lo + (q_hi - q_lo) * rng.random(3)
            qh = refine_one(rob, q0, p, steps=refine_steps, lm_lambda=1e-2, step_clip=np.deg2rad(8.0))
            e = np.linalg.norm(rob.ee_pos(qh) - p)
            if e < best_e:
                best_e, best_q = e, qh
        if best_q is not None and best_e <= accept_tol_m:
            Q_new.append(best_q.astype(np.float32))
            P_new.append(p.astype(np.float32))
        else:
            residual_mm = best_e * 1000.0 if np.isfinite(best_e) else float("inf")
            print(
                f"[Mining][Skip] Discarded pool index {idx}: residual {residual_mm:.2f} mm exceeds tolerance "
                f"{accept_tol_m * 1000:.2f} mm."
            )
    if len(Q_new) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.asarray(Q_new, dtype=np.float32), np.asarray(P_new, dtype=np.float32)

# ---------------- Validation helpers ----------------
@torch.no_grad()
def eval_val_sets(model: IKNet,
                  test_X: torch.Tensor,
                  test_Y: torch.Tensor,
                  test_P: torch.Tensor,
                  stats: dict,
                  circle_center: tuple,
                  circle_radius: float,
                  circle_tilt_theta: float,
                  circle_tilt_phi: float,
                  circle_points: int = 400) -> dict:
    """
    Compute:
      - val_CART (MSE on Cartesian positions),
      - val_ANG  (sincos loss),
      - val_RMSE (Cartesian RMSE),
      - circle_RMSE (RMSE on the chosen circle).
    """
    mse = nn.MSELoss()

    test_X_d = to_device(test_X)
    test_Y_d = to_device(test_Y)
    test_P_d = to_device(test_P)

    # General validation (random workspace split)
    with amp.autocast(enabled=AMP_ENABLED):
        pred_v = model(test_X_d)
        q_v = decode_angles(pred_v)
        xyz_v = fk_pos_torch(q_v)
        loss_cart_v = mse(xyz_v, test_P_d)
        loss_ang_v_tensor = 0.5 * mse(pred_v, test_Y_d) + 0.5 * cosine_pair_loss(pred_v, test_Y_d)
        rmse_v_tensor = torch.sqrt(((xyz_v - test_P_d).pow(2).sum(dim=1)).mean())
    loss_cart_v = float(loss_cart_v.float().item())
    loss_ang_v = float(loss_ang_v_tensor.float().item())
    rmse_v = float(rmse_v_tensor.float().item())

    # Circle-specific validation
    P_circ = build_circle_points(circle_center, circle_radius, circle_tilt_theta, circle_tilt_phi, circle_points)
    X_circ = build_features(P_circ, stats["mean"], stats["std"], stats["r_mean"], stats["r_std"])
    X_circ_t = to_device(torch.from_numpy(X_circ))
    with amp.autocast(enabled=AMP_ENABLED):
        sc = model(X_circ_t)
        q_c = decode_angles(sc)
        xyz_c = fk_pos_torch(q_c)
    P_circ_t = to_device(torch.from_numpy(P_circ))
    circle_rmse = float(torch.sqrt(((xyz_c - P_circ_t).pow(2).sum(dim=1)).mean()).float().item())

    del test_X_d, test_Y_d, test_P_d, X_circ_t, P_circ_t

    return {
        "val_CART": loss_cart_v,
        "val_ANG":  loss_ang_v,
        "val_RMSE": rmse_v,
        "circle_RMSE": circle_rmse
    }

# ---------------- CLI helpers ----------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the IK model with RunPod-friendly defaults.")
    parser.add_argument("--batch-size", type=int, default=BATCH,
                        help="Mini-batch size to use during training (default: %(default)s).")
    parser.add_argument("--min-batch-size", type=int, default=16,
                        help="Lower bound for automatic batch size reduction on CUDA OOM events.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs to run (default: %(default)s).")
    parser.add_argument("--max-samples", type=int, default=N_SAMPLES,
                        help="Maximum number of deduplicated samples to keep in the dataset.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="Force a specific compute device instead of auto-detect.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Optional path to append stdout/stderr logs for post-mortem debugging.")
    return parser.parse_args()


# ---------------- Main Training Pipeline ----------------
def main(args: argparse.Namespace, batch_size: int):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    epochs = max(1, int(args.epochs))
    n_samples = max(1, int(args.max_samples))

    print(
        f"[Config] device={DEVICE} | epochs={epochs} | batch_size={batch_size} | "
        f"target_samples={n_samples}",
        flush=True,
    )
    rob = Robot(q_ranges=[Q1_RANGE, Q2_RANGE, Q3_RANGE])
    dataset_rng = np.random.default_rng(SEED + 2021)

    # Pre-compute randomized focus circle configurations for dataset balancing.
    # The canonical circle defined by ``FOCUS_*`` remains reserved for
    # evaluation/metrics and is not part of this list.
    focus_rng = np.random.default_rng(SEED + 404)
    training_focus_configs = sample_training_focus_configs(focus_rng, FOCUS_TRAIN_CONFIGS)
    if training_focus_configs:
        print(f"[Focus] Using {len(training_focus_configs)} randomized training circle(s) for augmentation.")
    else:
        print("[Focus] Focus circle augmentation disabled (no randomized circles requested).")

    # 1) Sample a large set of random joint configurations
    Q_raw, P_raw = sample_fk_dataset(
        rob,
        n_joint=int(n_samples * 5.0),  # oversample 5x before dedup
        q_ranges=[Q1_RANGE, Q2_RANGE, Q3_RANGE],
        rng=dataset_rng,
    )

    # 2) Deduplicate by voxel
    Q_all, P_all = dedup_by_voxel(P_raw, Q_raw, voxel=0.015)
    # If too many points, randomly trim
    if P_all.shape[0] > n_samples:
        idx = np.random.permutation(P_all.shape[0])[:n_samples]
        Q_all, P_all = Q_all[idx], P_all[idx]
    else:
        # If too few after dedup, sample more
        need = n_samples - P_all.shape[0]
        if need > 0:
            Q_more, P_more = sample_fk_dataset(
                rob,
                n_joint=need,
                q_ranges=[Q1_RANGE, Q2_RANGE, Q3_RANGE],
                rng=dataset_rng,
            )
            Q_all = np.concatenate([Q_all, Q_more], axis=0)
            P_all = np.concatenate([P_all, P_more], axis=0)

    print(f"Final dataset size: {P_all.shape[0]} (after dedup)")

    # 3) Targets: sin/cos of angles
    S_all = np.sin(Q_all)
    C_all = np.cos(Q_all)
    Y_all = np.concatenate([S_all, C_all], axis=1).astype(np.float32)  # shape (N,6)

    # 4) Split into train/validation partitions (store as numpy for later re-normalization)
    n_total = P_all.shape[0]
    n_train = int(TRAIN_SPLIT * n_total)
    train_P_np = P_all[:n_train].copy()
    test_P_np = P_all[n_train:].copy()
    train_Y_np = Y_all[:n_train].copy()
    test_Y_np = Y_all[n_train:].copy()

    # Append fresh focus-circle IK samples for each randomized configuration.
    total_configs = len(training_focus_configs)
    for cfg_idx, cfg in enumerate(training_focus_configs, start=1):
        deg_theta = np.rad2deg(cfg.tilt_theta)
        deg_phi = np.rad2deg(cfg.tilt_phi)
        print(
            f"[Focus] Config {cfg_idx}/{total_configs}: "
            f"tilt_theta={deg_theta:.1f}° tilt_phi={deg_phi:.1f}° offset={cfg.offset:.3f} m."
        )

        while True:
            target_focus_total = int(np.ceil(FOCUS_FRACTION * train_P_np.shape[0]))
            target_focus_cfg = int(np.ceil(target_focus_total / total_configs)) if total_configs else 0
            focus_mask = compute_focus_mask(train_P_np, cfg.center, cfg.normal,
                                            cfg.u_axis, cfg.v_axis, cfg.radius,
                                            PLANE_BAND, RING_BAND)
            current_focus = int(focus_mask.sum())
            if target_focus_cfg == 0 or current_focus >= target_focus_cfg:
                break

            needed_focus = target_focus_cfg - current_focus
            print(
                f"[Focus] Config {cfg_idx}/{total_configs} collecting up to {needed_focus} new samples..."
            )
            Q_focus, P_focus = sample_focus_ik(
                rob,
                needed_focus,
                cfg.center,
                cfg.normal,
                cfg.u_axis,
                cfg.v_axis,
                cfg.radius,
                focus_rng,
                accept_tolerance_m=0.006,
                log_every_attempts=2000,
            )
            if Q_focus.shape[0] < needed_focus:
                print(
                    f"[Focus] Requested {needed_focus} focus samples but only obtained {Q_focus.shape[0]}."
                )
            if Q_focus.shape[0] == 0:
                print(f"[Focus] Config {cfg_idx}/{total_configs} produced no additional samples; moving on.")
                break

            Q_focus, P_focus = dedup_by_voxel(P_focus, Q_focus, voxel=0.015)
            if Q_focus.shape[0] == 0:
                print(f"[Focus] Config {cfg_idx}/{total_configs} yielded duplicates only after dedup; moving on.")
                break

            Y_focus = np.concatenate([np.sin(Q_focus), np.cos(Q_focus)], axis=1).astype(np.float32)
            train_P_np = np.concatenate([train_P_np, P_focus], axis=0)
            train_Y_np = np.concatenate([train_Y_np, Y_focus], axis=0)

        focus_mask = compute_focus_mask(train_P_np, cfg.center, cfg.normal,
                                        cfg.u_axis, cfg.v_axis, cfg.radius,
                                        PLANE_BAND, RING_BAND)
        coverage_pct = focus_mask.mean() * 100.0 if focus_mask.size > 0 else 0.0
        print(
            f"[Focus] Config {cfg_idx}/{total_configs} coverage: {coverage_pct:.1f}% of training set."
        )

    focus_union_mask = compute_focus_union_mask(train_P_np, training_focus_configs,
                                                PLANE_BAND, RING_BAND)
    train_focus_frac = float(focus_union_mask.mean()) if focus_union_mask.size > 0 else 0.0
    if training_focus_configs:
        print(
            "[Focus] Training focus coverage across all randomized circles: "
            f"{train_focus_frac*100:.1f}% (target ~{FOCUS_FRACTION*100:.0f}%)."
        )
        assert abs(train_focus_frac - FOCUS_FRACTION) <= max(0.1, 0.25 * FOCUS_FRACTION), \
            "Focus coverage deviates significantly from requested fraction."

    # 6) Build tensors with up-to-date normalization statistics
    norm_stats, train_X, test_X, train_Y, test_Y, train_P, test_P = rebuild_dataset_tensors(
        train_P_np, train_Y_np, test_P_np, test_Y_np
    )

    print(f"Train set: {train_X.shape[0]}, Val set: {test_X.shape[0]}")

    # 6) Initialize model, optimizer, scheduler, EMA
    model = IKNet(in_features=6, hidden=HIDDEN, n_blocks=N_BLOCKS, dropout_p=DROP_OUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        threshold=1e-5,
        min_lr=LR_MIN,
    )

    ema = EMAHelper(model, beta=EMA_BETA) if EMA_ENABLED else None
    mining_rng = np.random.default_rng(SEED + 12345)

    # Early stopping variables
    best_score = float('inf')    # blended score (lower is better)
    best_state = None            # snapshot of (EMA or model) weights at best score
    patience_counter = 0
    patience_limit = min(EARLY_STOP_PATIENCE, epochs)
    mining_stall_epochs = min(MINING_STALL_EPOCHS, patience_limit)

    def evaluate_current():
        # Optionally evaluate with EMA weights
        if EMA_ENABLED and EMA_USE_FOR_EVAL:
            # backup current state, load EMA shadow for eval
            current_state = {k: v.clone() for k, v in model.state_dict().items()}
            ema.load_into(model)
            model.eval()
            # Validation remains tied to the canonical focus circle defined by ``FOCUS_*``.
            metrics = eval_val_sets(model, test_X, test_Y, test_P,
                                    norm_stats,
                                    FOCUS_CENTER, FOCUS_RADIUS, FOCUS_TILT_THETA, FOCUS_TILT_PHI)
            # restore current state
            model.load_state_dict(current_state, strict=True)
            return metrics
        else:
            model.eval()
            return eval_val_sets(model, test_X, test_Y, test_P,
                                 norm_stats,
                                 FOCUS_CENTER, FOCUS_RADIUS, FOCUS_TILT_THETA, FOCUS_TILT_PHI)

    scaler = amp.GradScaler(enabled=AMP_ENABLED)

    # 7) Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb_cpu, yb_cpu, idxb in iterate_minibatches_with_idx(train_X, train_Y, batch_size):
            xb = to_device(xb_cpu)
            yb = to_device(yb_cpu)
            xyz_true = to_device(train_P[idxb])
            opt.zero_grad(set_to_none=True)
            with amp.autocast(enabled=AMP_ENABLED):
                pred = model(xb)               # (batch,6)
                loss_ang = 0.5 * mse(pred, yb) + 0.5 * cosine_pair_loss(pred, yb)
                q_pred = decode_angles(pred)   # (batch,3)
                xyz_pred = fk_pos_torch(q_pred)  # (batch,3)
                loss_cart = mse(xyz_pred, xyz_true)
                # Combined weighted loss
                loss_cart_detached = loss_cart.detach().float()
                loss_ang_detached = loss_ang.detach().float()
                ang_scale = (loss_cart_detached + 1e-6) / (loss_ang_detached + 1e-6)
                ang_scale = torch.clamp(ang_scale, min=0.01, max=10.0)
                effective_ang_weight = (W_ANG * ang_scale).to(loss_cart.dtype)
                loss = W_CART * loss_cart + effective_ang_weight * loss_ang

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item() * xb_cpu.size(0)

            # EMA update per step
            if EMA_ENABLED:
                ema.update(model)

            del (xb, yb, xyz_true, pred, q_pred, xyz_pred,
                 loss, loss_cart, loss_ang,
                 loss_cart_detached, loss_ang_detached,
                 ang_scale, effective_ang_weight)

        epoch_loss /= train_X.size(0)

        # ---- validation at epoch end ----
        metrics = evaluate_current()
        val_cart  = metrics["val_CART"]
        val_ang   = metrics["val_ANG"]
        val_rmse  = metrics["val_RMSE"]
        circ_rmse = metrics["circle_RMSE"]
        blended   = BLEND_WEIGHT * val_rmse + (1.0 - BLEND_WEIGHT) * circ_rmse

        # Manually report learning-rate reductions since some environments
        # (including the one running this script) ship a PyTorch build without
        # the ``verbose`` flag on ReduceLROnPlateau.
        prev_lrs = [group["lr"] for group in opt.param_groups]
        scheduler.step(blended)
        new_lrs = [group["lr"] for group in opt.param_groups]
        if any(new < old for new, old in zip(new_lrs, prev_lrs)):
            lr_str = ", ".join(f"{lr:.2e}" for lr in new_lrs)
            print(f"[LR Scheduler] Plateau detected. Reducing learning rate to: {lr_str}")

        # Log progress every epoch
        print(f"epoch {epoch:4d} | train_loss {epoch_loss:.4f} | "
              f"val_CART {val_cart:.4f} | val_ANG {val_ang:.4f} | "
              f"val_RMSE {val_rmse:.4f} m | circle_RMSE {circ_rmse:.4f} m | "
              f"score {blended:.4f}")

        free_cuda_cache()

        # Early stopping check (blended score)
        if blended < best_score:
            best_score = blended
            patience_counter = 0
            # snapshot current "best" weights (EMA or raw)
            if EMA_ENABLED and EMA_USE_FOR_EVAL:
                best_state = ema.copy_state()
            else:
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # ---- Hard Example Mining triggers ----
        mining_triggered = False
        if MINING_ENABLED:
            if (MINING_PERIOD_EPOCHS > 0 and epoch % MINING_PERIOD_EPOCHS == 0):
                mining_triggered = True
            if MINING_ON_STALL and patience_counter == mining_stall_epochs:
                mining_triggered = True

        if mining_triggered:
            print(f"[Mining] Triggered at epoch {epoch}. Gathering hard examples...")
            # Build mining pool: dense canonical circle + random workspace
            P_circle_pool = build_circle_points(FOCUS_CENTER, FOCUS_RADIUS,
                                                FOCUS_TILT_THETA, FOCUS_TILT_PHI,
                                                MINING_POOL_CIRCLE_PTS)
            Q_rand_pool, P_rand_pool = sample_fk_dataset(
                rob,
                n_joint=MINING_POOL_RANDOM,
                q_ranges=[Q1_RANGE, Q2_RANGE, Q3_RANGE],
                rng=dataset_rng,
            )
            pool_P = np.vstack([P_circle_pool, P_rand_pool]).astype(np.float32)

            # Compose stats dict for building features
            # Evaluate model and mine hardest pool points
            Q_hard, P_hard = mine_hard_examples(
                model, rob, norm_stats, pool_P, mining_rng,
                top_frac=MINING_TOP_FRAC,
                seeds=MINING_SEEDS_PER_POINT,
                max_append=MINING_MAX_APPEND,
                refine_steps=MINING_REFINE_STEPS,
                accept_tol_m=MINING_ACCEPT_TOLERANCE_M,
            )
            print(f"[Mining] Found {len(Q_hard)} hard examples.")

            if len(Q_hard) > 0:
                # Dedup and append to the existing training pool (keep test split fixed!)
                Q_hard, P_hard = dedup_by_voxel(P_hard, Q_hard, voxel=0.015)
                Y_hard = np.concatenate([np.sin(Q_hard), np.cos(Q_hard)], axis=1).astype(np.float32)

                # Append to numpy stores then rebuild tensors/stats
                train_P_np = np.concatenate([train_P_np, P_hard], axis=0)
                train_Y_np = np.concatenate([train_Y_np, Y_hard], axis=0)
                norm_stats, train_X, test_X, train_Y, test_Y, train_P, test_P = rebuild_dataset_tensors(
                    train_P_np, train_Y_np, test_P_np, test_Y_np
                )

                print(f"[Mining] Appended. New train size: {train_X.shape[0]}")

        # Early stopping termination
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch}: no blended improvement in {patience_limit} epochs.")
            break

    # ---- Post-training: load best weights and report ----
    if best_state is not None:
        if EMA_ENABLED and EMA_USE_FOR_EVAL:
            # Load EMA-best snapshot
            model.load_state_dict(best_state, strict=True)
        else:
            model.load_state_dict(best_state, strict=True)

    # Final report with best weights
    final_metrics = eval_val_sets(model, test_X, test_Y, test_P,
                                  norm_stats,
                                  FOCUS_CENTER, FOCUS_RADIUS, FOCUS_TILT_THETA, FOCUS_TILT_PHI)
    print(
        "Training complete. Best blended score: "
        f"{best_score:.4f} | val_RMSE {final_metrics['val_RMSE']:.4f} m | "
        f"circle_RMSE {final_metrics['circle_RMSE']:.4f} m."
    )

    # 8) Save model and normalization stats (weights already loaded with best)
    torch.save(model.state_dict(), "ik_model.pt")
    np.savez(
        "xyz_stats.npz",
        mean=norm_stats["mean"].astype(np.float32),
        std=norm_stats["std"].astype(np.float32),
        r_mean=np.float32(norm_stats["r_mean"]),
        r_std=np.float32(norm_stats["r_std"]),
    )
    print("Saved best model to ik_model.pt and normalization stats to xyz_stats.npz")

def run_with_retries(args: argparse.Namespace) -> None:
    """Run training with automatic batch size back-off on CUDA OOM."""

    batch_size = max(1, int(args.batch_size))
    min_batch = max(1, int(args.min_batch_size))
    if min_batch > batch_size:
        min_batch = batch_size
    attempt = 0

    free_cuda_cache()

    while True:
        attempt += 1
        try:
            main(args, batch_size=batch_size)
            break
        except RuntimeError as exc:
            message = str(exc)
            if "CUDA out of memory" in message and batch_size > min_batch:
                new_batch = max(min_batch, batch_size // 2)
                if new_batch == batch_size and batch_size > min_batch:
                    new_batch = min_batch
                print(
                    f"CUDA OOM detected (attempt {attempt}). "
                    f"Reducing batch size from {batch_size} to {new_batch} and retrying...",
                    flush=True,
                )
                batch_size = new_batch
                free_cuda_cache()
                continue
            if "CUDA out of memory" in message:
                print(
                    "CUDA OOM persists even at the minimum batch size. "
                    "Consider lowering --max-samples or running on CPU.",
                    flush=True,
                )
            raise


def cli_entrypoint() -> None:
    args = parse_args()

    with tee_stdout(args.log_file):
        resolved_device = resolve_device(args.device)
        global DEVICE, AMP_ENABLED
        DEVICE = resolved_device
        AMP_ENABLED = (resolved_device == "cuda" and torch.cuda.is_available())
        try:
            run_with_retries(args)
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                free_cuda_cache()
            raise


if __name__ == "__main__":
    cli_entrypoint()
