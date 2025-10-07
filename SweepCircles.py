import numpy as np
import torch

from Robot import Robot
from Circle import CircularTrajectory3D
from IKModel import IKNet, decode_angles

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Model hyperparams ----
MODEL_KW = dict(
    in_features=6,
    hidden=2048,
    n_blocks=10,
    dropout_p=0.15,
)

# ---- Inference options ----
MC_RUNS      = 1       # Monte Carlo dropout passes (set >1 to enable)
USE_DROPOUT  = False   # set True (together with MC_RUNS>1) to sample with dropout
REFINE_STEPS = 6       # set >0 to run DLS refinement per point (slower)

# ---- Sweep space (kept compact and sensible for this robot’s DH) ----
X_CENTERS   = [0.4, 0.5, 0.6]
Y_CENTERS   = [0.0, 0.2]
Z_CENTERS   = [0.9, 1.1]
RADII       = [0.35, 0.45]
TILTS_THETA = [np.pi/6, np.pi/4]  # around y
TILTS_PHI   = [0.0, np.pi/6]      # around z
N_POINTS    = 120                 # points per circle

# Joint limits for evaluation Robot
Q_RANGES = [
    (-np.pi, np.pi),
    (np.deg2rad(15.0), np.deg2rad(165.0)),
    (np.deg2rad(-85.0), np.deg2rad(85.0)),
]

# Refinement settings (if REFINE_STEPS > 0)
DAMPING_LAMBDA = 1e-2
STEP_CLIP      = np.deg2rad(8.0)

def build_features(P_xyz: np.ndarray, xyz_mean: np.ndarray, xyz_std: np.ndarray,
                   r_mean: float, r_std: float) -> np.ndarray:
    """
    [x_n, y_n, z_n, r_n, sin(phi), cos(phi)]
    """
    P   = P_xyz.astype(np.float32)
    Xn  = (P - xyz_mean) / xyz_std
    r   = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
    rn  = (r - r_mean) / (r_std + 1e-6)
    phi = np.arctan2(P[:, 1], P[:, 0])
    feats = np.column_stack([Xn, rn.astype(np.float32),
                             np.sin(phi).astype(np.float32),
                             np.cos(phi).astype(np.float32)]).astype(np.float32)
    return feats

def refine_one(robot: Robot, q_init: np.ndarray, p_target: np.ndarray,
               steps=1, lm_lambda=1e-2, step_clip=np.deg2rad(8.0)):
    """
    Damped least-squares refinement (few steps) around the model’s prediction.
    """
    q = q_init.astype(np.float64).copy()
    for _ in range(int(max(0, steps))):
        p_current = robot.ee_pos(q)
        error = (p_target.astype(np.float64) - p_current).reshape(3, 1)
        J = robot.jacobian_fd(q)
        JTJ = J.T @ J
        JTJ_reg = JTJ + (lm_lambda ** 2) * np.eye(3, dtype=np.float64)
        dq = np.linalg.solve(JTJ_reg, J.T @ error).reshape(-1)
        # step-size clamp
        norm_dq = np.linalg.norm(dq)
        if norm_dq > step_clip:
            dq *= (step_clip / norm_dq)
        q = robot.clamp(q + dq)
    return q

def circle_metrics(robot: Robot, center, radius, tilt_theta, tilt_phi, model, stats):
    # 1) Desired circle points
    circle = CircularTrajectory3D(center_x=center[0], center_y=center[1], center_z=center[2],
                                  radius=radius, tilt_theta=tilt_theta, tilt_phi=tilt_phi)
    x_d, y_d, z_d = circle.generate_trajectory(points=N_POINTS)
    P_des = np.stack([x_d, y_d, z_d], axis=1).astype(np.float32)

    # 2) Prepare features & predict joints (optionally MC-dropout)
    xyz_mean = stats["mean"].astype(np.float32)
    xyz_std  = stats["std"].astype(np.float32)
    r_mean   = float(stats["r_mean"])
    r_std    = float(stats["r_std"])
    X_feat   = build_features(P_des, xyz_mean, xyz_std, r_mean, r_std)
    X_tensor = torch.from_numpy(X_feat).to(DEVICE)

    with torch.no_grad():
        if MC_RUNS > 1 and USE_DROPOUT:
            preds = []
            model.train()
            for _ in range(MC_RUNS):
                preds.append(model(X_tensor))
            model.eval()
            sincos = torch.stack(preds).mean(dim=0)
        else:
            sincos = model(X_tensor)

        q_pred = decode_angles(sincos).cpu().numpy().astype(np.float64)

    # 3) FK to executed positions
    P_exec = []
    for i in range(q_pred.shape[0]):
        H, _ = robot.fk(q_pred[i])
        P_exec.append(H[:3, 3])
    P_exec = np.array(P_exec, dtype=np.float32)

    rmse_before = float(np.sqrt(np.mean(np.sum((P_exec - P_des) ** 2, axis=1))))

    if REFINE_STEPS > 0:
        Q_refined = []
        for i in range(q_pred.shape[0]):
            Q_refined.append(refine_one(robot, q_pred[i], P_des[i],
                                        steps=REFINE_STEPS, lm_lambda=DAMPING_LAMBDA, step_clip=STEP_CLIP))
        Q_refined = np.array(Q_refined, dtype=np.float64)
        P_exec_r = []
        for i in range(Q_refined.shape[0]):
            H, _ = robot.fk(Q_refined[i])
            P_exec_r.append(H[:3, 3])
        P_exec_r = np.array(P_exec_r, dtype=np.float32)
        rmse_after = float(np.sqrt(np.mean(np.sum((P_exec_r - P_des) ** 2, axis=1))))
    else:
        rmse_after = rmse_before

    # Simple manipulability proxies (fast, no exact IK solve needed):
    #  - proxy_cond: median 1/r_xy (worse if close to axis)
    #  - proxy_limits_dev: median deviation of predicted q from mid-limits
    r_xy = np.sqrt(P_des[:, 0]**2 + P_des[:, 1]**2)
    proxy_cond = float(np.median(1.0 / (r_xy + 1e-6)))  # lower is better
    q_lo = np.array([r[0] for r in Q_RANGES], dtype=np.float64)
    q_hi = np.array([r[1] for r in Q_RANGES], dtype=np.float64)
    q_mid = 0.5 * (q_lo + q_hi)
    dev = np.abs(q_pred - q_mid.reshape(1, 3))
    proxy_limits = float(np.median(dev))  # smaller is better

    return {
        "center": center,
        "radius": radius,
        "tilt_theta": float(tilt_theta),
        "tilt_phi": float(tilt_phi),
        "rmse": rmse_after,
        "rmse_no_refine": rmse_before,
        "proxy_cond": proxy_cond,
        "proxy_limits_dev": proxy_limits
    }

def main():
    # Load model + stats
    model = IKNet(**MODEL_KW).to(DEVICE)
    sd = torch.load("ik_model.pt", map_location=DEVICE)  # make sure file is here
    model.load_state_dict(sd)
    model.eval()
    stats = np.load("xyz_stats.npz")

    robot = Robot(q_ranges=Q_RANGES)

    results = []
    for x in X_CENTERS:
        for y in Y_CENTERS:
            for z in Z_CENTERS:
                for r in RADII:
                    for th in TILTS_THETA:
                        for ph in TILTS_PHI:
                            res = circle_metrics(robot, (x, y, z), r, th, ph, model, stats)
                            results.append(res)
                            print(f"Checked center=({x:.2f},{y:.2f},{z:.2f}), "
                                  f"R={r:.2f}, "
                                  f"tilt=({th:.2f},{ph:.2f}) -> RMSE {res['rmse']:.4f} m")

    # Sort by RMSE (primary) then proxy_cond (secondary)
    results.sort(key=lambda d: (d["rmse"], d["proxy_cond"]))

    print("\nTop 10 configurations by RMSE (then proxy_cond):")
    for i, r in enumerate(results[:10], 1):
        print(f"{i:2d}) "
              f"center=({r['center'][0]:.2f},{r['center'][1]:.2f},{r['center'][2]:.2f}), "
              f"R={r['radius']:.2f}, "
              f"tilt=({r['tilt_theta']:.2f},{r['tilt_phi']:.2f}) | "
              f"RMSE={r['rmse']:.4f} m (no-refine {r['rmse_no_refine']:.4f}), "
              f"proxy_cond={r['proxy_cond']:.4f}, "
              f"proxy_limits_dev={r['proxy_limits_dev']:.2f} rad")

if __name__ == "__main__":
    main()
