import numpy as np
import torch
import matplotlib.pyplot as plt

from Robot import Robot
from Circle import CircularTrajectory3D
from IKModel import IKNet, decode_angles
from TrainIK import (
    FOCUS_CENTER,
    FOCUS_RADIUS,
    FOCUS_TILT_PHI,
    FOCUS_TILT_THETA,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define model hyperparameters (must match training)
HIDDEN   = 2048
N_BLOCKS = 10
DROP_OUT = 0.15

# How many Monte Carlo dropout passes to average (set 1 to disable)
MC_RUNS = 5

# Damped least-squares IK refinement settings
REFINE_STEPS   = 6
DAMPING_LAMBDA = 1e-2
STEP_CLIP      = np.deg2rad(8.0)

def build_features(P_xyz: np.ndarray, xyz_mean: np.ndarray, xyz_std: np.ndarray,
                   r_mean: float, r_std: float) -> np.ndarray:
    """Construct feature vectors [x_n, y_n, z_n, r_n, sin(phi), cos(phi)] for positions."""
    P = P_xyz.astype(np.float32)
    Xn = (P - xyz_mean) / xyz_std
    r = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
    rn = (r - r_mean) / (r_std + 1e-6)
    phi = np.arctan2(P[:, 1], P[:, 0])
    sin_phi = np.sin(phi).astype(np.float32)
    cos_phi = np.cos(phi).astype(np.float32)
    feats = np.column_stack([Xn, rn.astype(np.float32), sin_phi, cos_phi]).astype(np.float32)
    return feats

def refine_one(robot: Robot, q_init: np.ndarray, p_target: np.ndarray,
               steps=1, lm_lambda=1e-2, step_clip=np.pi/8):
    """
    Refine a single IK solution using damped least squares (Levenberg-Marquardt).
    """
    q = q_init.astype(np.float64).copy()
    for _ in range(int(max(0, steps))):
        p_current = robot.ee_pos(q)
        error = (p_target.astype(np.float64) - p_current).reshape(3, 1)
        J = robot.jacobian_fd(q)
        JTJ = J.T @ J
        JTJ_reg = JTJ + (lm_lambda ** 2) * np.eye(3, dtype=np.float64)
        dq = np.linalg.solve(JTJ_reg, J.T @ error).reshape(-1)
        norm_dq = np.linalg.norm(dq)
        if norm_dq > step_clip:
            dq *= (step_clip / norm_dq)
        q = robot.clamp(q + dq)
    return q

def main():
    # Load model and stats
    model = IKNet(in_features=6, hidden=HIDDEN, n_blocks=N_BLOCKS, dropout_p=DROP_OUT).to(DEVICE)
    model.load_state_dict(torch.load("ik_model.pt", map_location=DEVICE))
    model.eval()
    stats = np.load("xyz_stats.npz")
    xyz_mean = stats["mean"].astype(np.float32)
    xyz_std  = stats["std"].astype(np.float32)
    r_mean   = float(stats["r_mean"])
    r_std    = float(stats["r_std"])

    # Define the testing circle shared with training focus diagnostics so it
    # remains on the reachable workspace sphere.
    circle = CircularTrajectory3D(
        center_x=FOCUS_CENTER[0],
        center_y=FOCUS_CENTER[1],
        center_z=FOCUS_CENTER[2],
        tilt_theta=FOCUS_TILT_THETA,
        tilt_phi=FOCUS_TILT_PHI,
        radius=FOCUS_RADIUS,
    )
    points = 400
    x_d, y_d, z_d = circle.generate_trajectory(points=points)
    P_des = np.stack([x_d, y_d, z_d], axis=1).astype(np.float32)

    residual = x_d**2 + y_d**2 + (z_d - 1.0) ** 2 - 2.0
    assert np.all(np.abs(residual) < 1e-3), (
        "Focus circle no longer lies on reachable workspace sphere; adjust "
        "FOCUS_CENTER/FOCUS_RADIUS."
    )

    # Build input features for all desired points
    X_feat = build_features(P_des, xyz_mean, xyz_std, r_mean, r_std)
    X_tensor = torch.from_numpy(X_feat).to(DEVICE)

    # Predict with optional MC dropout ensemble
    with torch.no_grad():
        if MC_RUNS > 1:
            preds = []
            model.train()  # enable dropout
            for _ in range(MC_RUNS):
                preds.append(model(X_tensor))
            model.eval()
            sincos_pred = torch.stack(preds).mean(dim=0)
        else:
            sincos_pred = model(X_tensor)
        q_pred = decode_angles(sincos_pred)  # (points,3)
        q_pred = q_pred.cpu().numpy().astype(np.float64)

    # Compute executed positions from predicted joints
    rob = Robot(q_ranges=[(-np.pi,np.pi), (np.deg2rad(15.0), np.deg2rad(165.0)), (np.deg2rad(-85.0), np.deg2rad(85.0))])
    P_exec = []
    for qi in q_pred:
        H, _ = rob.fk(qi)
        P_exec.append(H[:3, 3])
    P_exec = np.array(P_exec, dtype=np.float32)
    rmse_before = float(np.sqrt(np.mean(np.sum((P_exec - P_des) ** 2, axis=1))))
    print(f"[Before refinement] RMSE = {rmse_before:.4f} m")

    # Optional refinement of each point
    if REFINE_STEPS > 0:
        Q_refined = []
        for i in range(q_pred.shape[0]):
            q0 = q_pred[i]
            target = P_des[i]
            q_ref = refine_one(rob, q0, target, steps=REFINE_STEPS,
                               lm_lambda=DAMPING_LAMBDA, step_clip=STEP_CLIP)
            Q_refined.append(q_ref)
        Q_refined = np.array(Q_refined, dtype=np.float64)

        P_exec_refined = []
        for qi in Q_refined:
            H, _ = rob.fk(qi)
            P_exec_refined.append(H[:3, 3])
        P_exec_refined = np.array(P_exec_refined, dtype=np.float32)
        rmse_after = float(np.sqrt(np.mean(np.sum((P_exec_refined - P_des) ** 2, axis=1))))
        print(f"[After refinement] RMSE = {rmse_after:.4f} m")
    else:
        P_exec_refined = P_exec
        rmse_after = rmse_before

    # Plot the desired vs executed trajectories
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P_des[:,0], P_des[:,1], P_des[:,2], label="Desired circle", linewidth=2, color='C0')
    ax.plot(P_exec[:,0], P_exec[:,1], P_exec[:,2], label="Executed (predicted)", linestyle='--', color='C1')
    if REFINE_STEPS > 0:
        ax.plot(P_exec_refined[:,0], P_exec_refined[:,1], P_exec_refined[:,2],
                label="Executed (refined)", linestyle=':', color='C2')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("IK Model Tracking a 3D Circular Trajectory")
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
