# AIRA Capstone Project – Neural Inverse Kinematics

This repository contains a self-contained research sandbox for approximating the inverse kinematics (IK) of a 3-degree-of-freedom (3-DOF) serial robot arm. The project trains a deep neural network to map end-effector targets to joint angles, augments the training data with task-specific circles, and optionally refines predictions with damped least-squares (DLS) steps to improve accuracy on challenging trajectories. The codebase is organized so you can regenerate the dataset from scratch, retrain the model, and run evaluation sweeps with only Python and PyTorch installed.

## Features at a Glance

- **Analytical robot model** – `Robot.py` implements Denavit–Hartenberg (DH) forward kinematics, joint limit handling, and a finite-difference Jacobian for refinement steps.【F:Robot.py†L4-L97】
- **Deep IK network** – `IKModel.py` defines a SiLU-activated residual network that predicts `sin`/`cos` pairs for each joint and normalizes them to unit length before decoding to angles.【F:IKModel.py†L1-L47】
- **Automated dataset synthesis** – `TrainIK.py` samples joint configurations, deduplicates workspace points, and mixes in focus-circle IK solutions to balance coverage near important trajectories.【F:TrainIK.py†L909-L1017】
- **Robust training pipeline** – Training includes EMA weight tracking, adaptive loss balancing, learning-rate scheduling, early stopping, GPU-oom recovery, and optional hard-example mining passes.【F:TrainIK.py†L1018-L1334】
- **Task-focused evaluation** – `RunCircle.py` and `SweepCircles.py` load the saved model, reconstruct feature statistics, and assess trajectory tracking with optional Monte Carlo dropout and DLS refinement.【F:RunCircle.py†L1-L161】【F:SweepCircles.py†L1-L188】

## Repository Layout

```
AIRA_Capstone_Project/
├── BestModel/           # Pretrained weights (ik_model.pt) and normalization stats (xyz_stats.npz)
├── Circle.py            # 3D circle generator used for dataset augmentation and evaluation
├── IKModel.py           # Neural network architecture and angle decoding helper
├── Robot.py             # Robot kinematics, joint limits, and numerical Jacobian
├── RunCircle.py         # Single-circle evaluation and visualization script
├── SweepCircles.py      # Grid search over circles for robustness analysis
├── TrainIK.py           # End-to-end dataset generation and training CLI
└── README.md            # Project documentation (this file)
```

## Getting Started

1. **Install dependencies** (Python 3.9+ recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install torch numpy matplotlib
   ```

   > Replace `torch` with the GPU-enabled wheel for your CUDA version if you plan to train on a GPU.

2. **Download or locate pretrained assets.** The `BestModel/` directory ships with `ik_model.pt` and `xyz_stats.npz`. Copy these files to the repository root (or create symlinks) before running the evaluation scripts, since they expect weights and statistics alongside the Python modules.【F:RunCircle.py†L42-L70】【F:SweepCircles.py†L63-L82】

3. **Run a quick evaluation.** With the model files in place, execute:

   ```bash
   python RunCircle.py
   ```

   The script reports the RMSE before and after optional DLS refinement and plots the desired versus executed 3D trajectory.【F:RunCircle.py†L71-L156】

## Training the IK Model

The training script regenerates a dataset, trains the neural network, and exports the best-performing weights/statistics.

```bash
python TrainIK.py \
    --device cuda \
    --batch-size 128 \
    --epochs 1000 \
    --max-samples 30000
```

Key CLI options are documented in `parse_args()`; omit `--device` or set it to `cpu` to force CPU training.【F:TrainIK.py†L923-L950】 The script automatically retries with smaller batches if it encounters CUDA out-of-memory errors and mirrors logs to a file when `--log-file` is supplied.【F:TrainIK.py†L1235-L1285】

### How the Dataset Is Built

1. Sample a large pool of random joint angles within configurable limits, compute forward kinematics, and deduplicate workspace points by voxels to remove near-duplicates.【F:TrainIK.py†L953-L990】
2. Convert joint angles to `sin`/`cos` targets and split the dataset into training and validation partitions.【F:TrainIK.py†L992-L1010】
3. Augment the training split with IK solutions that deliberately hug randomized focus circles, ensuring the network sees enough data near the task trajectory of interest.【F:TrainIK.py†L1011-L1097】
4. Normalize positions and radial features, then materialize PyTorch tensors for model training.【F:TrainIK.py†L1099-L1107】

### Training Loop Highlights

- **Loss composition:** Combines Cartesian MSE with angle-space losses, scaling the angular component dynamically to balance gradients.【F:TrainIK.py†L1124-L1160】
- **Stability aids:** Gradient clipping, AMP mixed precision (on CUDA), EMA shadow weights, and learning-rate reductions on plateau keep optimization stable.【F:TrainIK.py†L1109-L1187】
- **Early stopping & mining:** Stops when a blended score saturates and optionally mines hard examples from dense circle and random pools to extend the training set.【F:TrainIK.py†L1189-L1256】
- **Exports:** Saves `ik_model.pt` and `xyz_stats.npz` to the project root once training finishes.【F:TrainIK.py†L1267-L1276】

## Evaluating Trajectories

- **Single circle (`RunCircle.py`):** Generates the canonical focus circle defined by `FOCUS_*` constants, predicts joints for each point, optionally applies DLS refinement (`REFINE_STEPS`), and plots the results.【F:RunCircle.py†L13-L159】
- **Parameter sweep (`SweepCircles.py`):** Iterates over a grid of circle centers, radii, and tilts; records RMSE before/after refinement; and prints the top-performing configurations for rapid diagnostics.【F:SweepCircles.py†L19-L188】

Both scripts rebuild the same feature normalization used during training (normalized XYZ, radial distance, and sine/cosine of azimuth) before invoking the model.【F:RunCircle.py†L24-L53】【F:SweepCircles.py†L40-L75】

## Customization Tips

- **Robot geometry:** Adjust DH parameters or joint limits via the `Robot` constructor if you want to experiment with different kinematic chains.【F:Robot.py†L33-L89】
- **Model capacity:** Modify `HIDDEN`, `N_BLOCKS`, or `DROP_OUT` in `TrainIK.py`, `RunCircle.py`, and `SweepCircles.py` to experiment with wider/deeper networks.【F:TrainIK.py†L70-L88】【F:RunCircle.py†L17-L24】【F:SweepCircles.py†L10-L18】
- **Focus coverage:** Tune `FOCUS_*` constants and band widths to emphasize different task trajectories or workspace regions during training.【F:TrainIK.py†L59-L82】【F:TrainIK.py†L1007-L1067】
- **Refinement depth:** Increase `REFINE_STEPS` or adjust `STEP_CLIP`/`DAMPING_LAMBDA` in the evaluation scripts to trade off accuracy against runtime.【F:RunCircle.py†L25-L47】【F:SweepCircles.py†L23-L46】

## Troubleshooting

- **Model or stats missing:** Ensure `ik_model.pt` and `xyz_stats.npz` are present in the working directory before running inference scripts; training will regenerate them if needed.【F:TrainIK.py†L1267-L1276】【F:RunCircle.py†L42-L70】
- **CUDA out of memory:** Lower `--batch-size`, reduce `--max-samples`, or let the automatic retry logic back off to a smaller batch size.【F:TrainIK.py†L1235-L1270】
- **Slow evaluation:** Disable refinement (`REFINE_STEPS = 0`) or Monte Carlo dropout (`MC_RUNS = 1`, `USE_DROPOUT = False`) in the evaluation scripts for faster but less accurate metrics.【F:RunCircle.py†L21-L44】【F:SweepCircles.py†L17-L36】

---

This README should give you enough context to explore, retrain, and extend the neural IK pipeline for the capstone project. Happy experimenting!