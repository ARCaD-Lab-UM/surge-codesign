#!/usr/bin/env python3
"""
Evaluate design objective on hardware LCM log data.

Loads a hopper_state_t log, constructs the same DesignObjective used in
simulation, and computes per-timestep objective components so they can be
compared side-by-side with simulation results.

Usage:
    python scripts/eval_hardware_log.py --log logs/lcmlog-2026-02-17.00
    python scripts/eval_hardware_log.py --log logs/lcmlog-2026-02-17.00 --save-dir logs/hw_eval
"""

import argparse
import os
import pdb
from pathlib import Path

import lcm
import matplotlib.pyplot as plt
import numpy as np
import torch
from arc_bridge.lcm_msgs import hopper_state_t
from scipy.spatial.transform import Rotation

from mups_codesign.config import CodesignConfig
from mups_codesign.design_objective import DesignObjective
from mups_codesign.design_space import DesignSpace

# ---------------------------------------------------------------------------
# LCM log loading (adapted from extract_and_plot_lcmlog.py)
# ---------------------------------------------------------------------------

def load_lcm_log(log_file_path: str, topic_name: str = "hopper_state"):
    """Load hopper_state_t messages from an LCM log file.

    Returns a dict of numpy arrays keyed by field name.
    """
    log = lcm.EventLog(log_file_path, "r")

    timestamps, positions, velocities = [], [], []
    rpy, omega = [], []
    qj_pos, qj_vel, qj_tau = [], [], []
    foot_forces = []

    for event in log:
        if event.channel != topic_name:
            continue
        msg = hopper_state_t.decode(event.data)
        timestamps.append(event.timestamp / 1e6)
        positions.append(list(msg.position))
        velocities.append(list(msg.velocity))
        rpy.append(list(msg.rpy))
        omega.append(list(msg.omega))
        qj_pos.append(list(msg.qj_pos))
        qj_vel.append(list(msg.qj_vel))
        qj_tau.append(list(msg.qj_tau))
        foot_forces.append(msg.foot_force)

    print(f"Loaded {len(timestamps)} messages from topic '{topic_name}'")
    assert len(timestamps) > 0, f"No messages found on topic '{topic_name}'"

    # Align timestamps to start from 0
    timestamps = np.array(timestamps)
    timestamps = timestamps - timestamps[0]

    return {
        "timestamps": timestamps,
        "positions": np.array(positions),       # (N, 3)
        "velocities": np.array(velocities),     # (N, 3)
        "rpy": np.array(rpy),                   # (N, 3)
        "omega": np.array(omega),               # (N, 3)
        "qj_pos": np.array(qj_pos),             # (N, num_joints)
        "qj_vel": np.array(qj_vel),             # (N, num_joints)
        "qj_tau": np.array(qj_tau),             # (N, num_joints)
        "foot_forces": np.array(foot_forces),   # (N,)
    }


# ---------------------------------------------------------------------------
# Build tensors expected by DesignObjective.calc_objective
# ---------------------------------------------------------------------------

def build_objective_inputs(data: dict, device: str = "cpu", dtype=torch.float32):
    """Convert numpy hardware data into the tensors expected by DesignObjective.

    Returns:
        srb_state:    (N, 13)  — [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        dof_state:    (N, 4)   — [qj_pos(2), qj_vel(2)]
        motor_torque: (N, 2)   — joint torques
    """
    N = len(data["timestamps"])

    # --- quaternion from RPY (scipy uses scalar-last, IsaacGym uses scalar-first) ---
    quats_xyzw = Rotation.from_euler("xyz", data["rpy"]).as_quat()  # (N, 4) x,y,z,w
    quats_wxyz = np.roll(quats_xyzw, 1, axis=1)  # (N, 4) w,x,y,z  (IsaacGym convention)

    srb_state = np.zeros((N, 13), dtype=np.float32)
    srb_state[:, 0:3] = data["positions"]       # x, y, z
    srb_state[:, 3:7] = quats_wxyz              # quaternion
    srb_state[:, 7:10] = data["velocities"]     # linear velocity
    srb_state[:, 10:13] = data["omega"]         # angular velocity

    dof_state = np.zeros((N, 4), dtype=np.float32)
    dof_state[:, 0:2] = data["qj_pos"][:, :2]  # joint positions
    dof_state[:, 2:4] = data["qj_vel"][:, :2]  # joint velocities

    motor_torque = data["qj_tau"][:, :2].astype(np.float32)  # (N, 2)

    to_tensor = lambda x: torch.tensor(x, dtype=dtype, device=device)
    return to_tensor(srb_state), to_tensor(dof_state), to_tensor(motor_torque)


# ---------------------------------------------------------------------------
# Compute per-timestep objective
# ---------------------------------------------------------------------------

def compute_objective_timeseries(
    srb_state: torch.Tensor,
    dof_state: torch.Tensor,
    motor_torque: torch.Tensor,
    config: CodesignConfig,
):
    """Evaluate the design objective at every timestep.

    Returns:
        total_obj:  (N,) tensor — weighted objective per timestep
        components: dict[str, (N,) tensor] — individual objective terms
    """
    # Override num_envs so DesignObjective does not complain about shape
    config.num_envs = srb_state.shape[0]
    obj_calc = DesignObjective(config)

    with torch.no_grad():
        total_obj, components = obj_calc.calc_objective(srb_state, dof_state, motor_torque)

    return total_obj, components


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_objective_timeseries(
    t: np.ndarray,
    total_obj: np.ndarray,
    components: dict,
    srb_state: np.ndarray,
    dof_state: np.ndarray,
    motor_torque: np.ndarray,
    desired_height: float,
    save_path: str = None,
):
    """Create a multi-panel figure of objective terms and relevant states."""

    n_obj = len(components) + 1  # individual terms + total
    n_state = 4                   # height, joint vel, torque, cumulative
    n_plots = n_obj + n_state
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.0 * n_plots), sharex=True)

    idx = 0

    # --- Objective terms ---
    for name, vals in components.items():
        axes[idx].plot(t, vals, linewidth=0.8)
        axes[idx].set_ylabel(name.replace("_", " ").title())
        axes[idx].set_title(f"Objective Component: {name}")
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    # Total weighted objective per timestep
    axes[idx].plot(t, total_obj, linewidth=0.8, color="black")
    axes[idx].set_ylabel("Total Objective")
    axes[idx].set_title("Total Weighted Objective (per timestep)")
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    # --- Cumulative objective ---
    cumulative = np.cumsum(total_obj)
    axes[idx].plot(t, cumulative, linewidth=0.8, color="purple")
    axes[idx].set_ylabel("Cumulative Obj")
    axes[idx].set_title(f"Cumulative Objective (final = {cumulative[-1]:.4f})")
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    # --- Base height ---
    base_height = srb_state[:, 2]
    axes[idx].plot(t, base_height, linewidth=0.8, label="base height")
    axes[idx].axhline(desired_height, color="r", linestyle="--", linewidth=0.8, label=f"desired = {desired_height}")
    axes[idx].set_ylabel("Height (m)")
    axes[idx].set_title("Base Height")
    axes[idx].legend(loc="upper right")
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    # --- Joint torques ---
    axes[idx].plot(t, motor_torque[:, 0], linewidth=0.8, label="Joint 1")
    axes[idx].plot(t, motor_torque[:, 1], linewidth=0.8, label="Joint 2")
    axes[idx].set_ylabel("Torque (Nm)")
    axes[idx].set_title("Motor Torques")
    axes[idx].legend(loc="upper right")
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    # --- Joint velocities ---
    axes[idx].plot(t, dof_state[:, 2], linewidth=0.8, label="Joint 1 vel")
    axes[idx].plot(t, dof_state[:, 3], linewidth=0.8, label="Joint 2 vel")
    axes[idx].set_ylabel("Joint Vel (rad/s)")
    axes[idx].set_xlabel("Time (s)")
    axes[idx].set_title("Joint Velocities")
    axes[idx].legend(loc="upper right")
    axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def print_summary(total_obj_np, components_np, t, config: CodesignConfig, t_range=None):
    """Print a concise numeric summary of the hardware objective evaluation.
    
    Metrics are reported as the average over non-overlapping 100-step windows.
    The data is truncated to an integer multiple of 100 steps.
    """
    WINDOW = 100
    n_steps = len(total_obj_np)

    if n_steps < WINDOW:
        raise ValueError(
            f"Need at least {WINDOW} timesteps (2 s at 50 Hz) for metric computation, "
            f"but only got {n_steps}. Provide a longer time range (>= 2 s)."
        )

    # Truncate to integer multiple of WINDOW
    n_windows = n_steps // WINDOW
    n_used = n_windows * WINDOW
    n_truncated = n_steps - n_used

    total_obj_trunc = total_obj_np[:n_used].reshape(n_windows, WINDOW)
    components_trunc = {k: v[:n_used].reshape(n_windows, WINDOW) for k, v in components_np.items()}

    # Per-window sums (each window = one "episode" of 100 steps)
    window_totals = total_obj_trunc.sum(axis=1)   # (n_windows,)
    window_components = {k: v.sum(axis=1) for k, v in components_trunc.items()}  # each (n_windows,)

    duration = t[-1] - t[0]
    print("\n" + "=" * 70)
    print("Hardware Design Objective Summary")
    print("=" * 70)
    if t_range is not None:
        print(f"  Time range       : [{t_range[0]:.3f}, {t_range[1]:.3f}] s")
    print(f"  Log duration     : {duration:.3f} s  ({n_steps} timesteps)")
    print(f"  Windows          : {n_windows} x {WINDOW} steps = {n_used} steps used"
          f"  ({n_truncated} trailing steps truncated)")
    print(f"  ---")
    print(f"  Total objective  (avg per {WINDOW} steps): {window_totals.mean():.6f}  "
          f"(std={window_totals.std():.6f})")
    print(f"  ---")
    print(f"  Objective weights:")
    for name, weight in config.objective_weights.items():
        print(f"    {name:25s}:  weight={weight}")
    print(f"  ---")
    print(f"  Objective components (avg per {WINDOW} steps):")
    for name, win_vals in window_components.items():
        weight = config.objective_weights.get(name, 0.0)
        print(f"    {name:25s}:  mean={win_vals.mean():.6f}  std={win_vals.std():.6f}  "
              f"weighted_mean={win_vals.mean() * weight:.6f}")
    print(f"  ---")
    print(f"  Per-window breakdown (sum over {WINDOW} steps each):")
    header = f"    {'window':>6s}"
    for name in window_components:
        short = name[:12]
        header += f"  {short:>14s}"
    header += f"  {'total':>14s}"
    print(header)
    for w in range(n_windows):
        row = f"    {w:>6d}"
        for name in window_components:
            row += f"  {window_components[name][w]:>14.6f}"
        row += f"  {window_totals[w]:>14.6f}"
        print(row)
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate design objective on hardware LCM log")
    parser.add_argument("--log", type=str, required=True, help="Path to LCM log file")
    parser.add_argument("--topic", type=str, default="hopper_state", help="LCM topic name")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save plots and results")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--no-plot", action="store_true", help="Skip interactive plot display")
    parser.add_argument("--t-start", type=float, default=None, help="Start of time range (seconds from 0)")
    parser.add_argument("--t-end", type=float, default=None, help="End of time range (seconds from 0)")
    args = parser.parse_args()

    # --- Load hardware log (timestamps already aligned to start from 0) ---
    data = load_lcm_log(args.log, topic_name=args.topic)
    t = data["timestamps"]  # already starts from 0

    # --- Apply time range filter ---
    t_range = None
    if args.t_start is not None or args.t_end is not None:
        t_start = args.t_start if args.t_start is not None else 0.0
        t_end = args.t_end if args.t_end is not None else t[-1]
        t_range = (t_start, t_end)
        mask = (t >= t_start) & (t <= t_end)
        assert mask.any(), f"No data in time range [{t_start}, {t_end}] (log spans [0, {t[-1]:.3f}] s)"
        # Slice all arrays
        t = t[mask]
        for key in data:
            if key == "timestamps":
                data[key] = t
            else:
                data[key] = data[key][mask]
        print(f"Selected time range [{t_start:.3f}, {t_end:.3f}] s  ({mask.sum()} / {len(mask)} timesteps)")

    # --- Downsample to 50 Hz for metric computation ---
    SAMPLE_RATE_HZ = 50.0
    sample_dt = 1.0 / SAMPLE_RATE_HZ
    t = data["timestamps"]
    # Pick indices closest to uniform 50 Hz grid
    t_grid = np.arange(t[0], t[-1], sample_dt)
    ds_indices = np.searchsorted(t, t_grid, side="left").clip(0, len(t) - 1)
    ds_indices = np.unique(ds_indices)  # remove duplicates
    n_before = len(t)
    t = t[ds_indices]
    for key in data:
        if key == "timestamps":
            data[key] = t
        else:
            data[key] = data[key][ds_indices]
    print(f"Downsampled to {SAMPLE_RATE_HZ:.0f} Hz: {n_before} -> {len(t)} timesteps (dt={sample_dt*1000:.1f} ms)")

    # --- Validate minimum duration (need >= 100 steps = 2 s at 50 Hz) ---
    if len(t) < 100:
        effective_dur = len(t) * sample_dt
        raise ValueError(
            f"After downsampling, only {len(t)} steps ({effective_dur:.2f} s). "
            f"Need at least 100 steps (2 s at {SAMPLE_RATE_HZ:.0f} Hz). "
            f"Provide a longer time range."
        )

    # --- Config using default design parameters ---
    config = CodesignConfig(
        num_envs=1,         # will be overridden per timestep count
        device=args.device,
        dt=0.02,       # match the 50 Hz sample rate
        # raw_init_param_values=(4115, 0.138, 0.1, 0.02),  # match sim default design parameters
        hw_height_offset=0.08
    )

    # --- Build tensors ---
    srb_state, dof_state, motor_torque = build_objective_inputs(
        data, device=args.device, dtype=config.dtype
    )

    # --- Compute per-timestep objective ---
    total_obj, components = compute_objective_timeseries(
        srb_state, dof_state, motor_torque, config
    )

    # Move to numpy for plotting
    total_obj_np = total_obj.cpu().numpy()
    components_np = {k: v.cpu().numpy() for k, v in components.items()}
    srb_state_np = srb_state.cpu().numpy()
    dof_state_np = dof_state.cpu().numpy()
    motor_torque_np = motor_torque.cpu().numpy()

    # --- Print summary ---
    print_summary(total_obj_np, components_np, t, config, t_range=t_range)

    # --- Print design parameters used ---
    ds = DesignSpace(config, requires_grad=False)
    print("Design parameters (default):")
    for name, val in zip(ds.param_names, ds.default_param_values.cpu().numpy()):
        print(f"  {name:10s} = {val:.6f}")

    # --- Save / plot ---
    save_path = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "hw_objective_timeseries.png")

        # Also save numeric results
        np.savez_compressed(
            os.path.join(args.save_dir, "hw_objective_data.npz"),
            time=t,
            total_objective=total_obj_np,
            **{f"component_{k}": v for k, v in components_np.items()},
            srb_state=srb_state_np,
            dof_state=dof_state_np,
            motor_torque=motor_torque_np,
        )
        print(f"Numeric data saved to {args.save_dir}/hw_objective_data.npz")

    if not args.no_plot:
        obj_calc = DesignObjective(config)
        plot_objective_timeseries(
            t, total_obj_np, components_np,
            srb_state_np, dof_state_np, motor_torque_np,
            desired_height=obj_calc.desired_base_height,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
