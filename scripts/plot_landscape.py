import glob
import os
import argparse

import numpy as np

from mups_codesign.data_logger import load_run
from mups_codesign.vis_helper import plot_contour, plot_surface


def _find_latest_landscape(landscape_dir, task, policy_tag):
    pattern = os.path.join(landscape_dir, f"{task}_{policy_tag}_landscape_*.npz")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def _load_landscape(path):
    data = np.load(path, allow_pickle=True)
    param1_grid = data["param1_grid"]
    param2_grid = data["param2_grid"]
    objective_grid = data["objective_grid"]
    param_names = None
    if "param_names" in data:
        param_names = [str(name) for name in data["param_names"].tolist()]
    grid_param_names = None
    if "grid_param_names" in data:
        grid_param_names = [str(name) for name in data["grid_param_names"].tolist()]
    policy_id = None
    if "policy_id" in data:
        policy_id = data["policy_id"]

    if not param_names:
        param_names = ["param_0", "param_1"]
    if not grid_param_names:
        grid_param_names = param_names[:2]

    return {
        "param1_grid": param1_grid,
        "param2_grid": param2_grid,
        "objective_grid": objective_grid,
        "param_names": param_names,
        "grid_param_names": grid_param_names,
        "policy_id": policy_id,
    }


def _find_latest_run_dir(log_root, run_prefix):
    candidates = []
    for path in glob.glob(os.path.join(log_root, f"{run_prefix}_*")):
        if os.path.isdir(path):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getctime)


def _load_optimization_trajectory(log_root, task, param_names):
    run_prefix = f"{task}_codesign"
    run_dir = _find_latest_run_dir(log_root, run_prefix)
    if not run_dir:
        return None, None
    records = load_run(run_dir, stream="iteration")
    trajectory = []
    for record in records:
        if "param/value/vector" in record:
            values = record["param/value/vector"]
        else:
            values = [record.get(f"param/value/{name}") for name in param_names]
        if values is None:
            continue
        if any(v is None for v in values):
            continue
        trajectory.append(values)
    if not trajectory:
        return None, run_dir
    return np.asarray(trajectory, dtype=float), run_dir


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot design landscape from saved NPZ.")
    parser.add_argument("--task", type=str, default="hopper", help="Task name used in landscape filenames.")
    parser.add_argument("--policy_id", type=str, help="Policy identifier to match landscape files.")
    parser.add_argument("--log_root", type=str, default="logs", help="Root directory for logs and landscapes.")
    parser.add_argument("--landscape_path", type=str, default=None, help="Explicit NPZ landscape path to plot.")
    parser.add_argument("--no_overlay", action="store_true", help="Disable optimization trajectory overlay.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    policy_id = args.policy_id

    landscape_path = args.landscape_path
    landscape_dir = os.path.join(args.log_root, "landscapes")
    if landscape_path is None:
        landscape_path = _find_latest_landscape(landscape_dir, args.task, policy_id)

    if landscape_path is None:
        raise FileNotFoundError(f"No landscape files found in {landscape_dir}")

    landscape = _load_landscape(landscape_path)
    param1_grid = landscape["param1_grid"]
    param2_grid = landscape["param2_grid"]
    objective_grid = landscape["objective_grid"]
    grid_param_names = landscape["grid_param_names"]
    landscape_policy_id = landscape["policy_id"]

    trajectory = None
    run_dir = None
    if not args.no_overlay:
        trajectory, run_dir = _load_optimization_trajectory(args.log_root, args.task, landscape["param_names"])
        if trajectory is not None and trajectory.shape[1] > 2:
            trajectory = trajectory[:, :2]

    output_dir = os.path.dirname(landscape_path)
    overlay_enabled = not args.no_overlay
    if overlay_enabled and run_dir:
        output_dir = run_dir
    basename = os.path.splitext(os.path.basename(landscape_path))[0]
    suffix = "_overlay" if overlay_enabled and run_dir else ""

    contour_path = os.path.join(output_dir, f"{basename}_contour{suffix}.png")
    plot_contour(
        param1_grid,
        param2_grid,
        objective_grid,
        grid_param_names,
        trajectory=trajectory,
        save_path=contour_path,
        show=True,
    )
    print(f"Saved contour plot to: {contour_path}")

    surface_path = os.path.join(output_dir, f"{basename}_surface{suffix}.png")
    plot_surface(
        param1_grid,
        param2_grid,
        objective_grid,
        grid_param_names,
        trajectory=trajectory,
        save_path=surface_path,
        show=False,
    )
    print(f"Saved surface plot to: {surface_path}")

    if run_dir:
        print(f"Overlayed optimization trajectory from: {run_dir}")
    elif overlay_enabled:
        print("No optimization logs found for overlay")
