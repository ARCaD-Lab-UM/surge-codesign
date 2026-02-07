"""
Load a gradient field .npz file and plot the gradient vector field.

Usage:
    python scripts/plot_gradient_field.py <path_to_npz_file>
    python scripts/plot_gradient_field.py  # uses most recent file in logs/landscapes/
"""

import argparse
import os
import sys

import numpy as np

from mups_codesign.vis_helper import plot_gradient_vector_field


def find_latest_gradient_field(gradient_field_dir="logs/gradient_fields"):
    """Find the most recent gradient field .npz file."""
    if not os.path.exists(gradient_field_dir):
        return None
    
    npz_files = [
        f for f in os.listdir(gradient_field_dir)
        if f.endswith(".npz") and "gradient_field" in f
    ]
    
    if not npz_files:
        return None
    
    # Sort by modification time, newest first
    npz_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(gradient_field_dir, f)),
        reverse=True
    )
    
    return os.path.join(gradient_field_dir, npz_files[0])


def main():
    parser = argparse.ArgumentParser(description="Plot gradient vector field from .npz file")
    parser.add_argument(
        "npz_file",
        nargs="?",
        default=None,
        help="Path to the gradient field .npz file. If not provided, uses the most recent file."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot interactively"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for the plot. Defaults to replacing .npz with .png"
    )
    parser.add_argument(
        "--grad-magnitude",
        type=float,
        default=1.0,
        help="Scaling factor for gradient vectors in the plot")
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Subsampling step for quiver plot (e.g., 2 means every other point)"
    )
    args = parser.parse_args()

    # Find the npz file
    npz_path = args.npz_file
    if npz_path is None:
        npz_path = find_latest_gradient_field()
        if npz_path is None:
            print("Error: No gradient field .npz file found in logs/gradient_fields/")
            print("Please provide a path to an .npz file or run collect_gradient_field.py first.")
            sys.exit(1)
        print(f"Using most recent file: {npz_path}")

    if not os.path.exists(npz_path):
        print(f"Error: File not found: {npz_path}")
        sys.exit(1)

    # Load the data
    print(f"Loading: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    param1_grid = data["param1_grid"]
    param2_grid = data["param2_grid"]
    objective_grid = data["objective_grid"]
    grad1_grid = data["grad1_grid"]
    grad2_grid = data["grad2_grid"]
    grid_param_names = tuple(data["grid_param_names"])

    # Print summary
    print(f"Grid shape: {param1_grid.shape}")
    print(f"Parameters: {grid_param_names}")
    print(f"Objective range: [{objective_grid.min():.4f}, {objective_grid.max():.4f}]")
    print(f"Grad1 range: [{grad1_grid.min():.6f}, {grad1_grid.max():.6f}]")
    print(f"Grad2 range: [{grad2_grid.min():.6f}, {grad2_grid.max():.6f}]")

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = npz_path.replace(".npz", ".png")

    # Plot
    plot_gradient_vector_field(
        param1_grid,
        param2_grid,
        grad1_grid,
        grad2_grid,
        objective_grid,
        grid_param_names,
        save_path=output_path,
        show=not args.no_show,
        grad_magnitude=args.grad_magnitude,
        step=args.step,
    )


if __name__ == "__main__":
    main()
