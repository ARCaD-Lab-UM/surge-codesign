import matplotlib.pyplot as plt
import numpy as np
from torchviz import make_dot

# Global setting for matplotlib
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.dpi"] = 150


def plot_contour(
    param1_grid,
    param2_grid,
    objective_grid,
    grid_param_names,
    trajectory=None,
    save_path=None,
    show=False,
):

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot design landscape
    contour = ax.contourf(param1_grid, param2_grid, objective_grid, levels=20, cmap="jet")
    fig.colorbar(contour, ax=ax, label="Design Objective")
    ax.set_xlabel(grid_param_names[0])
    ax.set_ylabel(grid_param_names[1])

    # Overlay the optimization trajectory if provided
    if trajectory is not None and trajectory.shape[0] > 0:
        n_points = trajectory.shape[0]
        alpha_values = np.linspace(0.7, 1, n_points)
        colors = np.zeros((n_points, 4))
        colors[:, 3] = alpha_values  # RGB stays at 0 (black), only alpha varies

        ax.scatter(trajectory[:, 0], trajectory[:, 1], color=colors, s=90, marker="^", label="Optimization Path")
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color="cyan", s=100, marker="s", edgecolor="black", label="Start")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color="magenta", s=150, marker="*", edgecolor="black", label="Final Best")
        ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)

    if show:
        plt.show()

    plt.close(fig)


def plot_surface(
    param1_grid,
    param2_grid,
    objective_grid,
    grid_param_names,
    trajectory=None,
    save_path=None,
    show=False,
):

    def _nearest_objective_values(points):
        if points.size == 0:
            return np.array([])
        param1_span = param1_grid[0, :]
        param2_span = param2_grid[:, 0]
        idx1 = np.abs(param1_span[None, :] - points[:, 0][:, None]).argmin(axis=1)
        idx2 = np.abs(param2_span[None, :] - points[:, 1][:, None]).argmin(axis=1)
        return objective_grid[idx2, idx1]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(param1_grid, param2_grid, objective_grid, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel(grid_param_names[0])
    ax.set_ylabel(grid_param_names[1])
    ax.set_zlabel("Design Objective")

    has_legend = False
    if trajectory is not None and trajectory.shape[0] > 0:
        z_vals = _nearest_objective_values(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], z_vals, color="black", linewidth=1.5, label="Optimization Path")
        ax.scatter(trajectory[0, 0], trajectory[0, 1], z_vals[0], color="cyan", edgecolor="black", s=60, marker="s", label="Start")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], z_vals[-1], color="magenta", edgecolor="black", s=80, marker="*", label="Final Best")
        has_legend = True
    if has_legend:
        ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)

def plot_gradient_vector_field(
    param1_grid,
    param2_grid,
    grad1_grid,
    grad2_grid,
    objective_grid,
    grid_param_names,
    grad_magnitude=1,
    save_path=None,
    show=False,
):
    """Plot gradient vector field overlaid on objective contour."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot objective contour as background
    contour = ax.contourf(
        param1_grid, param2_grid, objective_grid,
        levels=20, cmap="viridis", alpha=0.7
    )
    fig.colorbar(contour, ax=ax, label="Design Objective")
    
    # Use negative gradients to show descent direction
    grad1_normalized = -grad1_grid / grad_magnitude
    grad2_normalized = -grad2_grid / grad_magnitude

    # Plot gradient vector field (quiver)
    # Subsample for cleaner visualization if grid is dense
    step = max(1, param1_grid.shape[0] // 16)
    ax.quiver(
        param1_grid[::step, ::step],
        param2_grid[::step, ::step],
        grad1_normalized[::step, ::step],
        grad2_normalized[::step, ::step],
        color="white",
        alpha=0.9,
        scale=25,
        width=0.004,
        headwidth=4,
        headlength=5,
    )

    ax.set_xlabel(grid_param_names[0])
    ax.set_ylabel(grid_param_names[1])
    ax.set_title("Gradient Vector Field (Descent Direction)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved gradient vector field to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def save_ad_graph(loss, vars:dict, filename="autograd_graph"):
    dot = make_dot(loss, params=vars)
    dot.format = "png"
    if "DetachBackward" in dot.source:
        print("[WARNING] Graph contains detached tensors!")

    dot.render(filename, cleanup=True)
