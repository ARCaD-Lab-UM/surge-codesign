"""
Plot MUPS spring torque curves at the lower/upper bounds of the design space.
"""

import torch
import matplotlib.pyplot as plt

from mups_codesign.config import CodesignConfig
from mups_codesign.design_space import DesignSpace
from mups_codesign.mups_spring import MupsSpring
import mups_codesign.vis_helper # import global matplotlib settings


def plot_spring_bounds(cfg: CodesignConfig, knee_pos_limits=(-2.7, -0.4), num_points=100):
    device = torch.device(cfg.device)
    design_space = DesignSpace(cfg)
    mups_spring = MupsSpring(cfg)

    active_param_names = design_space.active_param_names
    param_bounds = design_space.active_param_bounds

    print("Active UPS spring parameters:")
    for i, name in enumerate(active_param_names):
        print(f"  {name}")

    knee_pos_span = torch.linspace(
        knee_pos_limits[0],
        knee_pos_limits[1],
        steps=num_points,
        device=device,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Lower bound
    print("Setting UPS spring parameters to lower bound of design space:")
    mups_spring.update_design_param_dict(
        active_param_names,
        param_bounds[:, 0].unsqueeze(0),
        print_info=True,
    )
    for knee_pos in knee_pos_span:
        dof_pos = torch.tensor([[0.0, knee_pos]], device=device)
        spring_torque = mups_spring.calc_spring_torque(dof_pos)
        axes[0].scatter(knee_pos.item(), spring_torque[0, 1].item(), color="blue")

    axes[0].set_title("Spring Torque at Lower Bound")
    axes[0].set_xlabel("Knee Position (rad)")
    axes[0].set_ylabel("Spring Torque (Nm)")
    axes[0].axhline(0.0, color="black", linestyle="--")
    axes[0].axvline(knee_pos_limits[0], color="black", linestyle="--")
    axes[0].axvline(knee_pos_limits[1], color="black", linestyle="--")
    axes[0].grid(True)

    # Upper bound
    print("Setting UPS spring parameters to upper bound of design space:")
    mups_spring.update_design_param_dict(
        active_param_names,
        param_bounds[:, 1].unsqueeze(0),
        print_info=True,
    )
    for knee_pos in knee_pos_span:
        dof_pos = torch.tensor([[0.0, knee_pos]], device=device)
        spring_torque = mups_spring.calc_spring_torque(dof_pos)
        axes[1].scatter(knee_pos.item(), spring_torque[0, 1].item(), color="red")

    axes[1].set_title("Spring Torque at Upper Bound")
    axes[1].set_xlabel("Knee Position (rad)")
    axes[1].set_ylabel("Spring Torque (Nm)")
    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[1].axvline(knee_pos_limits[0], color="black", linestyle="--")
    axes[1].axvline(knee_pos_limits[1], color="black", linestyle="--")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_nominal_spring_curve(cfg: CodesignConfig, knee_pos_limits=(-2.7, -0.4), num_points=100):
    device = torch.device(cfg.device)
    mups_spring = MupsSpring(cfg)

    knee_pos_span = torch.linspace(
        knee_pos_limits[0],
        knee_pos_limits[1],
        steps=num_points,
        device=device,
    )

    spring_torque_list = []
    for knee_pos in knee_pos_span:
        dof_pos = torch.tensor([[0.0, knee_pos]], device=device)
        spring_torque = mups_spring.calc_spring_torque(dof_pos)
        spring_torque_list.append(spring_torque[0, 1].item())

    plt.figure(figsize=(5, 4), dpi=200)
    plt.plot(knee_pos_span.cpu(), spring_torque_list, linewidth=2, color="black")
    plt.xlabel("Knee Position (rad)")
    plt.ylabel("Spring Torque (Nm)")
    plt.axhline(0.0, color="black", linestyle="--")
    plt.axvline(knee_pos_limits[0], color="black", linestyle="--")
    plt.axvline(knee_pos_limits[1], color="black", linestyle="--")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    torch.manual_seed(0)

    param_names = ("ups_ks", "ups_l0", "ups_l2", "ups_l4")
    cfg = CodesignConfig(num_envs=1, device="cpu", active_param_names=param_names)
    # plot_spring_bounds(cfg)
    plot_nominal_spring_curve(cfg)
