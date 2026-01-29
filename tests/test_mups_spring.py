import torch

import matplotlib.pyplot as plt

from mups_codesign.config import CodesignConfig
from mups_codesign.mups_spring import MupsSpring
from mups_codesign.design_space import DesignSpace


def _assert_requires_grad(tag, spring_torque):
    assert spring_torque.requires_grad, f"{tag} spring torque should require grad"


def _test_gradient_preserving(cfg):
    device = torch.device(cfg.device)
    mups_spring = MupsSpring(cfg)
    design_space = DesignSpace(cfg)
    active_param_names = design_space.get_active_param_names()

    random_params = torch.randn((1, design_space.active_dim), device=device, requires_grad=True)
    mups_spring.set_ups_params_from_design(active_param_names, random_params, print_info=True)

    dof_pos = torch.randn((1, 2), device=device)
    spring_torque = mups_spring.calc_spring_torque(dof_pos)
    _assert_requires_grad("Random input", spring_torque)


def _plot_bounds_from_design_space(cfg):
    device = torch.device(cfg.device)
    design_space = DesignSpace(cfg)
    mups_spring = MupsSpring(cfg)

    active_param_names = design_space.get_active_param_names()
    param_values = design_space.default_params[:design_space.active_dim].unsqueeze(0)

    print("Default UPS spring parameters from design:")
    for i, name in enumerate(active_param_names):
        value = param_values[0, i].item()
        print(f"  {name}: {value:.4f}")

    param_bounds = design_space.get_active_param_bounds()

    # Generate knee position span
    knee_pos_limits = [-2.7, -0.4] # Radians
    num_points = 100
    knee_pos_span = torch.linspace(
        knee_pos_limits[0],
        knee_pos_limits[1],
        steps=num_points,
        device=device
    )

    # Plot spring torque curves for lower and upper bounds of design space
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Lower bound
    print("Setting UPS spring parameters to lower bound of design space:")
    mups_spring.set_ups_params_from_design(
        active_param_names,
        param_bounds[:, 0].unsqueeze(0),
        print_info=True
    )
    for knee_pos in knee_pos_span:
        dof_pos = torch.tensor([[0.0, knee_pos]], device=device)
        spring_torque = mups_spring.calc_spring_torque(dof_pos)
        axes[0].scatter(knee_pos.item(), spring_torque[0, 1].item(), color="blue")

    axes[0].set_title("Spring Torque at Lower Bound of Design Space")
    axes[0].set_xlabel("Knee Position (rad)")
    axes[0].set_ylabel("Spring Torque (Nm)")
    axes[0].grid(True)

    # Upper bound
    print("Setting UPS spring parameters to upper bound of design space:")
    mups_spring.set_ups_params_from_design(
        active_param_names,
        param_bounds[:, 1].unsqueeze(0),
        print_info=True
    )
    for knee_pos in knee_pos_span:
        dof_pos = torch.tensor([[0.0, knee_pos]], device=device)
        spring_torque = mups_spring.calc_spring_torque(dof_pos)
        axes[1].scatter(knee_pos.item(), spring_torque[0, 1].item(), color="red")

    axes[1].set_title("Spring Torque at Upper Bound of Design Space")
    axes[1].set_xlabel("Knee Position (rad)")
    axes[1].set_ylabel("Spring Torque (Nm)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    torch.manual_seed(0)

    cfg = CodesignConfig(num_envs=1, device="cpu", active_dim=2)

    _test_gradient_preserving(cfg)
    _plot_bounds_from_design_space(cfg)
