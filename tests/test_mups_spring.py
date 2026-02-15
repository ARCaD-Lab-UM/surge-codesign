import pytest
import torch

from mups_codesign.config import CodesignConfig
from mups_codesign.design_space import DesignSpace
from mups_codesign.mups_spring import MupsSpring


def test_spring_torque_requires_grad():
    torch.manual_seed(0)

    cfg = CodesignConfig(num_envs=1, device="cpu", dtype=torch.float32, active_param_names=("ups_ks", "ups_l0"), raw_init_param_values=None)
    mups_spring = MupsSpring(cfg)
    design_space = DesignSpace(cfg)
    active_param_names = design_space.active_param_names

    param_values = torch.randn((cfg.num_envs, len(active_param_names)), device=cfg.device, requires_grad=True)
    mups_spring.update_design_param_dict(active_param_names, param_values, print_info=False)

    dof_pos = torch.randn((cfg.num_envs, 2), device=cfg.device)
    spring_torque = mups_spring.calc_spring_torque(dof_pos)

    assert spring_torque.shape == (cfg.num_envs, 2)
    assert spring_torque.requires_grad, "spring torque should require grad"
    assert torch.isfinite(spring_torque).all(), "spring torque should be finite"

@pytest.mark.parametrize("seed", [0, 42, 123])
def test_spring_torque_autograd_matches_finite_diff(seed):
    """Verify autograd gradients match finite difference gradients within threshold."""
    torch.manual_seed(seed)

    cfg = CodesignConfig(num_envs=1, device="cpu", dtype=torch.float64, active_param_names=("ups_ks", "ups_l0"), raw_init_param_values=None)
    design_space = DesignSpace(cfg)
    active_param_names = design_space.active_param_names
    num_params = len(active_param_names)

    # Fixed dof_pos for consistency
    dof_pos = torch.randn((cfg.num_envs, 2), device=cfg.device, dtype=cfg.dtype)

    # Test parameters
    param_values_base = design_space.sample_active_param_values().unsqueeze(0)

    # --- Compute autograd gradient ---
    param_values = param_values_base.clone().requires_grad_(True)
    mups_spring = MupsSpring(cfg)
    mups_spring.update_design_param_dict(active_param_names, param_values, print_info=False)
    spring_torque = mups_spring.calc_spring_torque(dof_pos)
    loss = spring_torque.sum()
    loss.backward()
    autograd_grad = param_values.grad.clone()

    # --- Compute finite difference gradient ---
    eps = 1e-6
    fd_grad = torch.zeros_like(param_values_base)

    for i in range(num_params):
        # Forward perturbation
        param_plus = param_values_base.clone()
        param_plus[0, i] += eps
        mups_spring_plus = MupsSpring(cfg)
        mups_spring_plus.update_design_param_dict(active_param_names, param_plus, print_info=False)
        torque_plus = mups_spring_plus.calc_spring_torque(dof_pos)
        loss_plus = torque_plus.sum()

        # Backward perturbation
        param_minus = param_values_base.clone()
        param_minus[0, i] -= eps
        mups_spring_minus = MupsSpring(cfg)
        mups_spring_minus.update_design_param_dict(active_param_names, param_minus, print_info=False)
        torque_minus = mups_spring_minus.calc_spring_torque(dof_pos)
        loss_minus = torque_minus.sum()

        # Central difference
        fd_grad[0, i] = (loss_plus - loss_minus) / (2 * eps)

    # --- Compare gradients ---
    abs_diff = (autograd_grad - fd_grad).abs()
    rel_diff = abs_diff / (fd_grad.abs() + 1e-8)

    rtol = 1e-4
    atol = 1e-6

    # Print gradient comparison info (visible with pytest -s)
    print(f"\nGradient Comparison:")
    print(f"  Autograd grad:\n{autograd_grad}")
    print(f"  Finite diff grad:\n{fd_grad}")
    print(f"  Abs diff (max): {abs_diff.max().item():.2e}")
    print(f"  Rel diff (max): {rel_diff.max().item():.2e}")

    assert torch.allclose(autograd_grad, fd_grad, rtol=rtol, atol=atol), (
        f"Autograd gradient does not match finite difference.\n"
        f"Autograd: {autograd_grad}\n"
        f"Finite diff: {fd_grad}\n"
        f"Abs diff: {abs_diff}\n"
        f"Rel diff: {rel_diff}"
    )
