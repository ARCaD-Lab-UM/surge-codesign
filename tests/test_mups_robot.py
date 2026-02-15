import pytest
import torch

from mups_codesign.config import CodesignConfig
from mups_codesign.design_objective import DesignObjective
from mups_codesign.design_space import DesignSpace
from mups_codesign.mups_robot import MupsRobot


def compute_fd_grad(loss_fn, params, eps=1e-6):
    """Compute finite difference gradient using central differences."""
    grad = torch.zeros_like(params)
    for i in range(params.numel()):
        p_plus = params.clone()
        p_plus.view(-1)[i] += eps
        p_minus = params.clone()
        p_minus.view(-1)[i] -= eps
        grad.view(-1)[i] = (loss_fn(p_plus) - loss_fn(p_minus)) / (2 * eps)
    return grad


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("num_steps", [1, 5])
def test_srb_gradient_matches_fd(seed, num_steps):
    """Test autograd gradient through SRB dynamics matches finite difference."""
    torch.manual_seed(seed)

    objective_weights = {
        "heating_energy": 0.0,
        "mechanical_energy": 0.0,
        "height_tracking_error": 1.0,
    }

    cfg = CodesignConfig(
        num_envs=1,
        device="cpu",
        dtype=torch.float64,
        active_param_names=("ups_ks", "ups_l0"),
        raw_init_param_values=None,
        objective_weights=objective_weights,
    )

    design_space = DesignSpace(cfg, requires_grad=True)
    param_names = design_space.active_param_names
    param_base = design_space.sample_active_param_values()

    # Initial state: foot in contact (low height)
    root_state_init = torch.zeros((cfg.num_envs, 13), dtype=cfg.dtype)
    root_state_init[:, 2] = 0.25
    root_state_init[:, 6] = 1.0

    dof_state = torch.zeros((cfg.num_envs, 4), dtype=cfg.dtype)
    dof_state[:, 0] = 0.9
    dof_state[:, 1] = -1.8

    actions = [torch.randn((cfg.num_envs, 2), dtype=cfg.dtype) for _ in range(num_steps)]

    def compute_loss(params):
        robot = MupsRobot(cfg)
        robot.set_design_params(param_names, params.unsqueeze(0))
        obj_calc = DesignObjective(cfg)

        total_obj = torch.zeros(1, dtype=cfg.dtype)
        root_state = root_state_init.clone()

        for t in range(num_steps):
            srb_state, motor_torque, _ = robot.step_srb_dynamics(
                root_state, dof_state.clone(), actions[t].clone()
            )
            obj, _ = obj_calc.calc_objective(srb_state, dof_state, motor_torque)
            total_obj = total_obj + obj.sum()
            root_state = srb_state

        return total_obj

    # Autograd
    params = param_base.clone().requires_grad_(True)
    loss = compute_loss(params)
    loss.backward()
    ad_grad = params.grad.clone()

    # Finite difference
    fd_grad = compute_fd_grad(compute_loss, param_base)

    # Print for debugging (pytest -s)
    print(f"\n[seed={seed}, steps={num_steps}] loss={loss.item():.6f}")
    print(f"  params: {param_base.tolist()}")
    print(f"  autograd: {ad_grad.tolist()}")
    print(f"  fd_grad:  {fd_grad.tolist()}")
    print(f"  diff:     {(ad_grad - fd_grad).abs().tolist()}")

    assert torch.allclose(ad_grad, fd_grad, rtol=1e-4, atol=1e-6), \
        f"Gradient mismatch: autograd={ad_grad}, fd={fd_grad}"

