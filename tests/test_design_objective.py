import pytest
import torch

from mups_codesign.config import CodesignConfig
from mups_codesign.design_objective import DesignObjective


@pytest.mark.parametrize("use_log1p", [True, False])
def test_calc_objective_requires_grad(use_log1p):
    torch.manual_seed(0)

    cfg = CodesignConfig(num_envs=4, device="cpu", dtype=torch.float32, raw_init_param_values=None, use_log1p=use_log1p)
    obj = DesignObjective(cfg)

    srb_state = torch.randn((cfg.num_envs, 13), device=cfg.device, dtype=cfg.dtype, requires_grad=True)
    dof_state = torch.randn((cfg.num_envs, 6), device=cfg.device, dtype=cfg.dtype, requires_grad=True)
    motor_torque = torch.randn((cfg.num_envs, 2), device=cfg.device, dtype=cfg.dtype, requires_grad=True)

    total, components = obj.calc_objective(srb_state, dof_state, motor_torque)

    assert total.shape == (cfg.num_envs,)
    for name, value in components.items():
        assert value.shape == (cfg.num_envs,), f"{name} shape mismatch"

    assert total.requires_grad, "design objective should require grad"
    assert all(value.requires_grad for value in components.values()), "all components should require grad"

    total.mean().backward()
