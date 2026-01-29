import torch

from mups_codesign.config import CodesignConfig
from mups_codesign.mups_spring import MupsSpring
from mups_codesign.design_space import DesignSpace


def test_spring_torque_requires_grad():
    torch.manual_seed(0)

    cfg = CodesignConfig(num_envs=1, device="cpu", dtype=torch.float32, active_param_names=("ups_ks", "ups_l0"))
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
