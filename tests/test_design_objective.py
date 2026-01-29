import torch

from mups_codesign.design_objective import DesignObjective


def _print_components(tag, total, components):
    print(f"\n{tag}")
    print("  total objective:", total)
    for name, value in components.items():
        print(f"  {name}:", value)


def _assert_requires_grad(total, components):
    assert total.requires_grad, "design objective should require grad"
    for name, value in components.items():
        assert value.requires_grad, f"{name} should require grad"


def _run_case(use_log1p):
    torch.set_printoptions(precision=4, sci_mode=False)
    torch.manual_seed(0)

    device = torch.device("cpu")
    dtype = torch.float32
    num_envs = 4

    obj = DesignObjective(num_envs=num_envs, device=device, dtype=dtype)
    obj.use_log1p = use_log1p

    # Fake states with gradients enabled
    srb_state = torch.randn((num_envs, 13), device=device, dtype=dtype, requires_grad=True)
    dof_state = torch.randn((num_envs, 6), device=device, dtype=dtype, requires_grad=True)
    motor_torque = 10 * torch.randn((num_envs, 2), device=device, dtype=dtype, requires_grad=True)

    total, components = obj.calc_design_objective(srb_state, dof_state, motor_torque)

    tag = f"Design objective (use_log1p={use_log1p})"
    _print_components(tag, total, components)
    _assert_requires_grad(total, components)


if __name__ == "__main__":
    _run_case(use_log1p=True)
    _run_case(use_log1p=False)
