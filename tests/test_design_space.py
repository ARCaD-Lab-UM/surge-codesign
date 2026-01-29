import torch

from mups_codesign.config import CodesignConfig
from mups_codesign.design_space import DesignSpace


def _make_config(active_param_names=("ups_ks", "ups_l0")):
    return CodesignConfig(
        active_param_names=active_param_names,
        device="cpu",
        dtype=torch.float32,
    )


def test_active_param_indices_match_names():
    config = _make_config(active_param_names=("ups_l0", "ups_l4"))
    design_space = DesignSpace(config)

    expected_indices = [design_space.param_names.index(name) for name in config.active_param_names]
    assert design_space.active_param_indices == expected_indices


def test_active_param_values_default_match():
    config = _make_config(active_param_names=("ups_ks", "ups_l2"))
    design_space = DesignSpace(config)

    expected_values = design_space.default_param_values[design_space.active_param_indices]
    assert torch.allclose(design_space.active_param_values, expected_values)


def test_project_active_params_into_bounds():
    config = _make_config(active_param_names=("ups_ks", "ups_l0"))
    design_space = DesignSpace(config)

    with torch.no_grad():
        design_space.active_normalized_param_values[:] = design_space.active_normalized_param_bounds[:, 1] + 1.0

    design_space.project_active_params_into_bounds()

    assert torch.all(design_space.active_normalized_param_values <= design_space.active_normalized_param_bounds[:, 1])
    assert torch.all(design_space.active_normalized_param_values >= design_space.active_normalized_param_bounds[:, 0])
