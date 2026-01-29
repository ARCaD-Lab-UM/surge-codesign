import pdb
import torch

from mups_codesign.config import CodesignConfig
from mups_codesign.design_space import DesignSpace


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)

    config = CodesignConfig(
        active_dim=2,
        device="cpu",
        dtype=torch.float32
    )
    design_space = DesignSpace(config)

    active_param_names = design_space.get_active_param_names()
    active_param_bounds = design_space.get_active_param_bounds()
    active_param_scales = design_space.get_active_param_scales()

    print("Design parameter names:", design_space.param_names)
    print("Default parameters:", design_space.default_params)
    print("Parameter bounds:\n", design_space.param_bounds)
    print("Parameter scales:", design_space.param_scales)

    print("\nActive design parameters:")
    for i in range(design_space.active_dim):
        name = active_param_names[i]
        bounds = active_param_bounds[i]
        scale = active_param_scales[i]
        print(f"  {name}: bounds = [{bounds[0]:.4f}, {bounds[1]:.4f}], scale = {scale:.4f}")
