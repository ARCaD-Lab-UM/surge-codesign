import torch

from mups_codesign.design_space import DesignSpace
from mups_codesign.mups_spring import MupsSpring


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)

    device = torch.device("cpu")
    design_space = DesignSpace(active_dim=2, device=device)
    mups_spring = MupsSpring(num_envs=1, device=device)

    active_param_names = design_space.get_active_param_names()
    param_values = design_space.default_params[:design_space.active_dim].unsqueeze(0)
    param_values *= 2.0  # Scale up for testing

    mups_spring.set_ups_params_from_design(active_param_names, param_values)
    print("Set UPS spring parameters from design:")
    for name in active_param_names:
        value = mups_spring.ups_param_dict[name][0].item()
        print(f"  {name}: {value:.4f}")