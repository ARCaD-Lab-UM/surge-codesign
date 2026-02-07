# Differential CoDesign

Differentiable co-design pipeline for the MUPS hopping robot.

## Installation

This repo assumes Isaac Gym and legged_gym are pre-installed. Install this package locally:
```bash
pip install -e .
```

## Quick start

Run unit tests:

```bash
pytest -v -s
```

Run design optimization:

```bash
python scripts/run_codesign.py --headless
```

Collect design landscape for a policy:
```bash
python scripts/collect_landscape.py --headless
```

Plot optimization trajectory over design landscape:
```bash
python scripts/plot_landscape.py --policy_id rainbow_v6
```

### Prioritized TODOs:

- [x] Fix hardcoded design_param_names in mups_robot.py
- [x] Unify changeable parameters with design space parameters
- [x] Fix broken tests and make them unit-testable
- [x] Fix energy calculation to account for time step correctly
- [x] Wrap control loop into a rollout function
- [x] Unify control loop helper for run_codesign and plot_landscape
- [x] Run a test with 4 dim design space with unchanged policy to see if pipeline works
- [x] Implement logger to save important statistics during optimization
- [x] Revamp plot_landscape script to dump one landscape per policy
- [x] Check if landscape match with previous runs
- [x] Retry hacked 4 dim optimization and check tensorboard to see if those make sense
- [x] No need for hopper standalone env and config, instead, use the actual hopper env and config
- [x] We need to plot the episode trajectory of hopper
- [x] Retrain policy with 4 dim design space, tune design range if necessary
- [x] Test NN as an individual block
- [x] What's the role of num_envs in design optimization? Only matter if any domain rand is on.
- [x] Make config actually useful
- [x] Disable awkward printing from isaacgym
- [x] Add unit tests for FD vs AD
- [x] Visualize gradient field over landscape
- [ ] ~~Compare FD error with AD over the landscape~~ FD through isaac env gives super noise gradients
- [ ] Start interfacing with CMA-ES
