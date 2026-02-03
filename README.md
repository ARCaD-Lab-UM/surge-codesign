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
pytest -v
```

Run design optimization:

```bash
python scripts/run_codesign.py --task hopper --headless --load_run rainbow_v1
```

Collect design landscape for a policy:
```bash
python scripts/collect_landscape.py --task hopper --headless --load_run rainbow_v1
```

Plot optimization trajectory over design landscape:
```bash
python scripts/plot_landscape.py --task hopper --policy_id rainbow_v1
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
- [ ] Train with design parameters directly in actor input
- [ ] How to visualize the true landscape we are moving on top of?
- [ ] How to run with different seeds?
- [ ] Start interfacing with CMA-ES
