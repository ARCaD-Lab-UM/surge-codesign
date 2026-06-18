# SurGE: Surrogate Gradient-guided Evolution for Co-design of Legged Robots with Parallel Elasticity

<img src="docs/diff_codesign.png" width=800/>

## TODO
- [ ] Update readme to a paper-code-release style, i.e. more info about the paper
- [ ] Add more figures from the paper
- [ ] Add bibtex citation
- [ ] Unify naming of the robot, existing names: `hopper`, `mups_robot`, `hopper_v2` ...


## Installation

```bash
conda env create -f environment.yml
conda activate codesign
# Install Isaac Gym (Preview 4) into this env per NVIDIA's instructions, then:
pip install -e .
```


## Quick Start

Run unit tests to assert gradients are the same from finite difference (FD) and automatic differentiation (AD) for each CoDesign modules:

```bash
pytest tests/ -v -s
```

Run design optimizations (the pretrained `checkpoints/rainbow_v7` policy is loaded automatically):

```bash
python scripts/run_codesign.py       # Pure gradient descent
python scripts/run_cma_codesign.py   # Pure CMA-ES
python scripts/run_injected_es.py    # CMA-ES with gradient-injected candidates
python scripts/run_meanshift_es.py   # CMA-ES with gradient mean-shift
```

Collect design landscape for `policy_id` configured in `<mups_codesign/config.py>`:
```bash
python scripts/collect_landscape.py
```

Plot latest optimization trajectory over last collected objective landscape:
```bash
python scripts/plot_landscape.py --policy_id rainbow_v7
```
<!-- <img src="docs/opt_traj_overlap_landscape.png" width=500/> -->

Collect gradient vector field of 2D objective landscape from AD:
```bash
python scripts/collect_gradient_field.py
```

Collect gradient vector field of 2D objective landscape from FD:
```bash
python scripts/collect_gradient_field_fd.py
```

Plot last collected gradient vector field over objective landscape:
```bash
python scripts/plot_gradient_field.py --grad-magnitude 5
```
Use `--grad-magnitude` to scale vector magnitude for minimum overlap.

<!-- <img src="docs/gradient_field_ad.png" width=500/> -->


## Train a Locomotion Policy

A pretrained policy (`rainbow_v7`) ships in `checkpoints/`, so this step is optional. To train
your own, use the vendored `legged_gym`:

```bash
python scripts/train_policy.py --task hopper   # runs saved to logs/hopper/
python scripts/play_policy.py --task hopper    # visualize the latest run
```

Training runs are written to `logs/hopper/<timestamp>_<run_name>/`. The codesign config loads
its policy from `<policy_root>/<policy_id>/` (defaults: `policy_root="checkpoints"`,
`policy_id="rainbow_v7"`). To use a freshly trained policy, set `policy_root="logs/hopper"` and
`policy_id` to the new run directory name in `src/mups_codesign/config.py`.


### Development Logs

<details>

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
- [x] Visualize AD gradient field over landscape
- [x] Visualize FD gradient field over landscape
- [x] Make rollout_control_loop easier to take different combination of params
- [x] Start interfacing with CMA-ES
- [x] Move `evaluate_population()` and `compute_surrogate_gradient()` to `optim_helper.py`
- [x] Fix injected step calculation
- [x] Run experiments with different injection number
- [x] Create a submodule for hopper policy training, prepare for open-source release

</details>
