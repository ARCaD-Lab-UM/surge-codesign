# SurGE: Surrogate Gradient-guided Evolution for Co-design of Legged Robots with Parallel Elasticity

<img src="docs/diff_codesign.png" width=900/>

## TODO
- [x] Add more figures from the paper
- [x] Add bibtex citation
- [ ] Update description to a paper-code-release style, i.e. more info about the paper and related medias
- [ ] Unify naming of the robot, existing names: `hopper`, `mups_robot`, `hopper_v2` ...


## Installation

```bash
conda env create -f environment.yml
conda activate codesign
# Install Isaac Gym (Preview 4) into this env per NVIDIA's instructions, then:
pip install -e .
```


## Quick Start

<img src="docs/landscape_combined.png" width=500/>

### Main Scripts
Run design optimizations (the pretrained `checkpoints/rainbow_v7` policy is loaded automatically):

```bash
python scripts/run_codesign.py       # Pure gradient descent
python scripts/run_cma_codesign.py   # Pure CMA-ES
python scripts/run_injected_es.py    # CMA-ES with gradient-injected candidates
python scripts/run_meanshift_es.py   # CMA-ES with gradient mean-shift
```

### Visualization and Analysis
Collect design landscape for `policy_id` configured in `<mups_codesign/config.py>`:
```bash
python scripts/collect_landscape.py
```

Plot latest optimization trajectory over last collected objective landscape:
```bash
python scripts/plot_landscape.py --policy_id rainbow_v7
```

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


## Train a Locomotion Policy

<img src="docs/policy_arch.png" width=500/>

A pretrained policy (`rainbow_v7`) ships in `checkpoints/`. To train
your own policy:

```bash
python scripts/train_policy.py --task hopper   # runs saved to logs/hopper/
python scripts/play_policy.py --task hopper    # visualize the latest run
```

Training runs are written to `logs/hopper/<timestamp>_<run_name>/`. The codesign config loads
its policy from `<policy_root>/<policy_id>/` (defaults: `policy_root="checkpoints"`,
`policy_id="rainbow_v7"`). To use a freshly trained policy, set `policy_root="logs/hopper"` and
`policy_id` to the new run directory name in `src/mups_codesign/config.py`.

## Citation
If you find this code useful for your research, please consider citing our paper:
```bibtex
@inproceedings{zhuang@surge,
  title={SurGE: Surrogate Gradient-guided Evolution for Co-design of Legged Robots with Parallel Elasticity},
  author={},
  booktitle={},
  year={2026}
}
```
