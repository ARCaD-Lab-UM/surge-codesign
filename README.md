# SurGE

[**Paper**]() (coming soon) | [**Project Page**]() (coming soon) | [**Video**](https://youtu.be/amKPB2cOvBo)

Official implementation of *"SurGE: Surrogate Gradient-guided Evolution for Co-design of Legged Robots with Parallel Elasticity"*.


## Overview

<img src="docs/diff_codesign.png" width=900/>


SurGE jointly optimizes the spring design and control policy of legged robots with parallel elasticity.
To recover gradient information despite this non-differentiability, SurGE differentiates a surrogate pipeline, a kinodynamic single-rigid-body model paired with a design-aware policy, and injects the resulting surrogate gradient into CMA-ES through a mean shift with cosine-annealed decay. This converges faster and more reproducibly than pure gradient-based or pure evolutionary search, with the improvement transferring to hardware.


## Installation

### Dependencies
The code is tested on the following setup:
* Ubuntu 22.04
* Python 3.8
* Isaac Gym (Preview 4)

### Environment Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/ARCaD-Lab-UM/mups-codesign.git
   cd mups-codesign
   ```
2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate codesign
   ```
3. Download and install Isaac Gym into this env per [NVIDIA's instructions](https://developer.nvidia.com/isaac-gym).
4. Install the co-design code:
   ```bash
   pip install -e .
   ```

The editable install builds all three packages from `src/`: `mups_codesign`, `legged_gym`, and `rsl_rl`.


## Quick Start

### Co-design with SurGE
```bash
python scripts/run_surge_codesign.py
```

### Baselines

```bash
python scripts/run_gd_codesign.py    # vanilla gradient descent
python scripts/run_cma_codesign.py   # vanilla CMA-ES
```

### Visualization & Analysis

<img src="docs/landscape_combined.png" width=500/>

```bash
python scripts/collect_landscape.py                          # collect objective landscape
python scripts/plot_landscape.py --policy_id rainbow_v7      # plot landscape
python scripts/collect_gradient_field.py                     # collect gradient field
python scripts/plot_gradient_field.py --grad-magnitude 5     # plot gradient field
```


## Train a Design-Aware Locomotion Policy

<img src="docs/policy_arch.png" width=450/>

Pretrained checkpoints are shipped in `checkpoints/`.
To visualize a pretrained policy:
```bash
python scripts/play_policy.py --task hopper --load_pretrained_ckpt
```

To train a new one:
```bash
python scripts/train_policy.py --task hopper --headless
```

> [!NOTE]
> To use a newly trained policy in co-design, set `policy_root="logs/<exp_name>"` and `policy_id="<run_name>"` in `src/mups_codesign/config.py`


## Code Structure

The co-design logic lives in `src/mups_codesign/`, alongside our customized RL framework.

```
src/mups_codesign/         # core co-design package
  config.py                # configuration
  design_space.py          # design parameters and bounds
  design_objective.py      # design objective
  mups_robot.py            # differentiable Kino-SRB surrogate
  mups_spring.py           # differentiable UPS spring model
  optim_helper.py          # rollout and gradient engine
  data_logger.py           # logging
  vis_helper.py            # plotting
src/legged_gym/            # customized legged-gym (hopper simulation env)
src/rsl_rl/                # customized rsl-rl (RL framework)
```


## Citation

If you find this code useful for your research, please consider citing our paper:
```bibtex
@article{zhuang2026surge,
  title={SurGE: Surrogate Gradient-guided Evolution for Co-design of Legged Robots with Parallel Elasticity},
  author={Yulun Zhuang, Yue Qin, Justin Lu, Zelin Shen, Yichen Wang, Sicheng He and Yanran Ding},
  journal={arXiv preprint (coming soon)},
  year={2026}
}
```


## Troubleshooting

If you see an error about missing `libpython3.8.so.1.0` when importing Isaac Gym, copy it from conda lib to isaacgym bindings, i.e.
```bash
cp path_to_conda/envs/codesign/lib/libpython3.8.so.1.0 path_to_isaacgym/python/isaacgym/_bindings/linux-x86_64/
```
