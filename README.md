# SurGE

[Paper Link](https://arxiv.org/abs/TODO) | [Project Page](https://silvery107.github.io/surge) | [Video](https://youtu.be/amKPB2cOvBo)

This repository contains the code for our paper "SurGE: Surrogate Gradient-guided Evolution for Co-design of Legged Robots with Parallel Elasticity". 


## Overview

<img src="docs/diff_codesign.png" width=900/>


SurGE is a co-design framework that combines gradients from surrogate dynamics model with evolutionary search to optimize both the design and control of legged robots with parallel elasticity. 

Our method achieves superior performance compared to pure gradient-based or pure evolutionary approaches, and can effectively navigate complex design landscapes. 
The codebase includes implementations of the SurGE algorithm, training scripts for locomotion policies, and tools for visualizing design landscapes and optimization trajectories.


## Installation

### Dependencies
* Ubuntu 22.04
* Python 3.8
* Isaac Gym (Preview 4)

### Environment Setup
```bash
conda env create -f environment.yml
conda activate codesign
# Install Isaac Gym into this env per NVIDIA's instructions, then:
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

<img src="docs/policy_arch.png" width=450/>

A pretrained policy is shipped in `checkpoints/`. To train
a new policy:

```bash
python scripts/train_policy.py --task hopper   # runs saved to logs/hopper/
python scripts/play_policy.py --task hopper    # visualize the latest run
```

> [!NOTE]  
> Training runs are logged to `logs/hopper/<timestamp>_<run_name>/`. The codesign config loads its policy from `<policy_root>/<policy_id>/` (defaults: `policy_root="checkpoints"`, `policy_id="rainbow_v7"`). 
> 
> To use a freshly trained policy, set `policy_root="logs/hopper"` and `policy_id` to the new run directory name in `src/mups_codesign/config.py`.


## Code Structure
TODO


## Citation
If you find this code useful for your research, please consider citing our paper:
```bibtex
@article{zhuang2026surge,
  title={SurGE: Surrogate Gradient-guided Evolution for Co-design of Legged Robots with Parallel Elasticity},
  author={Yulun Zhuang, Yue Qin, Justin Lu, Zelin Shen, Yichen Wang, Sicheng He and Yanran Ding},
  journal={arXiv preprint TODO},
  year={2026}
}
```
