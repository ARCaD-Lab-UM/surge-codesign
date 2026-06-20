import isaacgym

import os
import pdb
import sys
import time
from dataclasses import asdict
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from cma import CMAEvolutionStrategy

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *

from surge_codesign.mups_robot import MupsRobot
from surge_codesign.config import CodesignConfig
from surge_codesign.data_logger import DataLogger
from surge_codesign.design_space import DesignSpace
from surge_codesign.design_objective import DesignObjective
from surge_codesign.optim_helper import evaluate_population, compute_surrogate_gradient, setup_isaac_env_and_policy, parse_seed
from surge_codesign.vis_helper import plot_optimization_history


# Set print precision
np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)


if __name__ == '__main__':
    POPULATION_SIZE = int(os.environ.get('POPULATION_SIZE', '16'))
    SIGMA_INIT = float(os.environ.get('SIGMA_INIT', '0.2'))  # Initial CMA-ES step size
    GRAD_STEP_SIZE = float(os.environ.get('GRAD_STEP_SIZE', '0.1'))
    GRAD_CLIP_NORM = 1.0      # Clip surrogate gradient norm before mean shift

    #* Initialize codesign config
    seed_override = parse_seed()
    design_config = CodesignConfig(
        **({'seed': seed_override} if seed_override is not None else {}),
        num_envs=POPULATION_SIZE,
        device="cuda",
        n_design_iter=int(os.environ.get('N_DESIGN_ITER', '50')),
        n_control_iter=int(os.environ.get('N_CONTROL_ITER', '100')),
        learning_rate=None,
    )

    # Setup isaacgym environment and control policy
    env, control_policy = setup_isaac_env_and_policy(design_config)

    N_DESIGN_ITER = design_config.n_design_iter
    N_CONTROL_ITER = design_config.n_control_iter
    n_params = len(design_config.active_param_names)
    print(f"Generations: {N_DESIGN_ITER}, Control Iterations: {N_CONTROL_ITER}")
    print(f"Population Size: {POPULATION_SIZE}, Initial Sigma: {SIGMA_INIT}")
    print(f"Mean Shift: step_size={GRAD_STEP_SIZE} (no Mahalanobis clipping)")
    print(f"Gradient Clip Norm: {GRAD_CLIP_NORM}")
    print(f"Initial Design Parameters: {design_config.raw_init_param_values}")

    #* Initialize codesign modules (with gradient tracking for surrogate gradient computation)
    srb_env = MupsRobot(design_config)
    design_objective_calculator = DesignObjective(design_config)
    design_space = DesignSpace(design_config, requires_grad=True)
    logger = DataLogger(root_dir=design_config.log_dir, run_name="hopper_codesign_surge")

    #* Setup CMA-ES optimizer
    init_normalized = design_space.active_normalized_param_values.detach().cpu().numpy()
    bounds_normalized = design_space.active_normalized_param_bounds.cpu().numpy()
    lower_bounds = bounds_normalized[:, 0]
    upper_bounds = bounds_normalized[:, 1]

    cma_options = {
        'popsize': POPULATION_SIZE,
        'bounds': [lower_bounds.tolist(), upper_bounds.tolist()],
        'maxiter': N_DESIGN_ITER,
        'verb_disp': 1,
        'verb_log': 0,
        'seed': design_config.seed,
    }

    es = CMAEvolutionStrategy(
        x0=init_normalized.tolist(),
        sigma0=SIGMA_INIT,
        inopts=cma_options
    )

    # Log metadata
    logger.log_metadata({
        "design_config": asdict(design_config),
        "param_names": list(design_space.active_param_names),
        "n_design_iter": N_DESIGN_ITER,
        "n_control_iter": N_CONTROL_ITER,
        "surge": {
            "population_size": POPULATION_SIZE,
            "sigma_init": SIGMA_INIT,
            "grad_step_size": GRAD_STEP_SIZE,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "bounds_normalized": bounds_normalized.tolist(),
        },
    })

    best_loss = float("inf")
    best_params = None
    generation = 0

    # History for the end-of-run optimization plot
    best_loss_history = []
    gen_best_params_history = []

    # Cosine step-size decay: GRAD_STEP_SIZE -> 0 at DECAY_END_FRAC of total iterations
    NO_COSINE_DECAY = os.environ.get('NO_COSINE_DECAY', '0') == '1'
    DECAY_END_FRAC = float(os.environ.get('DECAY_END_FRAC', '1'))
    DECAY_END = int(DECAY_END_FRAC * N_DESIGN_ITER)
    def cosine_step_size(gen):
        if NO_COSINE_DECAY:
            return GRAD_STEP_SIZE
        if gen >= DECAY_END:
            return 0.0
        return GRAD_STEP_SIZE * 0.5 * (1.0 + np.cos(np.pi * gen / DECAY_END))

    pbar = tqdm(total=N_DESIGN_ITER, desc="SurGE Generation", ncols=80, file=sys.stdout,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]")

    while not es.stop():
        iter_start = time.time()

        # Effective step size with cosine decay
        effective_step = cosine_step_size(generation)

        #* Step 1: Compute surrogate gradient at current CMA-ES mean
        current_mean = es.mean.copy()
        surrogate_grad, surrogate_loss = compute_surrogate_gradient(
            current_mean,
            env,
            control_policy,
            srb_env,
            design_objective_calculator,
            design_space,
            design_config,
        )

        # Clip gradient norm
        grad_norm = np.linalg.norm(surrogate_grad)
        if grad_norm > GRAD_CLIP_NORM:
            surrogate_grad = surrogate_grad * (GRAD_CLIP_NORM / grad_norm)
        grad_norm_clipped = np.linalg.norm(surrogate_grad)

        #* Step 2: Inject gradient via direct mean shift
        # Construct injection candidate along negative (surrogate) gradient direction
        grad_direction = es.C @ surrogate_grad # (num_params, )

        # Scale step to match CMA-ES typical step length: sigma * sqrt(n)
        new_mean = current_mean - effective_step * es.sigma * np.sqrt(n_params) * grad_direction / np.sqrt(surrogate_grad @ grad_direction)
        new_mean = np.clip(new_mean, lower_bounds, upper_bounds)
        mean_shift_norm = np.linalg.norm(new_mean - current_mean)
        es.mean = new_mean

        #* Step 3: Sample candidates from CMA-ES (around shifted mean)
        candidates_normalized = es.ask()

        #* Step 4: Evaluate candidates in TRUE dynamics (parallel, no gradients)
        fitness_values, all_objective_terms, candidates_raw = evaluate_population(
            candidates_normalized,
            env,
            control_policy,
            srb_env,
            design_objective_calculator,
            design_space,
            design_config,
        )

        #* Step 5: Update CMA-ES with fitness values
        es.tell(candidates_normalized, fitness_values)

        # Get current best from this generation
        gen_best_idx = np.argmin(fitness_values)
        gen_best_fitness = fitness_values[gen_best_idx]
        gen_best_normalized = candidates_normalized[gen_best_idx]
        gen_best_objective_terms = all_objective_terms[gen_best_idx]
        gen_best_raw = candidates_raw[gen_best_idx]

        # Update global best
        if gen_best_fitness < best_loss:
            best_loss = gen_best_fitness
            best_params = gen_best_raw.copy()

        # Track history for the end-of-run optimization plot
        best_loss_history.append(best_loss)
        gen_best_params_history.append(gen_best_raw)

        # Print generation summary
        print("")
        print(f"Generation {generation + 1}/{N_DESIGN_ITER}")
        print(f"  Gen Best Fitness: {gen_best_fitness:.4f}, Global Best: {best_loss:.4f}")
        print(f"  Surrogate Loss at Mean: {surrogate_loss:.4f}")
        print(f"  Gen Best Params (raw): {gen_best_raw}")
        print(f"  Gradient Norm (before/after clip): {grad_norm:.4f} / {grad_norm_clipped:.4f}")
        print(f"  Mean Shift: {'applied (norm={:.6f})'.format(mean_shift_norm)}, eff_step={effective_step:.4f}")
        print(f"  CMA-ES Sigma: {es.sigma:.6f}")
        term_summary = ", ".join(f"{name}: {val:.4f}" for name, val in gen_best_objective_terms.items())
        print(f"  Objective Components -> {term_summary}")

        iter_time_s = time.time() - iter_start

        # Log iteration data
        logger.log_iteration(
            iteration=generation,
            objective_total=gen_best_fitness,
            objective_terms=gen_best_objective_terms,
            params_value=gen_best_raw,
            params_normalized=np.array(gen_best_normalized),
            grad_norm=float(grad_norm),
            grad_terms=surrogate_grad,
            best_loss=best_loss,
            best_params=best_params,
            extra={
                "population_size": POPULATION_SIZE,
                "sigma": es.sigma,
                "iter_time_s": iter_time_s,
                "mean_fitness": np.mean(fitness_values),
                "std_fitness": np.std(fitness_values),
                "surrogate_loss": surrogate_loss,
                "grad_norm_clipped": float(grad_norm_clipped),
                "mean_shift_norm": float(mean_shift_norm),
                "grad_step_size": GRAD_STEP_SIZE,
                "effective_step_size": effective_step,
                "decay_end_frac": DECAY_END_FRAC,
            },
        )

        generation += 1
        pbar.update(1)

    pbar.close()

    print("\n" + "="*60)
    print("SurGE Design Optimization Completed.")
    print(f"Final Best Loss: {best_loss:.4f}")
    print(f"Final Best Parameters: {best_params}")
    print(f"CMA-ES Stop Reason: {es.stop()}")
    print("="*60)

    logger.close()
    print(f"Logs saved to {logger.run_dir}")

    #* Plot best-so-far objective and generation-best design parameters
    plot_optimization_history(
        best_so_far=best_loss_history,
        gen_best_params=gen_best_params_history,
        param_names=list(design_space.active_param_names),
        save_path=os.path.join(logger.run_dir, "optimization_history.png"),
        show=True,
    )
