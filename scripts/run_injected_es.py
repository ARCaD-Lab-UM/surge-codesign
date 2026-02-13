import isaacgym

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

from mups_codesign.mups_robot import MupsRobot
from mups_codesign.config import CodesignConfig
from mups_codesign.data_logger import DataLogger
from mups_codesign.design_space import DesignSpace
from mups_codesign.design_objective import DesignObjective
from mups_codesign.optim_helper import rollout_control_loop, setup_isaac_env_and_policy, parse_seed


# Set print precision
np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)


def evaluate_population(
    candidates_normalized: list,
    env,
    control_policy,
    srb_env: MupsRobot,
    design_objective_calculator: DesignObjective,
    design_space: DesignSpace,
    design_config: CodesignConfig,
):
    """
    Evaluate all candidates in parallel by assigning each to a different environment.
    Uses no_grad for pure fitness evaluation in true dynamics.
    """
    pop_size = len(candidates_normalized)

    candidates_tensor = torch.tensor(
        np.array(candidates_normalized),
        dtype=design_config.dtype,
        device=design_config.device
    )
    candidates_raw = candidates_tensor * design_space.active_param_scales

    param_names = design_space.active_param_names
    env.set_design_params({name: val for name, val in zip(param_names, candidates_raw.T.detach())})
    srb_env.set_design_params(param_names, candidates_raw)

    with torch.no_grad():
        env.reset()
        total_design_objective, objective_term_sums = rollout_control_loop(
            env,
            control_policy,
            srb_env,
            design_objective_calculator,
            design_config.n_control_iter,
            headless=env.headless,
            modify_priv_obs=False
        )

    fitness_values = total_design_objective.cpu().numpy().tolist()
    objective_terms_list = [objective_term_sums for _ in range(pop_size)]

    return fitness_values, objective_terms_list, candidates_raw.cpu().numpy()


def compute_surrogate_gradient(
    normalized_params: np.ndarray,
    env,
    control_policy,
    srb_env: MupsRobot,
    design_objective_calculator: DesignObjective,
    design_space: DesignSpace,
    design_config: CodesignConfig,
):
    """
    Compute gradient of the surrogate objective at a single design point.
    Uses the differentiable surrogate dynamics pipeline.

    Args:
        normalized_params: Design parameters in normalized space, shape (num_params,)

    Returns:
        gradient: Gradient in normalized space, shape (num_params,)
        loss_value: Scalar loss at this point
    """
    with torch.no_grad():
        design_space.active_normalized_param_values.copy_(
            torch.tensor(normalized_params, dtype=design_config.dtype, device=design_config.device)
        )
    design_space.project_active_params_into_bounds()

    design_space.active_normalized_param_values.requires_grad_(True)

    param_names = design_space.active_param_names
    param_values = design_space.active_param_values
    param_values_detached = design_space.detached_active_param_values

    env.set_design_params({name: val for name, val in zip(param_names, param_values_detached.detach())})
    srb_env.set_design_params(
        param_names,
        param_values.unsqueeze(0).expand(design_config.num_envs, -1)
    )

    with torch.no_grad():
        env.reset()

    total_design_objective, _ = rollout_control_loop(
        env,
        control_policy,
        srb_env,
        design_objective_calculator,
        design_config.n_control_iter,
        headless=env.headless,
        modify_priv_obs=True
    )

    loss = total_design_objective.mean()
    loss.backward()

    gradient = None
    if design_space.active_normalized_param_values.grad is not None:
        gradient = design_space.active_normalized_param_values.grad.detach().cpu().numpy().copy()
    else:
        raise ValueError("Gradient is None from AD")

    design_space.active_normalized_param_values.grad = None
    design_space.active_normalized_param_values.requires_grad_(False)

    return gradient, loss.item()


if __name__ == '__main__':
    #* CMA-ES + Injection configuration
    POPULATION_SIZE = 16      # Number of candidates per generation
    SIGMA_INIT = 0.3          # Initial CMA-ES step size
    N_INJECT = 1              # Number of gradient-based solutions to inject per generation
    GRAD_STEP_SIZE = 1e-2      # Step size multiplier for gradient injection (in sigma-normalized units)
    GRAD_CLIP_NORM = 1.0      # Clip surrogate gradient norm before constructing injected solution
    USE_NATURAL_GRADIENT = True  # Use C @ grad (natural gradient in CMA coordinates) for injection

    #* Initialize codesign config
    seed_override = parse_seed()
    design_config = CodesignConfig(
        **({'seed': seed_override} if seed_override is not None else {}),
        num_envs=POPULATION_SIZE,
        device="cuda",
        n_design_iter=50,
        n_control_iter=100,
        learning_rate=None,
        raw_init_param_values=(7000, 0.15, 0.1, 0.02),
    )

    # Setup isaacgym environment and control policy
    env, control_policy = setup_isaac_env_and_policy(design_config)

    N_DESIGN_ITER = design_config.n_design_iter
    N_CONTROL_ITER = design_config.n_control_iter
    n_params = len(design_config.active_param_names)
    print(f"CMA-ES + Injection (Hansen 2011) - Generations: {N_DESIGN_ITER}, Control Iterations: {N_CONTROL_ITER}")
    print(f"Population Size: {POPULATION_SIZE}, Initial Sigma: {SIGMA_INIT}")
    print(f"Injection: {N_INJECT} solution(s)/gen, step_size={GRAD_STEP_SIZE}, natural_grad={USE_NATURAL_GRADIENT}")
    print(f"Gradient Clip Norm: {GRAD_CLIP_NORM}")
    print(f"Initial Design Parameters: {design_config.raw_init_param_values}")

    #* Initialize codesign modules (with gradient tracking for surrogate gradient computation)
    srb_env = MupsRobot(design_config)
    design_objective_calculator = DesignObjective(design_config)
    design_space = DesignSpace(design_config, requires_grad=True)
    logger = DataLogger(root_dir=design_config.log_dir, run_name="hopper_codesign_cma_inject")

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
        "cma_inject": {
            "population_size": POPULATION_SIZE,
            "sigma_init": SIGMA_INIT,
            "n_inject": N_INJECT,
            "grad_step_size": GRAD_STEP_SIZE,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "use_natural_gradient": USE_NATURAL_GRADIENT,
            "bounds_normalized": bounds_normalized.tolist(),
        },
    })

    best_loss = float("inf")
    best_params = None
    generation = 0

    pbar = tqdm(total=N_DESIGN_ITER, desc="CMA-Inject Generation", ncols=80, file=sys.stdout)

    while not es.stop():
        iter_start = time.time()

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

        #* Step 2: Compute and inject gradient-based solution to CMA-ES
        # Hansen 2011: inject x_inj = mean - step * direction
        # Direction can be raw gradient or natural gradient (C @ grad)
        if USE_NATURAL_GRADIENT:
            # Use CMA-ES covariance to transform gradient into natural coordinates
            # This respects the learned landscape geometry: C @ grad
            C = es.sm.covariance_matrix  # (n, n)
            grad_direction = C @ surrogate_grad
        else:
            grad_direction = surrogate_grad

        # Normalize direction and scale by sigma * sqrt(n), matching CMA-ES internal step scale
        dir_norm = np.linalg.norm(grad_direction)
        if dir_norm > 1e-12:
            # Construct injected solution: step along negative gradient direction
            # Scale so the step magnitude is comparable to CMA-ES's typical step length
            injection_solution = current_mean - GRAD_STEP_SIZE * es.sigma * np.sqrt(n_params) * (grad_direction / dir_norm)
            # Clip to bounds before injection (pycma also handles internal clipping)
            injection_solution = np.clip(injection_solution, lower_bounds, upper_bounds)
            es.inject([injection_solution])
        else:
            # Gradient is essentially zero; skip injection this generation
            injection_solution = None

        candidates_normalized = es.ask()

        #* Step 4: Evaluate all candidates in TRUE dynamics (parallel, no gradients)
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

        # Print generation summary
        print("")
        print(f"Generation {generation + 1}/{N_DESIGN_ITER}")
        print(f"  Gen Best Fitness: {gen_best_fitness:.4f}, Global Best: {best_loss:.4f}")
        print(f"  Surrogate Loss at Mean: {surrogate_loss:.4f}")
        print(f"  Gen Best Params (raw): {gen_best_raw}")
        print(f"  Gradient Norm (before/after clip): {grad_norm:.4f} / {grad_norm_clipped:.4f}")
        print(f"  Injection: {'active' if injection_solution is not None else 'skipped (zero grad)'}")
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
                "injection_active": injection_solution is not None,
                "use_natural_gradient": USE_NATURAL_GRADIENT,
                "grad_step_size": GRAD_STEP_SIZE,
            },
        )

        generation += 1
        pbar.update(1)

    pbar.close()

    print("\n" + "="*60)
    print("CMA-ES + Injection (Hansen 2011) Design Optimization Completed.")
    print(f"Final Best Loss: {best_loss:.4f}")
    print(f"Final Best Parameters: {best_params}")
    print(f"CMA-ES Stop Reason: {es.stop()}")
    print("="*60)

    logger.close()
    print(f"Logs saved to {logger.run_dir}")
