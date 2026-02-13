import isaacgym

import pdb
import sys
import time
from dataclasses import asdict
import numpy as np
from tqdm import tqdm

import torch
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
    
    Args:
        candidates_normalized: List of normalized design parameters, each shape (num_params,)
        
    Returns:
        fitness_values: List of objective values (one per candidate)
        objective_terms_list: List of objective term dicts (one per candidate)
    """
    pop_size = len(candidates_normalized)
    
    # Convert all candidates to tensor: (pop_size, num_params)
    candidates_tensor = torch.tensor(
        np.array(candidates_normalized),
        dtype=design_config.dtype,
        device=design_config.device
    )
    # Convert to raw values
    candidates_raw = candidates_tensor * design_space.active_param_scales  # (pop_size, num_params)
    
    # Set design parameters: each environment gets a different candidate
    param_names = design_space.active_param_names
    env.set_design_params({name: val for name, val in zip(param_names, candidates_raw.T.detach())})  # (2, pop_size)
    srb_env.set_design_params(param_names, candidates_raw)  # (pop_size, num_params)
    
    with torch.no_grad():
        env.reset()
    
    # Run control loop without gradients - all envs run in parallel
    with torch.no_grad():
        total_design_objective, objective_term_sums = rollout_control_loop(
            env,
            control_policy,
            srb_env,
            design_objective_calculator,
            design_config.n_control_iter,
            headless=env.headless,
            modify_priv_obs=False
        )
    
    # total_design_objective shape: (num_envs,) = (pop_size,)
    fitness_values = total_design_objective.cpu().numpy().tolist()
    
    # objective_term_sums contains averaged values across all envs,
    # but we need per-candidate terms for logging the best one
    # For simplicity, we'll use the same aggregate terms for all candidates
    objective_terms_list = [objective_term_sums for _ in range(pop_size)]
    
    return fitness_values, objective_terms_list, candidates_raw.cpu().numpy()


if __name__ == '__main__':
    #* CMA-ES specific configuration
    POPULATION_SIZE = 16  # Number of candidates per generation = num_envs for parallel eval
    SIGMA_INIT = 0.3      # Initial step size (in normalized space)
    
    #* Initialize codesign config (num_envs = population size for parallel evaluation)
    seed_override = parse_seed()
    design_config = CodesignConfig(
        **({'seed': seed_override} if seed_override is not None else {}),
        num_envs=POPULATION_SIZE,
        device="cuda",
        n_design_iter=50,       # Number of CMA-ES generations
        n_control_iter=100,     # Control steps per evaluation
        learning_rate=None,     # Not used for CMA-ES
        raw_init_param_values=(7000, 0.15, 0.1, 0.02),
    )

    # Setup isaacgym environment and control policy
    env, control_policy = setup_isaac_env_and_policy(design_config)

    # Codesign parameters
    N_DESIGN_ITER = design_config.n_design_iter
    N_CONTROL_ITER = design_config.n_control_iter
    print(f"CMA-ES Generations: {N_DESIGN_ITER}, Control Iterations: {N_CONTROL_ITER}")
    print(f"Population Size: {POPULATION_SIZE}, Initial Sigma: {SIGMA_INIT}")
    print(f"Initial Design Parameters: {design_config.raw_init_param_values}")

    #* Initialize codesign modules (no gradient tracking needed)
    srb_env = MupsRobot(design_config)
    design_objective_calculator = DesignObjective(design_config)
    design_space = DesignSpace(design_config, requires_grad=False)
    logger = DataLogger(root_dir=design_config.log_dir, run_name="hopper_codesign_cmaes")

    #* Setup CMA-ES optimizer in normalized parameter space
    # Get initial normalized parameters and bounds
    init_normalized = design_space.active_normalized_param_values.detach().cpu().numpy()
    bounds_normalized = design_space.active_normalized_param_bounds.cpu().numpy()
    lower_bounds = bounds_normalized[:, 0].tolist()
    upper_bounds = bounds_normalized[:, 1].tolist()
    
    # Initialize CMA-ES
    cma_options = {
        'popsize': POPULATION_SIZE,
        'bounds': [lower_bounds, upper_bounds],
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
        "cma_es": {
            "population_size": POPULATION_SIZE,
            "sigma_init": SIGMA_INIT,
            "bounds_normalized": bounds_normalized.tolist(),
        },
    })
    
    best_loss = float("inf")
    best_params = None
    generation = 0

    # CMA-ES optimization loop
    pbar = tqdm(total=N_DESIGN_ITER, desc="CMA-ES Generation", ncols=80, file=sys.stdout)
    
    while not es.stop():
        iter_start = time.time()
        
        # Ask for new candidate solutions (in normalized space)
        candidates_normalized = es.ask()
        
        # Evaluate all candidates in parallel (one per environment)
        fitness_values, all_objective_terms, candidates_raw = evaluate_population(
            candidates_normalized,
            env,
            control_policy,
            srb_env,
            design_objective_calculator,
            design_space,
            design_config,
        )
        
        # Tell CMA-ES the fitness values
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
        print("")  # Flush newline after tqdm
        print(f"Generation {generation + 1}/{N_DESIGN_ITER}")
        print(f"  Gen Best Fitness: {gen_best_fitness:.4f}, Global Best: {best_loss:.4f}")
        print(f"  Gen Best Params (raw): {gen_best_raw}")
        print(f"  Global Best Params (raw): {best_params}")
        term_summary = ", ".join(f"{name}: {val:.4f}" for name, val in gen_best_objective_terms.items())
        print(f"  Objective Components -> {term_summary}")
        print(f"  CMA-ES Sigma: {es.sigma:.6f}")
        
        iter_time_s = time.time() - iter_start
        
        # Log iteration data
        logger.log_iteration(
            iteration=generation,
            objective_total=gen_best_fitness,
            objective_terms=gen_best_objective_terms,
            params_value=gen_best_raw,
            params_normalized=np.array(gen_best_normalized),
            grad_norm=0.0,  # No gradients in CMA-ES
            grad_terms=np.zeros(design_space.num_active_params),
            best_loss=best_loss,
            best_params=best_params,
            extra={
                "population_size": POPULATION_SIZE,
                "sigma": es.sigma,
                "iter_time_s": iter_time_s,
                "mean_fitness": np.mean(fitness_values),
                "std_fitness": np.std(fitness_values),
            },
        )
        
        generation += 1
        pbar.update(1)
    
    pbar.close()
    
    print("\n" + "="*60)
    print("CMA-ES Design Optimization Completed.")
    print(f"Final Best Loss: {best_loss:.4f}")
    print(f"Final Best Parameters: {best_params}")
    print(f"CMA-ES Stop Reason: {es.stop()}")
    print("="*60)

    # Close logger
    logger.close()
    print(f"Logs saved to {logger.run_dir}")

