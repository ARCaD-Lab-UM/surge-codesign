"""
Unit test to verify autograd matches finite difference when an RL policy is in the loop.

The test:
1. Samples design params from design space
2. Creates fake policy input, feeding design params to the last two dims of privileged obs
3. Gets action from policy
4. Computes scalar loss based on action (squared sum)
5. Compares autograd vs finite difference gradients
"""

import pytest
import torch
import torch.nn as nn

from mups_codesign.config import CodesignConfig
from mups_codesign.design_space import DesignSpace


def make_simple_policy(obs_dim, priv_obs_dim, action_dim, hidden_dims=(64, 32), dtype=torch.float64):
    """Create a simple MLP policy that takes obs and privileged obs, outputs actions."""
    input_dim = obs_dim + priv_obs_dim
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim, dtype=dtype))
        layers.append(nn.ELU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, action_dim, dtype=dtype))
    return nn.Sequential(*layers)


def compute_fd_grad(loss_fn, params, eps=1e-6):
    """Compute finite difference gradient using central differences."""
    grad = torch.zeros_like(params)
    for i in range(params.numel()):
        p_plus = params.clone()
        p_plus.view(-1)[i] += eps
        p_minus = params.clone()
        p_minus.view(-1)[i] -= eps
        grad.view(-1)[i] = (loss_fn(p_plus) - loss_fn(p_minus)) / (2 * eps)
    return grad


def compute_policy_gradients(seed, num_steps):
    """Helper to compute AD and FD gradients for policy with recurrent state dependency.
    
    Returns (ad_grad, fd_grad, loss, param_base) for comparison.
    """
    torch.manual_seed(seed)

    # Config
    cfg = CodesignConfig(
        num_envs=1,
        device="cpu",
        dtype=torch.float64,
        active_param_names=("ups_ks", "ups_l0", "ups_l2", "ups_l4"),
        raw_init_param_values=None,
    )

    # Dimensions matching hopper config structure
    obs_dim = 21  # num_proprio from hopper config
    priv_obs_dim = 11  # num_privileged_obs from hopper config
    action_dim = 2  # num_actions from hopper config

    # Create policy with frozen weights
    policy = make_simple_policy(obs_dim, priv_obs_dim, action_dim, dtype=cfg.dtype)
    for param in policy.parameters():
        param.requires_grad = False

    # Design space
    design_space = DesignSpace(cfg, requires_grad=True)
    num_design_params = design_space.num_active_params
    param_scales = design_space.active_param_scales
    param_base = design_space.sample_active_param_values()

    # Initial observation (fixed)
    obs_init = torch.randn((cfg.num_envs, obs_dim), dtype=cfg.dtype)
    priv_obs_base = torch.randn((cfg.num_envs, priv_obs_dim), dtype=cfg.dtype)

    def compute_loss(params):
        """Compute loss with recurrent state dependency.
        
        Previous action contributes to next observation, creating dependencies.
        """
        total_loss = torch.zeros(1, dtype=cfg.dtype)
        prev_action = torch.zeros((cfg.num_envs, action_dim), dtype=cfg.dtype)
        obs = obs_init.clone()
        
        for t in range(num_steps):
            # Modify obs based on previous action (simulates state update)
            obs = obs.clone()

            #! This is the critical break that make AD differ from FD
            with torch.no_grad():
                obs[:, :action_dim] = obs[:, :action_dim] + prev_action
            
            priv_obs = priv_obs_base.clone()
            normalized_params = params / param_scales
            priv_obs[:, -num_design_params:] = normalized_params.unsqueeze(0)
            
            # Concatenate obs and priv_obs as policy input
            policy_input = torch.cat([obs, priv_obs], dim=-1)
            action = policy(policy_input)
            
            # Compute loss as squared sum of actions
            step_loss = action.square().sum()
            total_loss = total_loss + step_loss
            
            prev_action = action
        
        return total_loss

    # Autograd
    params = param_base.clone().requires_grad_(True)
    loss = compute_loss(params)
    loss.backward()
    ad_grad = params.grad.clone()

    # Finite difference
    fd_grad = compute_fd_grad(compute_loss, param_base)

    # Print for debugging (pytest -s)
    print(f"\n[seed={seed}, steps={num_steps}] loss={loss.item():.6f}")
    print(f"  params: {param_base.tolist()}")
    print(f"  autograd: {ad_grad.tolist()}")
    print(f"  fd_grad:  {fd_grad.tolist()}")
    print(f"  diff:     {(ad_grad - fd_grad).abs().tolist()}")

    return ad_grad, fd_grad, loss, param_base


@pytest.mark.parametrize("seed", [0, 42])
def test_policy_gradient_single_step_passes(seed):
    """Test that single-step (num_steps=1) autograd matches finite difference.
    
    With only one step, there's no recurrence, so torch.no_grad() doesn't break anything.
    """
    ad_grad, fd_grad, loss, param_base = compute_policy_gradients(seed, num_steps=1)
    
    assert torch.allclose(ad_grad, fd_grad, rtol=1e-4, atol=1e-6), \
        f"Gradient mismatch: autograd={ad_grad}, fd={fd_grad}"


@pytest.mark.parametrize("seed", [0, 42])
def test_policy_gradient_multi_step_fails(seed):
    """Test that multi-step (num_steps=10) autograd does NOT match finite difference.
    
    The torch.no_grad() block in the observation update breaks the gradient chain,
    causing AD to miss gradients that flow through the recurrent state dependency.
    """
    ad_grad, fd_grad, loss, param_base = compute_policy_gradients(seed, num_steps=10)
    
    # This should FAIL to be close - assert that it does NOT match
    assert not torch.allclose(ad_grad, fd_grad, rtol=1e-4, atol=1e-6), \
        f"Expected gradient mismatch but they matched: autograd={ad_grad}, fd={fd_grad}"
