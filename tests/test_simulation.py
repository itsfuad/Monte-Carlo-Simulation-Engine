import pytest
import torch
import numpy as np
from src.core import MonteCarloSimulation
from src.samplers.smart_sampler import SmartSampler

def test_smart_sampler():
    dimension = 2
    sampler = SmartSampler(dimension=dimension)
    
    # Test sample generation
    n_samples = 1000
    samples = sampler.sample(n_samples)
    assert samples.shape == (n_samples, dimension)
    
    # Test update method
    weights = torch.ones(n_samples)
    sampler.update(samples, weights)
    
    # Test log probability computation
    log_probs = sampler.get_log_prob(samples)
    assert log_probs.shape == (n_samples,)

def test_monte_carlo_simulation():
    # Simple target function (unit circle area estimation)
    def target_function(samples):
        radius = torch.sum(samples ** 2, dim=1)
        return (radius <= 1.0).float()
    
    dimension = 2
    sampler = SmartSampler(dimension=dimension)
    
    simulation = MonteCarloSimulation(
        sampler=sampler,
        target_function=target_function,
        n_samples=1000,
        batch_size=100
    )
    
    # Run simulation
    results = simulation.run()
    
    # Check results format
    assert 'mean' in results
    assert 'std' in results
    assert 'n_samples' in results
    assert 'convergence_history' in results
    
    # Check if the estimate is reasonable (within 10% of pi)
    estimated_pi = 4.0 * results['mean']
    assert abs(estimated_pi - np.pi) / np.pi < 0.1

def test_convergence():
    # Target function that always returns 1
    def target_function(samples):
        return torch.ones(len(samples))
    
    dimension = 1
    sampler = SmartSampler(dimension=dimension)
    
    simulation = MonteCarloSimulation(
        sampler=sampler,
        target_function=target_function,
        n_samples=1000,
        batch_size=100,
        convergence_threshold=1e-6
    )
    
    results = simulation.run()
    
    # Check if mean is close to 1
    assert abs(results['mean'] - 1.0) < 1e-5
    
    # Check if convergence occurred
    assert len(results['convergence_history']) > 0
    assert results['convergence_history'][-1] < 1e-6

if __name__ == '__main__':
    pytest.main([__file__]) 