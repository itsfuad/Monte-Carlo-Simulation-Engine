import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import torch
import numpy as np
from src.core import MonteCarloSimulation
from src.samplers.smart_sampler import SmartSampler
from src.samplers.stratified_sampler import StratifiedSampler
from src.samplers.quasi_random_sampler import QuasiRandomSampler
from src.core.optimizers import AdaptiveOptimizer

@pytest.fixture
def setup_samplers():
    dimension = 2
    return {
        'smart': SmartSampler(dimension=dimension),
        'stratified': StratifiedSampler(dimension=dimension),
        'quasi': QuasiRandomSampler(dimension=dimension)
    }

@pytest.fixture
def setup_optimizer():
    return AdaptiveOptimizer(learning_rate=0.01)

def test_smart_sampler(setup_samplers):
    sampler = setup_samplers['smart']
    
    # Test sample generation
    n_samples = 1000
    samples = sampler.sample(n_samples)
    assert samples.shape == (n_samples, sampler.dimension)
    
    # Test update method
    weights = torch.ones(n_samples)
    sampler.update(samples, weights)
    
    # Test visualization methods
    fig = sampler.plot_sampling_distribution()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test best samples getter
    best_samples, best_weights = sampler.get_best_samples()
    assert best_samples is not None
    assert best_weights is not None

def test_stratified_sampler(setup_samplers):
    sampler = setup_samplers['stratified']
    
    # Test sample generation
    n_samples = 1000
    samples = sampler.sample(n_samples)
    assert samples.shape == (n_samples, sampler.dimension)
    
    # Test update method
    weights = torch.ones(n_samples)
    sampler.update(samples, weights)
    
    # Test visualization methods
    fig = sampler.plot_strata()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    fig = sampler.plot_stratum_weights()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_quasi_random_sampler(setup_samplers):
    sampler = setup_samplers['quasi']
    
    # Test sample generation
    n_samples = 1000
    samples = sampler.sample(n_samples)
    assert samples.shape == (n_samples, sampler.dimension)
    
    # Test visualization methods
    fig = sampler.plot_sequence(n_points=100)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    fig = sampler.plot_discrepancy(max_points=200, step=50)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test reset
    old_index = sampler.current_index
    sampler.reset()
    assert sampler.current_index == 0
    assert old_index > 0

def test_optimizer_visualization(setup_optimizer):
    optimizer = setup_optimizer
    
    # Generate some dummy data
    n_samples = 100
    dimension = 2
    samples = torch.randn(n_samples, dimension)
    values = torch.randn(n_samples)
    
    # Run optimization
    for _ in range(5):
        optimizer.optimize(samples, values)
    
    # Test visualization methods
    fig = optimizer.plot_optimization_history()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    fig = optimizer.plot_gradient_history()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_monte_carlo_simulation():
    # Simple target function (unit circle area estimation)
    def target_function(samples):
        radius = torch.sum(samples ** 2, dim=1)
        return (radius <= 1.0).float()
    
    dimension = 2
    sampler = SmartSampler(
        dimension=dimension,
        initial_std=torch.ones(dimension) * 1.0,  # Start with wider sampling
        learning_rate=0.01
    )
    optimizer = AdaptiveOptimizer(learning_rate=0.001)  # Slower learning for stability
    
    simulation = MonteCarloSimulation(
        sampler=sampler,
        target_function=target_function,
        optimizer=optimizer,
        n_samples=100000,  # Much more samples
        batch_size=1000,   # Larger batches
        convergence_threshold=1e-4,
        max_iterations=500  # Allow more iterations
    )
    
    # Run simulation
    results = simulation.run()
    
    # Check results format
    assert 'mean' in results
    assert 'std' in results
    assert 'n_samples' in results
    assert 'convergence_history' in results
    assert 'optimization_history' in results
    
    # Check if the estimate is reasonable (within 15% of pi)
    estimated_pi = 4.0 * results['mean']
    assert abs(estimated_pi - np.pi) / np.pi < 0.15
    
    # Additional checks for simulation quality
    assert results['n_samples'] >= 10000  # Ensure minimum number of samples used
    assert len(results['convergence_history']) >= 10  # Ensure minimum iterations

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