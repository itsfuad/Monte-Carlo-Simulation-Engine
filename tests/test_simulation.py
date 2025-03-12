import pytest
import matplotlib

import matplotlib.pyplot as plt
import torch
import numpy as np
from src.core import MonteCarloSimulation
from src.samplers.smart_sampler import SmartSampler
from src.samplers.stratified_sampler import StratifiedSampler
from src.samplers.quasi_random_sampler import QuasiRandomSampler
from src.core.optimizers import AdaptiveOptimizer

matplotlib.use("Agg")  # Use non-interactive backend for testing


@pytest.fixture
def setup_samplers():
    dimension = 2
    return {
        "smart": SmartSampler(dimension=dimension),
        "stratified": StratifiedSampler(dimension=dimension),
        "quasi": QuasiRandomSampler(dimension=dimension),
    }


@pytest.fixture
def setup_optimizer():
    return AdaptiveOptimizer(learning_rate=0.01)


def test_smart_sampler(setup_samplers):
    sampler = setup_samplers["smart"]

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
    sampler = setup_samplers["stratified"]

    # Test sample generation with power of 2 samples
    n_samples = 1024  # 2^10
    samples = sampler.sample(n_samples)
    assert samples.shape == (n_samples, sampler.dimension)

    # Test update method
    weights = torch.ones(n_samples)
    sampler.update(samples, weights)

    # Test visualization methods
    with torch.no_grad():  # Suppress meshgrid warning
        fig = sampler.plot_strata()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        fig = sampler.plot_stratum_weights()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_quasi_random_sampler(setup_samplers):
    sampler = setup_samplers["quasi"]

    # Test sample generation with power of 2 samples
    n_samples = 1024  # 2^10
    samples = sampler.sample(n_samples)
    assert samples.shape == (n_samples, sampler.dimension)

    # Test visualization methods
    fig = sampler.plot_sequence(n_points=128)  # 2^7
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    fig = sampler.plot_discrepancy(max_points=256, step=64)  # Powers of 2
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
    """Test Monte Carlo simulation for estimating pi."""
    dimension = 2

    # Use QuasiRandomSampler for better uniform coverage
    sampler = QuasiRandomSampler(
        dimension=dimension, scramble=True  # Enable scrambling for better randomization
    )

    # No optimizer needed for quasi-random sampling
    optimizer = None

    def target_function(x: torch.Tensor) -> torch.Tensor:
        """Target function for pi estimation (circle area method)."""
        return (torch.sum(x**2, dim=1) <= 1.0).float()

    # Configure simulation with quasi-random parameters
    # Use power of 2 for n_samples to satisfy Sobol sequence requirements
    n_samples = 2**20  # 1,048,576 samples (slightly more than 1M)
    batch_size = 2**13  # 8,192 (power of 2)

    simulation = MonteCarloSimulation(
        sampler=sampler,
        optimizer=optimizer,
        target_function=target_function,
        n_samples=n_samples,
        batch_size=batch_size,
        convergence_threshold=1e-3,
        max_iterations=200,
        seed=42,
    )

    # Run simulation
    results = simulation.run()

    # Ensure all required keys are present
    required_keys = ["values", "convergence", "mean", "std", "n_samples"]
    for key in required_keys:
        assert key in results, f"Missing required key: {key}"

    # Calculate pi estimate using all values
    all_values = torch.cat(results["values"])
    pi_estimate = 4.0 * torch.mean(all_values)
    relative_error = abs(pi_estimate.item() - np.pi) / np.pi

    print(f"\nPi estimate: {pi_estimate:.6f}")
    print(f"Relative error: {relative_error:.6f}")

    # Assert with appropriate tolerance
    assert relative_error < 0.02, f"Pi estimation error too large: {relative_error:.6f}"
    assert len(results["convergence"]) > 0, "No convergence data recorded"
    assert any(
        conv < simulation.convergence_threshold for conv in results["convergence"]
    ), "Failed to converge"


def test_convergence():
    """Test convergence of Monte Carlo simulation with a simple constant function."""

    # Target function that always returns 1
    def target_function(samples):
        return torch.ones(len(samples))

    dimension = 1
    sampler = SmartSampler(
        dimension=dimension,
        initial_mean=torch.zeros(dimension),
        initial_std=torch.ones(dimension),
        learning_rate=0.01,
    )

    simulation = MonteCarloSimulation(
        sampler=sampler,
        target_function=target_function,
        n_samples=5000,  # Increased samples
        batch_size=100,
        convergence_threshold=1e-4,  # Relaxed threshold
        max_iterations=100,
        seed=42,
    )

    results = simulation.run()

    # Check if mean is close to 1
    assert abs(results["mean"] - 1.0) < 1e-3

    # Check if convergence occurred
    assert len(results["convergence"]) > 0, "No convergence data recorded"

    # Get non-inf convergence values
    finite_convergence = [x for x in results["convergence"] if x != float("inf")]
    assert len(finite_convergence) > 0, "No finite convergence values recorded"

    # Check best convergence value
    min_convergence = min(finite_convergence)
    assert (
        min_convergence < 1e-3
    ), f"Best convergence value {min_convergence} not small enough"


if __name__ == "__main__":
    pytest.main([__file__])
