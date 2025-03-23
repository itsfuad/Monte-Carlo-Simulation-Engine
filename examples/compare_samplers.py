import torch
import numpy as np
from src.core import MonteCarloSimulation
from src.samplers.uniform_sampler import UniformSampler
from src.samplers.smart_sampler import SmartSampler
from src.samplers.stratified_sampler import StratifiedSampler
from src.samplers.quasi_random_sampler import QuasiRandomSampler
from tqdm import tqdm
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend
import matplotlib.pyplot as plt

def estimate_pi_with_sampler(sampler, n_samples: int, n_runs: int = 5) -> tuple[float, float]:
    """Estimate pi using Monte Carlo method with a specific sampler.
    
    Args:
        sampler: The sampler to use
        n_samples: Number of samples per run
        n_runs: Number of runs to average over
        
    Returns:
        Tuple of (mean estimate, standard error)
    """
    # Define a target function (e.g., estimating the area of a unit circle)
    def target_function(samples):
        # Check if points fall within a unit circle
        radius = torch.sum(samples ** 2, dim=1)
        return (radius <= 1.0).float()
    
    # Create simulation instance
    simulation = MonteCarloSimulation(
        sampler=sampler,
        target_function=target_function,
        n_samples=n_samples,
        batch_size=1000,  # Increased batch size for better performance
        use_gpu=torch.cuda.is_available()
    )
    
    # Run multiple simulations
    estimates = []
    for _ in tqdm(range(n_runs), desc=f"Running {sampler.__class__.__name__}"):
        results = simulation.run()
        estimated_pi = 4.0 * results['mean']  # Scale by 4 since we're sampling from [-1,1]^2
        estimates.append(estimated_pi)
    
    # Calculate statistics
    mean_estimate = np.mean(estimates)
    std_error = np.std(estimates) / np.sqrt(n_runs)
    
    return mean_estimate, std_error

def estimate_gaussian_integral_with_sampler(sampler, n_samples: int, n_runs: int = 5) -> tuple[float, float]:
    """Estimate integral of exp(-x^2-y^2) over [-2,2]×[-2,2] using Monte Carlo."""
    def target_function(samples):
        # exp(-x^2-y^2) = exp(-r^2) where r is the radius
        radius = torch.sum(samples ** 2, dim=1)
        return torch.exp(-radius)
    
    simulation = MonteCarloSimulation(
        sampler=sampler,
        target_function=target_function,
        n_samples=n_samples,
        batch_size=1000,
        use_gpu=torch.cuda.is_available()
    )
    
    estimates = []
    for _ in tqdm(range(n_runs), desc=f"Running {sampler.__class__.__name__}"):
        results = simulation.run()
        # Scale by area of sampling region (4×4=16)
        estimated_integral = 16.0 * results['mean']
        estimates.append(estimated_integral)
    
    mean_estimate = np.mean(estimates)
    std_error = np.std(estimates) / np.sqrt(n_runs)
    
    return mean_estimate, std_error

def plot_results(results: dict, problem_name: str):
    """Plot comparison of results from different samplers."""
    samplers = list(results.keys())
    estimates = [results[s]['mean'] for s in samplers]
    errors = [results[s]['std_error'] for s in samplers]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot estimates
    x = np.arange(len(samplers))
    ax1.bar(x, estimates, yerr=errors, capsize=5)
    if problem_name == "pi":
        ax1.axhline(y=np.pi, color='r', linestyle='--', label='True π')
    ax1.set_xticks(x)
    ax1.set_xticklabels(samplers, rotation=45)
    ax1.set_title(f'{problem_name.capitalize()} Estimates by Sampler')
    ax1.set_ylabel('Estimated Value')
    if problem_name == "pi":
        ax1.legend()
    
    # Plot relative errors
    if problem_name == "pi":
        relative_errors = [abs(est - np.pi) / np.pi for est in estimates]
    else:
        true_value = np.pi  # True value of the Gaussian integral
        relative_errors = [abs(est - true_value) / true_value for est in estimates]
    
    ax2.bar(x, relative_errors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(samplers, rotation=45)
    ax2.set_title('Relative Errors by Sampler')
    ax2.set_ylabel('Relative Error')
    
    plt.tight_layout()
    return fig

def run_simulation(problem_name: str, n_samples: int, n_runs: int):
    """Run simulation for a specific problem."""
    dimension = 2
    
    # Initialize samplers with appropriate bounds
    if problem_name == "pi":
        samplers = {
            'Uniform': UniformSampler(dimension=dimension, lower_bound=-1.0, upper_bound=1.0),
            'Smart': SmartSampler(dimension=dimension, learning_rate=0.01),
            'Stratified': StratifiedSampler(dimension=dimension, strata_per_dim=5),
            'QuasiRandom': QuasiRandomSampler(dimension=dimension, scramble=True)
        }
        estimate_func = estimate_pi_with_sampler
    else:  # gaussian
        samplers = {
            'Uniform': UniformSampler(dimension=dimension, lower_bound=-2.0, upper_bound=2.0),
            'Smart': SmartSampler(dimension=dimension, learning_rate=0.01),
            'Stratified': StratifiedSampler(dimension=dimension, strata_per_dim=5),
            'QuasiRandom': QuasiRandomSampler(dimension=dimension, scramble=True)
        }
        estimate_func = estimate_gaussian_integral_with_sampler
    
    print(f"\nRunning {problem_name} estimation using {n_samples:,} samples per run, {n_runs} runs...")
    
    results = {}
    for name, sampler in samplers.items():
        print(f"\nTesting {name} sampler...")
        mean_estimate, std_error = estimate_func(sampler, n_samples, n_runs)
        results[name] = {
            'mean': mean_estimate,
            'std_error': std_error
        }
        
        print(f"\nResults for {name} sampler:")
        print(f"Estimated value: {mean_estimate:.10f}")
        print(f"Actual value:    {np.pi:.10f}")  # True value of Gaussian integral
        print(f"Absolute error:  {abs(mean_estimate - np.pi):.2e}")
        print(f"Standard error:   {std_error:.2e}")
        print(f"95% confidence interval: [{mean_estimate - 1.96*std_error:.10f}, {mean_estimate + 1.96*std_error:.10f}]")
    
    return results

def main():
    n_samples = 1_000_000
    n_runs = 5
    
    # Run π estimation
    pi_results = run_simulation("pi", n_samples, n_runs)
    plot_results(pi_results, "pi")
    plt.show()
    
    # Run Gaussian integral estimation
    gaussian_results = run_simulation("gaussian", n_samples, n_runs)
    plot_results(gaussian_results, "gaussian")
    plt.show()

if __name__ == "__main__":
    main() 