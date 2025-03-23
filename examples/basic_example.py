import torch
import numpy as np
from src.core import MonteCarloSimulation
from src.samplers.uniform_sampler import UniformSampler
from tqdm import tqdm

def estimate_pi(n_samples: int, n_runs: int = 5) -> tuple[float, float]:
    """Estimate pi using Monte Carlo method with multiple runs.
    
    Args:
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
    
    # Initialize the sampler with uniform distribution in [-1, 1]^2
    dimension = 2  # 2D circle
    sampler = UniformSampler(
        dimension=dimension,
        lower_bound=-1.0,
        upper_bound=1.0
    )
    
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
    for _ in tqdm(range(n_runs), desc="Running simulations"):
        results = simulation.run()
        estimated_pi = 4.0 * results['mean']  # Scale by 4 since we're sampling from [-1,1]^2
        estimates.append(estimated_pi)
    
    # Calculate statistics
    mean_estimate = np.mean(estimates)
    std_error = np.std(estimates) / np.sqrt(n_runs)
    
    return mean_estimate, std_error

def main():
    # Use 1 million samples per run
    n_samples = 1_000_000
    n_runs = 5
    
    print(f"Estimating pi using {n_samples:,} samples per run, {n_runs} runs...")
    mean_estimate, std_error = estimate_pi(n_samples, n_runs)
    
    print("Results:")
    print(f"Estimated value of pi: {mean_estimate:.10f}")
    print(f"Actual value of pi:    {np.pi:.10f}")
    print(f"Absolute error:        {abs(mean_estimate - np.pi):.2e}")
    print(f"Standard error:        {std_error:.2e}")
    print(f"95% confidence interval: [{mean_estimate - 1.96*std_error:.10f}, {mean_estimate + 1.96*std_error:.10f}]")

if __name__ == "__main__":
    main() 