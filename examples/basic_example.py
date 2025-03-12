import torch
import numpy as np
from src.core import MonteCarloSimulation
from src.samplers.smart_sampler import SmartSampler

def main():
    # Define a target function (e.g., estimating the area of a unit circle)
    def target_function(samples):
        # Check if points fall within a unit circle
        radius = torch.sum(samples ** 2, dim=1)
        return (radius <= 1.0).float()
    
    # Initialize the sampler
    dimension = 2  # 2D circle
    sampler = SmartSampler(
        dimension=dimension,
        learning_rate=0.01
    )
    
    # Create and run the simulation
    simulation = MonteCarloSimulation(
        sampler=sampler,
        target_function=target_function,
        n_samples=10000,
        batch_size=100,
        use_gpu=torch.cuda.is_available(),
        convergence_threshold=1e-4
    )
    
    # Run the simulation
    results = simulation.run()
    
    # The estimated value of pi (area of unit circle = pi)
    estimated_pi = 4.0 * results['mean']  # Scale by 4 since we're sampling from [-1,1]^2
    
    print(f"Estimated value of pi: {estimated_pi:.6f}")
    print(f"Actual value of pi: {np.pi:.6f}")
    print(f"Absolute error: {abs(estimated_pi - np.pi):.6f}")
    print(f"Number of samples used: {results['n_samples']}")

if __name__ == "__main__":
    main() 