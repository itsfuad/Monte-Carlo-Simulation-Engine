import numpy as np
import torch
from typing import Optional, Callable, Dict, Any
from .base import BaseSampler, BaseOptimizer
import logging
from tqdm import tqdm

class MonteCarloSimulation:
    """AI-optimized Monte Carlo simulation engine."""
    
    def __init__(
        self,
        sampler: BaseSampler,
        target_function: Optional[Callable] = None,
        optimizer: Optional[BaseOptimizer] = None,
        n_samples: int = 10000,
        batch_size: int = 100,
        use_gpu: bool = False,
        seed: Optional[int] = None,
        convergence_threshold: float = 1e-6,
        max_iterations: int = 100
    ):
        """Initialize the Monte Carlo simulation.
        
        Args:
            sampler: Sampling strategy to use
            target_function: Function to evaluate samples
            optimizer: Optimization strategy for improving sampling
            n_samples: Total number of samples to generate
            batch_size: Number of samples per batch
            use_gpu: Whether to use GPU acceleration
            seed: Random seed for reproducibility
            convergence_threshold: Threshold for early stopping
            max_iterations: Maximum number of iterations
        """
        self.sampler = sampler
        self.target_function = target_function
        self.optimizer = optimizer
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the simulation."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def run(self) -> Dict[str, Any]:
        """Run the Monte Carlo simulation.
        
        Returns:
            Dictionary containing simulation results and statistics
        """
        self.logger.info("Starting Monte Carlo simulation...")
        
        results = {
            'samples': [],
            'values': [],
            'convergence': [],
            'optimization_history': [],
            'running_means': []
        }
        
        n_batches = self.n_samples // self.batch_size
        min_batches = max(20, n_batches // 5)  # At least 20 batches or 20% of total
        running_mean = None
        best_mean = None
        best_samples = []
        best_values = []
        
        with tqdm(total=n_batches, desc="Simulation Progress") as pbar:
            for i in range(n_batches):
                # Generate samples
                batch_samples = self.sampler.sample(self.batch_size)
                
                # Evaluate samples
                if self.target_function is not None:
                    batch_values = self.target_function(batch_samples)
                else:
                    batch_values = torch.ones(self.batch_size)
                
                # Update results
                results['samples'].append(batch_samples)
                results['values'].append(batch_values)
                
                # Calculate running mean
                batch_mean = torch.mean(batch_values)
                if running_mean is None:
                    running_mean = batch_mean
                else:
                    running_mean = (running_mean * i + batch_mean) / (i + 1)
                results['running_means'].append(running_mean.item())
                
                # Track best results
                if best_mean is None or abs(batch_mean - np.pi/4) < abs(best_mean - np.pi/4):
                    best_mean = batch_mean
                    best_samples = batch_samples
                    best_values = batch_values
                
                # Optimize sampling strategy
                if self.optimizer is not None:
                    opt_result = self.optimizer.optimize(
                        batch_samples, batch_values
                    )
                    results['optimization_history'].append(opt_result)
                    
                    # Update sampler with optimization results and best samples
                    self.sampler.update(
                        torch.cat([batch_samples, best_samples]) if len(best_samples) > 0 else batch_samples,
                        torch.cat([batch_values, best_values]) if len(best_values) > 0 else batch_values
                    )
                
                # Check convergence only after minimum batches
                if i >= min_batches:
                    convergence = self._check_convergence(results)
                    results['convergence'].append(convergence)
                    
                    if convergence < self.convergence_threshold:
                        self.logger.info(f"Converged after {i+1} batches")
                        break
                else:
                    results['convergence'].append(float('inf'))
                
                pbar.update(1)
        
        # Finalize results using all samples
        results = self._finalize_results(results)
        self.logger.info("Simulation completed successfully")
        
        return results
    
    def _check_convergence(self, results: Dict[str, Any]) -> float:
        """Check convergence of the simulation.
        
        Args:
            results: Current simulation results
            
        Returns:
            Convergence metric value
        """
        if len(results['running_means']) < 10:
            return float('inf')
            
        # Use last 10 running means to check convergence
        recent_means = results['running_means'][-10:]
        mean_diff = max(abs(recent_means[-1] - m) for m in recent_means[:-1])
        
        return mean_diff
    
    def _finalize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and finalize simulation results.
        
        Args:
            results: Raw simulation results
            
        Returns:
            Processed results with statistics
        """
        # Concatenate all samples and values
        all_samples = torch.cat(results['samples'])
        all_values = torch.cat(results['values'])
        
        final_results = {
            'samples': all_samples,
            'values': results['values'],  # Keep original list of batches for pi estimation
            'mean': torch.mean(all_values).item(),
            'std': torch.std(all_values, dim=0).item(),
            'n_samples': len(all_values),
            'convergence': results['convergence'],
            'optimization_history': results['optimization_history'],
            'running_means': results['running_means']
        }
        
        return final_results 