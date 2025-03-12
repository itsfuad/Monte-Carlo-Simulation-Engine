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
            'optimization_history': []
        }
        
        n_batches = self.n_samples // self.batch_size
        min_batches = max(10, n_batches // 10)  # At least 10 batches or 10% of total
        
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
                
                # Optimize sampling strategy
                if self.optimizer is not None:
                    opt_result = self.optimizer.optimize(
                        batch_samples, batch_values
                    )
                    results['optimization_history'].append(opt_result)
                    
                    # Update sampler with optimization results
                    self.sampler.update(batch_samples, batch_values)
                
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
        
        # Finalize results
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
        values = torch.cat(results['values'])
        window_size = 5 * self.batch_size
        
        # Use multiple windows for more stable convergence check
        if len(values) < 3 * window_size:
            return float('inf')
            
        # Calculate means of three consecutive windows
        mean1 = torch.mean(values[-3*window_size:-2*window_size])
        mean2 = torch.mean(values[-2*window_size:-window_size])
        mean3 = torch.mean(values[-window_size:])
        
        # Check both recent change and trend
        recent_change = abs(mean3 - mean2)
        trend_consistency = abs(mean2 - mean1)
        
        return max(recent_change, trend_consistency)
    
    def _finalize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and finalize simulation results.
        
        Args:
            results: Raw simulation results
            
        Returns:
            Processed results with statistics
        """
        samples = torch.cat(results['samples'])
        values = torch.cat(results['values'])
        
        final_results = {
            'samples': samples,
            'mean': torch.mean(values, dim=0).item(),
            'std': torch.std(values, dim=0).item(),
            'n_samples': len(values),
            'convergence_history': results['convergence'],
            'optimization_history': results['optimization_history']
        }
        
        return final_results 