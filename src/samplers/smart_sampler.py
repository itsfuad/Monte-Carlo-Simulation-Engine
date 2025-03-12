import numpy as np
import torch
from typing import Union, Optional
from ..core.base import BaseSampler
from torch.distributions import MultivariateNormal

class SmartSampler(BaseSampler):
    """Smart sampling strategy using adaptive importance sampling."""
    
    def __init__(
        self,
        dimension: int,
        device: str = 'cpu',
        learning_rate: float = 0.01,
        initial_mean: Optional[Union[np.ndarray, torch.Tensor]] = None,
        initial_cov: Optional[Union[np.ndarray, torch.Tensor]] = None
    ):
        """Initialize the smart sampler.
        
        Args:
            dimension: Dimensionality of the sampling space
            device: Device to use for computations
            learning_rate: Learning rate for adapting the proposal distribution
            initial_mean: Initial mean for the proposal distribution
            initial_cov: Initial covariance for the proposal distribution
        """
        super().__init__(dimension, device)
        
        # Initialize proposal distribution parameters
        self.mean = (torch.zeros(dimension) if initial_mean is None 
                    else torch.as_tensor(initial_mean))
        self.cov = (torch.eye(dimension) if initial_cov is None 
                   else torch.as_tensor(initial_cov))
        
        self.learning_rate = learning_rate
        self.proposal = MultivariateNormal(self.mean, self.cov)
        
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate samples from the current proposal distribution.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tensor of shape (n_samples, dimension) containing the samples
        """
        return self.proposal.sample((n_samples,))
    
    def update(self, samples: torch.Tensor, weights: torch.Tensor) -> None:
        """Update the proposal distribution based on weighted samples.
        
        Args:
            samples: Previous samples of shape (n_samples, dimension)
            weights: Importance weights for each sample of shape (n_samples,)
        """
        # Normalize weights
        weights = weights / weights.sum()
        
        # Update mean
        new_mean = torch.sum(weights.unsqueeze(1) * samples, dim=0)
        
        # Update covariance
        centered = samples - new_mean.unsqueeze(0)
        new_cov = torch.mm(
            (weights.unsqueeze(1) * centered).T,
            centered
        )
        
        # Apply moving average update
        self.mean = ((1 - self.learning_rate) * self.mean + 
                    self.learning_rate * new_mean)
        self.cov = ((1 - self.learning_rate) * self.cov + 
                   self.learning_rate * new_cov)
        
        # Ensure covariance remains positive definite
        self.cov = (self.cov + self.cov.T) / 2  # Ensure symmetry
        min_eigenval = torch.min(torch.linalg.eigvals(self.cov).real)
        if min_eigenval < 1e-6:
            self.cov += (1e-6 - min_eigenval) * torch.eye(self.dimension)
        
        # Update proposal distribution
        self.proposal = MultivariateNormal(self.mean, self.cov)
    
    def get_log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        """Compute log probability of samples under the current proposal.
        
        Args:
            samples: Samples to evaluate of shape (n_samples, dimension)
            
        Returns:
            Log probabilities of shape (n_samples,)
        """
        return self.proposal.log_prob(samples) 