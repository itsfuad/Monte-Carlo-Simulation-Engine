import torch
import numpy as np
from typing import Union, Optional, Tuple, List
from ..core.base import BaseSampler
from torch.distributions import MultivariateNormal, Normal
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

class SmartSampler(BaseSampler):
    """Smart sampling strategy using adaptive importance sampling with neural network guidance."""
    
    def __init__(
        self,
        dimension: int,
        device: str = 'cpu',
        learning_rate: float = 0.01,
        initial_mean: Optional[Union[np.ndarray, torch.Tensor]] = None,
        initial_std: Optional[Union[np.ndarray, torch.Tensor]] = None
    ):
        """Initialize smart sampler.
        
        Args:
            dimension: Dimensionality of the sampling space
            device: Device to use for computations
            learning_rate: Learning rate for adapting distributions
            initial_mean: Initial mean for each dimension
            initial_std: Initial standard deviation for each dimension
        """
        super().__init__(dimension, device)
        
        # Initialize distribution parameters
        self.means = (torch.zeros(dimension) if initial_mean is None 
                     else torch.as_tensor(initial_mean))
        self.stds = (torch.ones(dimension) if initial_std is None 
                    else torch.as_tensor(initial_std))
        
        self.learning_rate = learning_rate
        self.distributions = [Normal(mean, std) for mean, std in zip(self.means, self.stds)]
        
        # Keep track of best performing regions
        self.best_samples = None
        self.best_weights = None
        self.history = []
        
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate samples using current distributions.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tensor of shape (n_samples, dimension) containing the samples
        """
        samples = []
        
        # Generate samples for each dimension
        for dist in self.distributions:
            dim_samples = dist.sample((n_samples,))
            samples.append(dim_samples)
        
        # Combine samples from all dimensions
        samples = torch.stack(samples, dim=1)
        
        # Mix in some samples from best performing regions if available
        if self.best_samples is not None and len(self.best_samples) > 0:
            n_best = min(n_samples // 4, len(self.best_samples))
            indices = torch.randperm(len(self.best_samples))[:n_best]
            best_samples = self.best_samples[indices]
            
            # Add random noise to best samples
            noise = torch.randn_like(best_samples) * 0.1
            best_samples = best_samples + noise
            
            # Replace some random samples with best samples
            samples[:n_best] = best_samples
        
        return samples
    
    def update(self, samples: torch.Tensor, weights: torch.Tensor) -> None:
        """Update sampling distributions based on performance.
        
        Args:
            samples: Previous samples of shape (n_samples, dimension)
            weights: Importance weights for each sample
        """
        # Normalize weights
        weights = weights / weights.sum()
        
        # Update means and standard deviations for each dimension
        for dim in range(self.dimension):
            # Compute weighted statistics
            weighted_mean = torch.sum(weights * samples[:, dim])
            weighted_var = torch.sum(weights * (samples[:, dim] - weighted_mean) ** 2)
            weighted_std = torch.sqrt(weighted_var + 1e-8)
            
            # Update parameters with moving average
            self.means[dim] = ((1 - self.learning_rate) * self.means[dim] + 
                             self.learning_rate * weighted_mean)
            self.stds[dim] = ((1 - self.learning_rate) * self.stds[dim] + 
                            self.learning_rate * weighted_std)
            
            # Update distribution
            self.distributions[dim] = Normal(self.means[dim], self.stds[dim])
        
        # Track best performing samples
        n_best = min(100, len(samples))
        best_indices = torch.argsort(weights, descending=True)[:n_best]
        self.best_samples = samples[best_indices].clone()
        self.best_weights = weights[best_indices].clone()
        
        # Store history
        self.history.append({
            'mean': self.means.clone(),
            'std': self.stds.clone(),
            'best_weight': torch.max(weights).item()
        })
    
    def get_distribution_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current distribution parameters.
        
        Returns:
            Tuple of (means, standard deviations)
        """
        return self.means.clone(), self.stds.clone()
    
    def get_best_samples(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get best performing samples and their weights.
        
        Returns:
            Tuple of (samples, weights) or None if no samples yet
        """
        if self.best_samples is None:
            return None
        return self.best_samples.clone(), self.best_weights.clone()
    
    def plot_sampling_distribution(self, dims: Optional[Tuple[int, int]] = None) -> Figure:
        """Plot the current sampling distribution and best samples.
        
        Args:
            dims: Tuple of (x_dim, y_dim) to plot. If None, uses first two dimensions.
            
        Returns:
            Matplotlib figure object
        """
        if dims is None:
            dims = (0, 1)
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert means and stds to numpy arrays
        means = self.means.detach().cpu().numpy()
        stds = self.stds.detach().cpu().numpy()
        
        # Plot best samples if available
        if self.best_samples is not None:
            best_samples = self.best_samples.detach().cpu().numpy()
            weights = self.best_weights.detach().cpu().numpy()
            
            scatter = ax.scatter(
                best_samples[:, dims[0]],
                best_samples[:, dims[1]],
                c=weights,
                cmap='viridis',
                alpha=0.6,
                label='Best Samples'
            )
            plt.colorbar(scatter, label='Sample Weights')
        
        # Plot distribution contours
        x = np.linspace(
            means[dims[0]] - 3*stds[dims[0]],
            means[dims[0]] + 3*stds[dims[0]],
            100
        )
        y = np.linspace(
            means[dims[1]] - 3*stds[dims[1]],
            means[dims[1]] + 3*stds[dims[1]],
            100
        )
        X, Y = np.meshgrid(x, y)
        
        # Create grid of points
        pos = np.dstack((X, Y))
        
        # Calculate PDF values
        Z = np.exp(-0.5 * (
            ((pos[:,:,0] - means[dims[0]]) / stds[dims[0]]) ** 2 +
            ((pos[:,:,1] - means[dims[1]]) / stds[dims[1]]) ** 2
        ))
        
        # Plot contours
        ax.contour(X, Y, Z)
        ax.set_xlabel(f'Dimension {dims[0]}')
        ax.set_ylabel(f'Dimension {dims[1]}')
        ax.set_title('Sampling Distribution')
        
        return fig
    
    def plot_convergence_history(self) -> Figure:
        """Plot the convergence history of the sampler.
        
        Returns:
            Matplotlib figure object
        """
        if not self.history:
            raise ValueError("No history available yet")
            
        means = torch.stack([h['mean'] for h in self.history])
        stds = torch.stack([h['std'] for h in self.history])
        best_weights = torch.tensor([h['best_weight'] for h in self.history])
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot mean evolution
        for d in range(self.dimension):
            axes[0].plot(means[:, d].cpu(), label=f'Dim {d}')
        axes[0].set_title('Evolution of Means')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Mean Value')
        axes[0].legend()
        
        # Plot std evolution
        for d in range(self.dimension):
            axes[1].plot(stds[:, d].cpu(), label=f'Dim {d}')
        axes[1].set_title('Evolution of Standard Deviations')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Standard Deviation')
        axes[1].legend()
        
        # Plot best weights evolution
        axes[2].plot(best_weights.cpu(), 'r-', label='Best Weight')
        axes[2].set_title('Evolution of Best Sample Weight')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Weight Value')
        axes[2].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_sample_history(self, n_iterations: int = 5) -> Figure:
        """Plot the evolution of sampling distribution over time.
        
        Args:
            n_iterations: Number of past iterations to plot
            
        Returns:
            Matplotlib figure object
        """
        if not self.history:
            raise ValueError("No history available yet")
            
        history_length = len(self.history)
        plot_indices = np.linspace(0, history_length-1, n_iterations, dtype=int)
        
        fig, axes = plt.subplots(1, n_iterations, figsize=(4*n_iterations, 4))
        if n_iterations == 1:
            axes = [axes]
            
        for i, idx in enumerate(plot_indices):
            h = self.history[idx]
            mean = h['mean']
            std = h['std']
            
            # Create grid for first two dimensions
            x = np.linspace(mean[0] - 3*std[0], mean[0] + 3*std[0], 100)
            y = np.linspace(mean[1] - 3*std[1], mean[1] + 3*std[1], 100)
            X, Y = np.meshgrid(x, y)
            pos = np.dstack((X, Y))
            
            # Calculate PDF
            Z = np.exp(-0.5 * (
                ((pos[:,:,0] - mean[0]) / std[0]) ** 2 +
                ((pos[:,:,1] - mean[1]) / std[1]) ** 2
            ))
            
            # Plot
            axes[i].contour(X, Y, Z, levels=10, alpha=0.5, colors='k')
            axes[i].plot(mean[0].item(), mean[1].item(), 'r*', markersize=10)
            axes[i].set_title(f'Iteration {idx}')
            
        plt.tight_layout()
        return fig 