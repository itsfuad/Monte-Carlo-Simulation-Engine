import torch
import numpy as np
from typing import Union, Optional, Tuple
from ..core.base import BaseSampler
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

class StratifiedSampler(BaseSampler):
    """Stratified sampling strategy for variance reduction."""
    
    def __init__(
        self,
        dimension: int,
        strata_per_dim: int = 5,
        device: str = 'cpu'
    ):
        """Initialize stratified sampler.
        
        Args:
            dimension: Dimensionality of the sampling space
            strata_per_dim: Number of strata per dimension
            device: Device to use for computations
        """
        super().__init__(dimension, device)
        self.strata_per_dim = strata_per_dim
        self.strata_weights = torch.ones(strata_per_dim ** dimension) / (strata_per_dim ** dimension)
        
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate stratified samples.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tensor of shape (n_samples, dimension) containing the samples
        """
        # Calculate samples per stratum
        samples_per_stratum = n_samples // len(self.strata_weights)
        remainder = n_samples % len(self.strata_weights)
        
        # Generate grid of strata
        grid = torch.meshgrid(*[torch.linspace(0, 1, self.strata_per_dim) 
                               for _ in range(self.dimension)])
        strata_centers = torch.stack([g.flatten() for g in grid], dim=1)
        
        # Generate samples within each stratum
        samples = []
        stratum_size = 1.0 / self.strata_per_dim
        
        for i, center in enumerate(strata_centers):
            n_stratum_samples = samples_per_stratum + (1 if i < remainder else 0)
            if n_stratum_samples == 0:
                continue
                
            # Generate random offsets within stratum
            offsets = torch.rand(n_stratum_samples, self.dimension) * stratum_size
            stratum_samples = center + offsets - stratum_size/2
            samples.append(stratum_samples)
        
        return torch.cat(samples, dim=0)
    
    def update(self, samples: torch.Tensor, weights: torch.Tensor) -> None:
        """Update stratum weights based on sample performance.
        
        Args:
            samples: Previous samples
            weights: Importance weights for each sample
        """
        # Find which stratum each sample belongs to
        scaled_samples = samples * self.strata_per_dim
        strata_indices = torch.floor(scaled_samples).long()
        strata_indices = torch.clamp(strata_indices, 0, self.strata_per_dim - 1)
        
        # Convert multi-dimensional indices to flat indices
        flat_indices = strata_indices[:, 0]
        for d in range(1, self.dimension):
            flat_indices = flat_indices * self.strata_per_dim + strata_indices[:, d]
            
        # Update stratum weights based on sample performance
        for i in range(len(self.strata_weights)):
            stratum_mask = (flat_indices == i)
            if torch.any(stratum_mask):
                stratum_weights = weights[stratum_mask]
                self.strata_weights[i] = torch.mean(stratum_weights).item()
                
        # Normalize weights
        self.strata_weights = self.strata_weights / torch.sum(self.strata_weights) 

    def plot_strata(self, dims: Optional[Tuple[int, int]] = None) -> Figure:
        """Plot the stratification grid and samples.
        
        Args:
            dims: Tuple of (x_dim, y_dim) to plot. If None, uses first two dimensions.
            
        Returns:
            Matplotlib figure object
        """
        if dims is None:
            dims = (0, 1)
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot grid lines for strata
        for i in range(self.strata_per_dim + 1):
            ax.axvline(x=i/self.strata_per_dim, color='gray', alpha=0.3)
            ax.axhline(y=i/self.strata_per_dim, color='gray', alpha=0.3)
            
        # Get samples to plot
        samples = self.sample(1000)
        samples_np = samples.detach().cpu().numpy()
        
        # Plot samples colored by stratum weight
        scaled_samples = samples_np * self.strata_per_dim
        strata_indices = np.floor(scaled_samples).astype(int)
        strata_indices = np.clip(strata_indices, 0, self.strata_per_dim - 1)
        
        # Convert multi-dimensional indices to flat indices
        flat_indices = strata_indices[:, dims[0]]
        for d in range(1, 2):  # Only for the two plotted dimensions
            flat_indices = flat_indices * self.strata_per_dim + strata_indices[:, dims[d]]
            
        weights = self.strata_weights[flat_indices].cpu().numpy()
        
        scatter = ax.scatter(
            samples_np[:, dims[0]],
            samples_np[:, dims[1]],
            c=weights,
            cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter, label='Stratum Weight')
        
        ax.set_xlabel(f'Dimension {dims[0]}')
        ax.set_ylabel(f'Dimension {dims[1]}')
        ax.set_title('Stratified Sampling Grid and Samples')
        
        return fig
    
    def plot_stratum_weights(self) -> Figure:
        """Plot the distribution of stratum weights.
        
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        weights = self.strata_weights.cpu().numpy()
        
        if self.dimension <= 2:
            # For 1D or 2D, show as a heatmap
            weight_grid = weights.reshape(self.strata_per_dim, -1)
            sns.heatmap(weight_grid, ax=ax, cmap='viridis', annot=True)
            ax.set_title('Stratum Weights Distribution')
            
        else:
            # For higher dimensions, show as a histogram
            ax.hist(weights, bins=30, alpha=0.7)
            ax.axvline(weights.mean(), color='r', linestyle='--', 
                      label=f'Mean Weight: {weights.mean():.3f}')
            ax.set_xlabel('Stratum Weight')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Stratum Weights')
            ax.legend()
        
        return fig 