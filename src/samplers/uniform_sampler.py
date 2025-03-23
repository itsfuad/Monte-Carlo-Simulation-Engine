from typing import Union
import torch
from src.core.base import BaseSampler


class UniformSampler(BaseSampler):
    """Uniform sampling strategy within specified bounds."""

    def __init__(
        self,
        dimension: int,
        lower_bound: Union[float, torch.Tensor] = -1.0,
        upper_bound: Union[float, torch.Tensor] = 1.0,
        device: str = "cpu",
    ):
        """Initialize uniform sampler.

        Args:
            dimension: Dimensionality of the sampling space
            lower_bound: Lower bound for each dimension
            upper_bound: Upper bound for each dimension
            device: Device to use for computations
        """
        super().__init__(dimension, device)

        # Convert bounds to tensors if they aren't already
        self.lower_bound = torch.as_tensor(lower_bound, device=device)
        self.upper_bound = torch.as_tensor(upper_bound, device=device)

        # Expand bounds if they're scalars
        if self.lower_bound.dim() == 0:
            self.lower_bound = self.lower_bound.expand(dimension)
        if self.upper_bound.dim() == 0:
            self.upper_bound = self.upper_bound.expand(dimension)

    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate samples using uniform distribution.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tensor of shape (n_samples, dimension) containing the samples
        """
        # Generate uniform random numbers in [0, 1]
        samples = torch.rand(n_samples, self.dimension, device=self.device)

        # Scale and shift to desired bounds
        samples = self.lower_bound + (self.upper_bound - self.lower_bound) * samples

        return samples

    def update(self, samples: torch.Tensor, weights: torch.Tensor) -> None:
        """No-op since uniform sampling doesn't need updates."""
