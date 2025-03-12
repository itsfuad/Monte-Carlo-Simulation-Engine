import torch
from typing import Optional, Tuple
from ..core.base import BaseSampler
from scipy.stats import qmc
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class QuasiRandomSampler(BaseSampler):
    """Quasi-random sampling using Sobol sequences."""

    def __init__(self, dimension: int, device: str = "cpu", scramble: bool = True):
        """Initialize quasi-random sampler.

        Args:
            dimension: Dimensionality of the sampling space
            device: Device to use for computations
            scramble: Whether to use Owen scrambling
        """
        super().__init__(dimension, device)
        self.scramble = scramble
        self.sampler = qmc.Sobol(d=dimension, scramble=scramble, seed=42)
        self.current_index = 0

    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate quasi-random samples using Sobol sequence.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tensor of shape (n_samples, dimension) containing the samples
        """
        # Generate Sobol sequence samples
        samples = self.sampler.random(n=n_samples)
        self.current_index += n_samples

        # Convert to torch tensor
        return torch.from_numpy(samples).float().to(self.device)

    def update(self, samples: torch.Tensor, weights: torch.Tensor) -> None:
        """Update method (no-op for quasi-random sampling).

        Args:
            samples: Previous samples
            weights: Importance weights for each sample
        """
        # Quasi-random sequences don't need updating
        pass

    def reset(self) -> None:
        """Reset the sampler to start sequence from beginning."""
        self.current_index = 0
        self.sampler = qmc.Sobol(d=self.dimension, scramble=self.scramble, seed=42)

    def plot_sequence(
        self, n_points: int = 1000, dims: Optional[Tuple[int, int]] = None
    ) -> Figure:
        """Plot the Sobol sequence points.

        Args:
            n_points: Number of points to plot
            dims: Tuple of (x_dim, y_dim) to plot. If None, uses first two dimensions.

        Returns:
            Matplotlib figure object
        """
        if dims is None:
            dims = (0, 1)

        # Generate samples
        samples = self.sample(n_points)
        samples_np = samples.cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot Sobol sequence
        ax1.scatter(samples_np[:, dims[0]], samples_np[:, dims[1]], alpha=0.6, s=20)
        ax1.set_title("Sobol Sequence")
        ax1.set_xlabel(f"Dimension {dims[0]}")
        ax1.set_ylabel(f"Dimension {dims[1]}")

        # Plot random uniform for comparison
        random_samples = self._rng.random((n_points, 2))
        ax2.scatter(random_samples[:, 0], random_samples[:, 1], alpha=0.6, s=20)
        ax2.set_title("Random Uniform")
        ax2.set_xlabel(f"Dimension {dims[0]}")
        ax2.set_ylabel(f"Dimension {dims[1]}")

        plt.suptitle("Quasi-Random vs Random Sampling")
        plt.tight_layout()

        return fig

    def plot_discrepancy(self, max_points: int = 1000, step: int = 50) -> Figure:
        """Plot the star discrepancy evolution.

        Args:
            max_points: Maximum number of points to evaluate
            step: Step size for evaluation

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        points = range(step, max_points + 1, step)
        discrepancies = []
        random_discrepancies = []

        for n in points:
            # Quasi-random samples
            samples = self.sample(n)
            samples_np = samples.cpu().numpy()
            disc = qmc.discrepancy(samples_np)
            discrepancies.append(disc)

            # Random samples
            random_samples = self._rng.random(size=(n, self.dimension))
            random_disc = qmc.discrepancy(random_samples)
            random_discrepancies.append(random_disc)

        ax.plot(points, discrepancies, "b-", label="Sobol Sequence")
        ax.plot(points, random_discrepancies, "r--", label="Random Uniform")

        ax.set_xlabel("Number of Points")
        ax.set_ylabel("Star Discrepancy")
        ax.set_title("Discrepancy Comparison")
        ax.legend()

        return fig
