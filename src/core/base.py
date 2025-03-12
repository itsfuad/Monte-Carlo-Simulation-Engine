from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Union, Any, Dict


class BaseSampler(ABC):
    """Base class for all sampling strategies."""

    def __init__(self, dimension: int, device: str = "cpu"):
        self.dimension = dimension
        self.device = device
        self._rng = np.random.default_rng(seed=42)

    @abstractmethod
    def sample(self, n_samples: int) -> Union[np.ndarray, torch.Tensor]:
        """Generate samples from the target distribution.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Samples array of shape (n_samples, dimension)
        """
        pass

    @abstractmethod
    def update(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        weights: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Update the sampler based on previous samples and their importance weights.

        Args:
            samples: Previous samples
            weights: Importance weights for each sample
        """
        pass


class BaseOptimizer(ABC):
    """Base class for simulation optimization strategies."""

    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.iteration = 0

    @abstractmethod
    def optimize(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        values: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize the simulation parameters based on samples and their values.

        Args:
            samples: Input samples
            values: Corresponding values or rewards
            **kwargs: Additional optimization parameters

        Returns:
            Dictionary containing optimization results
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the optimizer.

        Returns:
            Dictionary containing optimizer state
        """
        pass

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the optimizer state.

        Args:
            state: Dictionary containing optimizer state
        """
        pass
