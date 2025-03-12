import torch
import numpy as np
from typing import Dict, Any, Union
from .base import BaseOptimizer
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class AdaptiveOptimizer(BaseOptimizer):
    """Adaptive optimizer that adjusts sampling parameters based on results."""

    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        self.best_value = float("-inf")
        self.best_params = None

    def optimize(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        values: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        """Optimize sampling parameters using momentum-based updates.

        Args:
            samples: Input samples
            values: Corresponding values or rewards
            **kwargs: Additional optimization parameters

        Returns:
            Dictionary containing optimization results
        """
        # Convert inputs to torch tensors if needed
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)

        # Calculate gradient estimate
        weighted_samples = samples * values.unsqueeze(-1)
        grad = torch.mean(weighted_samples, dim=0)

        # Initialize velocity if not exists
        if self.velocity is None:
            self.velocity = torch.zeros_like(grad)

        # Update velocity using momentum
        self.velocity = self.momentum * self.velocity + self.learning_rate * grad

        # Track best parameters
        current_value = torch.mean(values).item()
        if current_value > self.best_value:
            self.best_value = current_value
            self.best_params = samples[torch.argmax(values)].clone()

        self.iteration += 1

        # Store history for visualization
        if not hasattr(self, "value_history"):
            self.value_history = []
        if not hasattr(self, "gradient_history"):
            self.gradient_history = []
        if not hasattr(self, "param_history"):
            self.param_history = []

        self.value_history.append(current_value)
        self.gradient_history.append(grad.tolist())
        if self.best_params is not None:
            self.param_history.append(self.best_params.tolist())

        return {
            "iteration": self.iteration,
            "gradient": grad.tolist(),
            "velocity": self.velocity.tolist(),
            "current_value": current_value,
            "best_value": self.best_value,
            "best_params": (
                self.best_params.tolist() if self.best_params is not None else None
            ),
        }

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the optimizer."""
        return {
            "iteration": self.iteration,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "velocity": self.velocity.tolist() if self.velocity is not None else None,
            "best_value": self.best_value,
            "best_params": (
                self.best_params.tolist() if self.best_params is not None else None
            ),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the optimizer state."""
        self.iteration = state["iteration"]
        self.learning_rate = state["learning_rate"]
        self.momentum = state["momentum"]
        self.velocity = (
            torch.tensor(state["velocity"]) if state["velocity"] is not None else None
        )
        self.best_value = state["best_value"]
        self.best_params = (
            torch.tensor(state["best_params"])
            if state["best_params"] is not None
            else None
        )

    def plot_optimization_history(self) -> Figure:
        """Plot the optimization history.

        Returns:
            Matplotlib figure object
        """
        if self.iteration == 0:
            raise ValueError("No optimization history available yet")

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot value evolution
        if hasattr(self, "value_history"):
            values = self.value_history
            axes[0].plot(values, "b-", label="Current Value")
            axes[0].plot(np.maximum.accumulate(values), "r--", label="Best Value")
            axes[0].set_xlabel("Iteration")
            axes[0].set_ylabel("Value")
            axes[0].set_title("Optimization Progress")
            axes[0].legend()

        # Plot parameter evolution if available
        if hasattr(self, "param_history"):
            params = np.array(self.param_history)
            for i in range(params.shape[1]):
                axes[1].plot(params[:, i], label=f"Param {i}")
            axes[1].set_xlabel("Iteration")
            axes[1].set_ylabel("Parameter Value")
            axes[1].set_title("Parameter Evolution")
            axes[1].legend()

        plt.tight_layout()
        return fig

    def plot_gradient_history(self) -> Figure:
        """Plot the gradient history.

        Returns:
            Matplotlib figure object
        """
        if not hasattr(self, "gradient_history"):
            raise ValueError("No gradient history available")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot gradient magnitudes
        grads = np.array(self.gradient_history)
        grad_norms = np.linalg.norm(grads, axis=1)

        ax1.plot(grad_norms, "b-")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Gradient Norm")
        ax1.set_title("Gradient Magnitude Evolution")

        # Plot gradient components
        for i in range(grads.shape[1]):
            ax2.plot(grads[:, i], label=f"Dim {i}")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Gradient Value")
        ax2.set_title("Gradient Components Evolution")
        ax2.legend()

        plt.tight_layout()
        return fig
