# AI-Optimized Monte Carlo Simulation Engine

A high-performance Monte Carlo simulation engine that leverages AI techniques for improved sampling and convergence.

## Features

- Smart sampling strategies with AI-guided importance sampling
- Stratified sampling for variance reduction
- Quasi-random sampling using Sobol sequences
- Adaptive optimization for parameter tuning
- Visualization tools for sampling and convergence analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MC-Sim-Engine
```

2. Install dependencies:
```bash
pip install -e .
```

## Usage

### Basic Example
```python
from src.core import MonteCarloSimulation
from src.samplers import SmartSampler

# Define your target function
def target_function(samples):
    return (samples ** 2).sum(dim=1) <= 1.0

# Create simulation
sampler = SmartSampler(dimension=2)
simulation = MonteCarloSimulation(
    sampler=sampler,
    target_function=target_function,
    n_samples=10000
)

# Run simulation
results = simulation.run()
```

### Available Scripts

- `run_tests.bat`: Run all tests
- `run_example.bat`: Run basic example
- `setup_env.bat`: Set up development environment

## Testing

Run tests using:
```bash
pytest tests/
```

## License

MIT License