# AI-Optimized Monte Carlo Simulation Engine

A high-performance Monte Carlo simulation engine with adaptive sampling strategies and optimization techniques. The engine supports multiple sampling methods, adaptive optimization, and provides comprehensive visualization tools for monitoring simulation progress.

## Features

- Multiple sampling strategies:
  - `SmartSampler`: Adaptive importance sampling with best-sample tracking
  - `StratifiedSampler`: Variance reduction through stratified sampling
  - `QuasiRandomSampler`: Low-discrepancy sequences using Sobol points
- Adaptive optimization with gradient-based updates
- Real-time visualization tools for:
  - Sampling distributions
  - Convergence monitoring
  - Distribution evolution
  - Optimization progress
- Comprehensive test suite with high coverage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mc-sim-engine.git
cd mc-sim-engine
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Unix/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage Example

```python
import torch
from src.core import MonteCarloSimulation
from src.samplers import SmartSampler
from src.core.optimizers import AdaptiveOptimizer

# Define target function (e.g., estimating π)
def target_function(samples):
    return (samples ** 2).sum(dim=1) <= 1.0

# Initialize components
sampler = SmartSampler(dimension=2)
optimizer = AdaptiveOptimizer(learning_rate=0.001)

# Create simulation
simulation = MonteCarloSimulation(
    sampler=sampler,
    target_function=target_function,
    optimizer=optimizer,
    n_samples=10000,
    batch_size=100,
    convergence_threshold=1e-6
)

# Run simulation
results = simulation.run()

# Access results
print(f"Estimated π: {4 * results['mean']:.6f}")
print(f"Number of samples: {results['n_samples']}")
print(f"Standard deviation: {results['std']:.6f}")
```

## Running Tests

Run the full test suite (Windows):
```bash
.\run_tests.bat
```

Or run specific components:
```bash
# Run all tests with coverage
pytest tests/test_simulation.py -v "--cov=src" "--cov-report=term-missing"

# Run specific test file
pytest tests/test_simulation.py -v
```

## Code Quality

Format code:
```bash
black src tests
```

Run linting:
```bash
flake8 src tests
```

Run type checking:
```bash
mypy src tests
```

## Project Structure

```
mc-sim-engine/
├── src/
│   ├── core/
│   │   ├── base.py          # Abstract base classes
│   │   ├── simulation.py    # Main simulation engine
│   │   └── optimizers.py    # Optimization strategies
│   └── samplers/
│       ├── smart_sampler.py      # Adaptive importance sampling
│       ├── stratified_sampler.py # Stratified sampling
│       └── quasi_random_sampler.py # Sobol sequence sampling
├── tests/
│   └── test_simulation.py   # Test suite
├── requirements.txt         # Project dependencies
├── setup.cfg               # Tool configurations
├── run_tests.bat          # Windows test runner
└── README.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.