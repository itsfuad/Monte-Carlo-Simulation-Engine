# AI-Optimized Monte Carlo Simulation Engine

A high-performance Monte Carlo simulation engine with adaptive sampling strategies and optimization techniques.

## Features

- Multiple sampling strategies:
  - Smart Sampler with adaptive importance sampling
  - Stratified Sampler for variance reduction
  - Quasi-Random Sampler using Sobol sequences
- Adaptive optimization with gradient-based updates
- Visualization tools for sampling distributions and convergence
- Comprehensive test suite with high coverage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mc-sim-engine.git
cd mc-sim-engine
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Running Tests

Run the full test suite with coverage:
```bash
pytest tests/test_simulation.py -v --cov=src --cov-report=term-missing
```

Run specific test files:
```bash
pytest tests/test_simulation.py -v
```

## Code Quality

Check code formatting:
```bash
black --check src tests
```

Run linting:
```bash
flake8 src tests
```

Run type checking:
```bash
mypy src tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.