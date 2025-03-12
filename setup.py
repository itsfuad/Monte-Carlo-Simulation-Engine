from setuptools import setup, find_packages

setup(
    name="mc-sim-engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "scipy",
        "seaborn",
        "tqdm"
    ],
    python_requires=">=3.8",
) 