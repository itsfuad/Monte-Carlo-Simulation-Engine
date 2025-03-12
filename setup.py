from setuptools import setup, find_packages

setup(
    name="mc_sim_engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "scipy",
        "matplotlib",
        "tqdm"
    ],
) 