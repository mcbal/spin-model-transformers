from setuptools import find_packages, setup

setup(
    name="spin-glass-transformers",
    packages=find_packages(exclude=["examples"]),
    version="0.0.1",
    license="MIT",
    description="Implementation of spin-glass transformers in JAX",
    author="Matthias Bal",
    author_email="matthiascbal@gmail.com",
    url="https://github.com/mcbal/spin-glass-transformers",
    install_requires=[
        "einops>=0.6",
        "equinox>=0.9.2",
        "jax>=0.3.25",
        "jaxlib>=0.3.25",
        "jaxopt>=0.5.5",
        "numpy",
        "optax>=0.1.4",
    ],
)
