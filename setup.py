from setuptools import find_packages, setup

setup(
    name="ising-transformers-jax",
    packages=find_packages(exclude=["examples", "notebooks"]),
    version="0.0.1",
    license="MIT",
    description="Ising Transformers in JAX",
    author="Matthias Bal",
    author_email="matthiascbal@gmail.com",
    url="https://github.com/mcbal/ising-transformers-jax",
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
