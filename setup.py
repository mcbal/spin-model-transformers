from setuptools import find_packages, setup

setup(
    name="ising-transformers-jax",
    packages=find_packages(exclude=[""]),
    version="0.0.0",
    license="Apache License 2.0",
    description="Ising Transformers in JAX",
    author="Matthias Bal",
    author_email="matthiascbal@gmail.com",
    long_description_content_type="text/markdown",
    url="https://github.com/mcbal/ising-transformers-jax",
    install_requires=[
        "einops>=0.4",
        "equinox>=0.5",
        "jax>=0.3.13",
        "jaxlib>=0.3.10",
        "jaxopt>=0.4.2",
        "numpy>=1.19",
        "optax>=0.1.2",
    ],
)
