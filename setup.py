from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="spin-model-transformers",
    version="0.0.1",
    author="Matthias Bal",
    author_email="matthiascbal@gmail.com",
    description="Physics-inspired transformer modules based on mean-field dynamics of vector-spin models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcbal/spin-model-transformers",
    license="Apache-2.0",
    packages=find_packages(exclude=["examples", "notebooks", "tests"]),
    python_requires=">=3.9",
    install_requires=[
        "einops>=0.6.1",
        "equinox>=0.11.2",
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "jaxopt>=0.8",
        "numpy>=1.26",
        "optax>=0.1.7",
    ],
    extras_require={
        "dev": [
            "black~=23.9.1",
            "nbqa~=1.7",
            "pre-commit~=3.4.0",
            "ruff~=0.0.291",
        ]
    },
)
