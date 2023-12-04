# Spin-model transformers


## Install

```bash
pip install -e .[dev]
pre-commit install
pre-commit run --all-files
```

## Examples

```python
import jax
from spin_model_transformers import SpinTransformer


key = jax.random.PRNGKey(2666)
x_key, mod_key = jax.random.split(key)

x = jax.random.normal(x_key, shape=(1, 256, 512))
transformer = SpinTransformer(depth=6, dim=512, num_heads=1, beta=1.0, key=mod_key)

out = jax.vmap(transformer)(x)  # (1, 256, 512)
```
