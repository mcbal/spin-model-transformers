from typing import Type

import jax
import jax.numpy as jnp

from equinox import Module


class Transformer(Module):

    layers: Module

    def __init__(self, layer_cls: Type[Module], *, dim, dim_head, num_heads, depth, key):
        layer_keys = jax.random.split(key, depth)

        self.layers = jax.tree_map(
            lambda *xs: jnp.stack(xs),
            *[layer_cls(dim=dim, dim_head=dim_head, num_heads=num_heads, key=layer_keys[i]) for i in range(depth)],
        )

    def __call__(self, x, mask=None):
        def apply_scan_fn(x, layer):
            return layer(x, mask=mask), None

        return jax.lax.scan(apply_scan_fn, x, xs=self.layers)[0]
