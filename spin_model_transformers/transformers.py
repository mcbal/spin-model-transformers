# Copyright 2023 Matthias Bal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Type

import jax
import jax.numpy as jnp
from equinox import Module


#####################################################################################
# Transformer model sequentially applying modules of same type using `jax.lax.scan`.
#####################################################################################


class Transformer(Module):
    layers: Module

    def __init__(
        self, layer_cls: Type[Module], *, dim, dim_head, num_heads, depth, key
    ):
        layer_keys = jax.random.split(key, depth)

        self.layers = jax.tree_map(
            lambda *xs: jnp.stack(xs),
            *[
                layer_cls(
                    dim=dim, dim_head=dim_head, num_heads=num_heads, key=layer_keys[i]
                )
                for i in range(depth)
            ],
        )

    def __call__(self, x, mask=None):
        def apply_scan_fn(x, layer):
            return layer(x, mask=mask), None

        return jax.lax.scan(apply_scan_fn, x, xs=self.layers)[0]
