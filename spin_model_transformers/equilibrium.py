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


import jax
import jax.numpy as jnp
from einops import rearrange
from equinox import Module, nn, static_field


##############################################################################
# Vector-spin model with diagonal coupling matrix (special case of symmetric).
##############################################################################


def safe_log(x, eps=1e-8):
    return jnp.log(x + eps)


def safe_reciprocal(x, eps=1e-8):
    return jax.lax.reciprocal(jax.lax.clamp(eps, x, jnp.finfo(x.dtype).max))


def phi_diag(t, h, J, beta):
    v = t - J
    return (
        beta * jnp.sum(t, axis=-1)
        - 0.5 * safe_log(v).sum(axis=-1)
        + 0.25
        * beta
        * jnp.einsum("... i f, ... i, ... i f -> ...", h, safe_reciprocal(v), h)
    )


def log_Z_diag(t, h, J, beta):
    return -0.5 * h.shape[-2] * (1.0 + jnp.log(2.0 * beta)) + phi_diag(t, h, J, beta)


def t_star_diag(h, J, beta):
    a = beta
    b = -0.5 - 2 * beta * J
    c = (
        beta * J**2
        + 0.5 * J
        - 0.25 * beta * jnp.einsum("... i f, ... i f -> ... i", h, h)
    )
    return (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)


class DiagonalVectorSpinGlassAttention(Module):
    dim: int = static_field()
    dim_head: int = static_field()
    num_heads: int = static_field()
    beta: float = static_field()
    to_qk: Module

    def __init__(
        self,
        *,
        dim,
        dim_head,
        num_heads,
        key,
        beta=1.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.beta = beta

        self.to_qk = nn.Linear(dim, 2 * dim_head * num_heads, use_bias=False, key=key)

    def _J(self, x, mask=None):
        x = rearrange(x, "...  h n d -> ... n (h d)", h=self.num_heads)

        q, k = jnp.split(jax.vmap(self.to_qk)(x), 2, axis=-1)
        q, k = map(
            lambda t: rearrange(t, "... n (h d) -> ... h n d", h=self.num_heads), (q, k)
        )

        sim = jnp.einsum("... i d, ... j d -> ... i j", q, k)

        if mask is not None:
            sim = jnp.where(mask, sim, jnp.finfo(sim.dtype).min)

        return jax.scipy.special.logsumexp(sim, axis=-1)

    def _multi_head_free_energy(self, x, mask, beta):
        def _free_energy(x, J, beta):
            return -log_Z_diag(t_star_diag(x, J, beta)(x, J, beta), x, J, beta) / beta

        return jax.vmap(_free_energy, in_axes=(0, 0, None))(
            x, self._J(x, mask=mask), beta
        )

    def __call__(self, x, mask):
        x = rearrange(x, "...  n (h d) -> ... h n d", h=self.num_heads, d=self.dim_head)
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)

        return -rearrange(
            jnp.diagonal(
                jax.jacrev(self._multi_head_free_energy, argnums=0)(
                    x, mask=mask, beta=self.beta
                )
            ),
            "... n d h -> ... n (h d)",
            h=self.num_heads,
        )


# TODO: Add equilibrium module for vector-spin model with symmetric couplings

# TODO: Add equilibrium module for adaptive TAP-style mean-field equations, see
#       https://github.com/mcbal/deep-implicit-attention
