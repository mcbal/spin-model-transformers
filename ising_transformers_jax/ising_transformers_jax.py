import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from einops import rearrange


def _phi(t, h, J, beta):
    V = jnp.diag(t) - J
    V_inv = jnp.linalg.solve(V, jnp.eye(t.shape[-1]))
    sign, logdet = jnp.linalg.slogdet(V)
    return (
        beta * jnp.sum(t, axis=-1)
        - 0.5 * sign * logdet
        + 0.25 * beta * jnp.einsum("... i f, ... i j, ... j f -> ...", h, V_inv, h)
    )


def _log_Z(t, h, J, beta):
    return -0.5 * h.shape[-2] * (1.0 + jnp.log(2.0 * beta)) + _phi(t, h, J, beta)


def _t_star_root(h, J, beta):
    a = beta
    b = -0.5
    c = -0.25 * beta * jnp.einsum("... i f, ... i f -> ... i", h, h) - 0.5 * jnp.diagonal(J, axis1=-1, axis2=-2)
    return ((-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)) * jnp.ones(*h.shape[:-1], dtype=h.dtype)


class IsingTransformerLayer(eqx.Module):

    dim: int = eqx.static_field()
    dim_head: int = eqx.static_field()
    num_heads: int = eqx.static_field()
    beta: float = eqx.static_field()
    to_qk: eqx.Module

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

        self.to_qk = eqx.nn.Linear(dim, 2 * dim_head * num_heads, use_bias=False, key=key)

    def _J(self, x, mask=None):
        x = rearrange(x, "...  h n d -> ... n (h d)", h=self.num_heads)

        q, k = jnp.split(jax.vmap(self.to_qk)(x), 2, axis=-1)
        q, k = map(lambda t: rearrange(t, "... n (h d) -> ... h n d", h=self.num_heads), (q, k))

        sim = jnp.einsum("... i d, ... j d -> ... i j", q, k)

        if mask is not None:
            sim = jnp.where(mask, sim, jnp.finfo(sim.dtype).min)

        return jax.nn.softmax(sim, axis=-1) / jnp.sqrt(self.dim_head)

    def _log_Z(self, h, mask, beta):
        def _log_Z_head(h, J, beta):
            return -_log_Z(_t_star_root(h, J, beta), h, J, beta) / beta

        return jax.vmap(_log_Z_head, in_axes=(0, 0, None))(h=h, J=self._J(h, mask=mask), beta=beta)

    def __call__(self, x, mask=None):
        x = rearrange(x, "...  n (h d) -> ... h n d", h=self.num_heads, d=self.dim_head)
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)

        magnetizations = rearrange(
            jnp.diagonal(jax.jacrev(self._log_Z / self.beta, argnums=0)(x, mask=mask, beta=self.beta)),
            "... n d h -> ... n (h d)",
            h=self.num_heads,
        )

        return magnetizations


class IsingTransformer(eqx.Module):

    layers: eqx.Module

    def __init__(self, *, dim, dim_head, num_heads, depth, key):
        layer_keys = jrandom.split(key, depth)

        self.layers = jax.tree_map(
            lambda *xs: jnp.stack(xs),
            *[
                IsingTransformerLayer(dim=dim, dim_head=dim_head, num_heads=num_heads, key=layer_keys[i])
                for i in range(depth)
            ],
        )

    def __call__(self, x, mask=None):
        def apply_scan_fn(x, layer):
            return layer(x, mask=mask), None

        return jax.lax.scan(apply_scan_fn, x, xs=self.layers)[0]
