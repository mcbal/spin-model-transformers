from functools import partial
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxopt import AndersonAcceleration


def _gamma(x, beta, R):
    return jnp.sqrt(1 + beta**2 * jnp.sum(x**2, axis=-1, keepdims=True) / R**2)


def _phi(theta, beta, R):
    return beta / (1 + _gamma(theta, beta, R)) * theta


def _inv_phi(m, beta, R):
    return 2 * R**2 / (beta * (R**2 - jnp.sum(m**2, axis=-1, keepdims=True))) * m


def _d2_m_d_alpha_2(m1, m0, x, J, beta, R):
    g0 = _gamma(_inv_phi(m0, beta, R), beta, R)
    g1 = _gamma(_inv_phi(m1, beta, R), beta, R)
    v = -_inv_phi(m1, beta, R) + x + jnp.einsum("i j, j d -> i d", J, m0)

    return (
        (beta**2 * (1 + 3 * g1))
        / (R**4 * g1**3)
        * (
            jnp.einsum("i d, i d -> i", m1, v)[:, None] ** 2
            + jnp.einsum(
                "i j, i d -> i d",
                J**2,
                jnp.sum(m1**2, axis=-1, keepdims=True),
            )
            / (1 + g0)
            - jnp.einsum(
                "i j, i d, j d, i e, j e -> i",
                J**2,
                m1,
                m0,
                m1,
                m0,
            )[:, None]
            / (R**2 * g0)
        )
        * m1
        - (beta**2)
        / (R**2 * (g1**2 + g1))
        * (
            jnp.sum(v**2, axis=-1, keepdims=True)
            + jnp.einsum(
                "i j, j -> i",
                J**2,
                R**2 - jnp.sum(m0**2, axis=-1),
            )[:, None]
        )
        * m1
        - 2.0
        * beta**2
        / (R**2 * (g1**2 + g1))
        * (
            jnp.einsum("i d, i d, i f -> i f", v, m1, v)
            + jnp.einsum("i j, i d -> i d", J**2, m1 / (1 + g0))
            - jnp.einsum(
                "i j, i d, j d, j f -> i f",
                J**2,
                m1,
                m0,
                m0,
            )
            / (R**2 * g0)
        )
    )


def _f(m1, m0, x, J, beta, R):
    g1 = _gamma(_inv_phi(m1, beta, R), beta, R)
    d2_m_d_alpha_2 = _d2_m_d_alpha_2(m1, m0, x, J, beta, R)

    ff = (
        (1 + g1)
        / (2 * beta)
        * (
            d2_m_d_alpha_2
            + (
                jnp.einsum("i d, i d -> i", m1, d2_m_d_alpha_2)[:, None]
                / ((R**2 * g1) / (1 + g1) - jnp.sum(m1**2, axis=-1, keepdims=True))
                * m1
            )
        )
    )
    return x + jnp.einsum("i j, j d -> i d", J, m0) + ff


def vector_tap_fp(m0, x, J, beta, R, tol: float = 1e-3, maxiter: int = 100):
    def _fun(m, _x, _J, _beta, _R):
        return _phi(_f(m, m, _x, _J, _beta, _R), _beta, _R)

    return (
        AndersonAcceleration(
            fixed_point_fun=_fun,
            tol=tol,
            maxiter=maxiter,
        )
        .run(_phi(x + J @ m0, beta, R), x, J, beta, R)
        .params
    )


class SpinTransformerModule(eqx.Module):
    dim: int
    dim_head: int
    num_heads: int
    scale: float
    to_qk: eqx.Module
    vector_tap_fp: Callable

    def __init__(
        self,
        *,
        dim,
        num_heads,
        beta,
        key,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = (self.dim_head / 2 - 1) ** 0.5

        self.to_qk = eqx.nn.Linear(
            dim, 2 * self.dim_head * num_heads, use_bias=False, key=key
        )
        self.vector_tap_fp = partial(
            vector_tap_fp, beta=beta, R=(self.dim_head / 2 - 1) ** 0.5
        )

    def _J(self, x, mask=None):
        x = rearrange(x, "... h n d -> ... n (h d)", h=self.num_heads)

        q, k = jnp.split(jax.vmap(self.to_qk)(x), 2, axis=-1)
        q, k = map(
            lambda t: rearrange(t, "... n (h d) -> ... h n d", h=self.num_heads), (q, k)
        )

        sim = jnp.einsum("... i d, ... j d -> ... i j", q, k)

        if mask is not None:
            sim = jnp.where(mask, sim, jnp.finfo(sim.dtype).min)

        return jax.nn.softmax(sim, axis=-1)

    def __call__(self, x, mask=None):
        x = rearrange(x, "... n (h d) -> ... h n d", h=self.num_heads, d=self.dim_head)
        x = self.scale * x / jnp.linalg.norm(x, axis=-1, keepdims=True)

        m0 = jnp.ones_like(x)
        m0 = m0 / jnp.linalg.norm(m0, axis=-1, keepdims=True)

        return rearrange(
            jax.vmap(self.vector_tap_fp, in_axes=(0, 0, 0))(
                m0, x, self._J(x, mask=mask)
            ),
            "... h n d -> ... n (h d)",
        )


class SpinTransformer(eqx.Module):
    modules: SpinTransformerModule

    def __init__(self, depth, dim, num_heads, beta, key):
        keys = jax.random.split(key, depth)

        make_modules = lambda k: SpinTransformerModule(
            dim=dim, num_heads=num_heads, beta=beta, key=k
        )
        self.modules = eqx.filter_vmap(make_modules)(keys)

    def __call__(self, x):
        dynamic_modules, static_modules = eqx.partition(self.modules, eqx.is_array)

        def f(_x, _dynamic_module):
            module = eqx.combine(_dynamic_module, static_modules)
            return module(_x), None

        out, _ = jax.lax.scan(f, x, dynamic_modules)
        return out
