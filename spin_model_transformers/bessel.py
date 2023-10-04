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


import jax.lax as lax
import jax.numpy as jnp
import numpy as np


def bessel_iv_ratio(x, nu, num_iter):
    """Compute ratio `I_{\nu+1}(x) / I_{\nu}(x)` of modified Bessel functions of the first kind.

    Reference:
        D. E. Amos, Computation of Modified Bessel Functions and Their Ratios.
            Mathematics of Computation, 28(125), 239-251 (1974)
    """
    v = np.maximum(20, nu)

    idx_range = jnp.arange(num_iter)
    r_init = x / (v + idx_range + 0.5 + jnp.sqrt((v + idx_range + 1.5) ** 2 + x**2))

    def _update(r, idx):
        return r.at[idx].set(
            x
            / (
                v
                + idx
                + 1.0
                + jnp.sqrt((v + idx + 1.0) ** 2 + x**2 * r[idx + 1] / r[idx])
            )
        )

    def _refine(r):
        def _inner(r, step):
            return lax.scan(
                lambda r, idx: (
                    lax.cond(
                        idx < num_iter - step - 1,
                        lambda r, idx: _update(r, idx),
                        lambda r, idx: r,
                        r,
                        idx,
                    ),
                    None,
                ),
                r,
                jnp.arange(num_iter - 1),
            )[0]

        return lax.scan(
            lambda r, step: (_inner(r, step), None), r, jnp.arange(num_iter)
        )[0]

    def _maybe_recurse(y):
        return lax.scan(
            lambda y, kk: (1.0 / (2 * kk / x + y), None),
            y,
            jnp.arange(v, stop=nu, step=-1),
        )[0]

    return _maybe_recurse(_refine(r_init)[0])
