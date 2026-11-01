import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ive

from spin_model_transformers.bessel import bessel_iv_ratio


class BesselRatioTest(unittest.TestCase):
    def test_default_cutoff(self):
        for nu in (0.0, 0.5, 1.0, 1.5, 2.5, 19.5, 20.0, 20.5, 32.0):
            with self.subTest(nu=nu):
                actual = bessel_iv_ratio(jnp.float32(1.0), nu, 2)
                expected = ive(nu + 1, 1.0) / ive(nu, 1.0)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)

    def test_custom_cutoff_under_jit(self):
        ratio = jax.jit(
            bessel_iv_ratio,
            static_argnames=("nu", "num_iter", "min_order"),
        )
        x = np.array([0.1, 1.0, 5.0, 100.0], dtype=np.float32)
        for min_order in (0, 10, 21, 32):
            with self.subTest(min_order=min_order):
                # Bind the keyword before vmap, which otherwise maps kwargs.
                def evaluate(value):
                    return ratio(value, 0.5, 16, min_order=min_order)

                actual = jax.vmap(evaluate)(jnp.asarray(x))
                expected = ive(1.5, x.astype(np.float64)) / ive(0.5, x.astype(np.float64))
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


if __name__ == "__main__":
    unittest.main()
