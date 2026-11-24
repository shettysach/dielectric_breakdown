import unittest

import jax
import jax.numpy as jnp
import numpy as np

from lightning_jax import OffLatticeConfig, OffLatticeResult, map_exterior, simulate_off_lattice


def one_map(angle, size):
    return OffLatticeResult(
        jnp.array([angle]), jnp.array([size]), jnp.empty(0), jnp.int32(1)
    )


class ConformalMapTests(unittest.TestCase):
    def test_single_bump_matches_analytic_positions_and_capacity(self):
        size = 0.09
        points = jnp.array([1.0, -1.0, 1e4])
        mapped, log_derivative = map_exterior(one_map(0.0, size), points)
        np.testing.assert_allclose(mapped[:2], [np.sqrt(1 + size) + np.sqrt(size), -1], atol=2e-6)
        np.testing.assert_allclose(np.exp(log_derivative[-1]), np.sqrt(1 + size), rtol=2e-6)

    def test_derivative_matches_finite_difference_and_map_rotates(self):
        point = jnp.array([1.2 + 0.47j, -1.15 + 0.2j, 0.18 - 1.25j])
        size = 0.074
        reference, log_derivative = map_exterior(one_map(0.0, size), point)
        h = 5e-4
        plus, _ = map_exterior(one_map(0.0, size), point + h)
        minus, _ = map_exterior(one_map(0.0, size), point - h)
        np.testing.assert_allclose(np.exp(log_derivative), np.abs((plus - minus) / (2 * h)), rtol=4e-4)
        phase = np.exp(0.61j)
        rotated, rotated_log_derivative = map_exterior(one_map(0.61, size), point * phase)
        np.testing.assert_allclose(rotated, reference * phase, atol=1e-6)
        np.testing.assert_allclose(rotated_log_derivative, log_derivative, atol=1e-6)

    def test_empty_cluster_is_identity(self):
        result = simulate_off_lattice(OffLatticeConfig(steps=0), jax.random.key(0))
        points = jnp.array([1 + 0j, 3 + 4j])
        mapped, log_derivative = map_exterior(result, points)
        np.testing.assert_array_equal(mapped, points)
        np.testing.assert_array_equal(log_derivative, 0)

    def test_seeded_growth_finishes_with_continuous_geometry(self):
        config = OffLatticeConfig(steps=32, candidates=128)
        key = jax.random.key(7)
        result = jax.device_get(simulate_off_lattice(config, key))
        repeated = jax.device_get(simulate_off_lattice(config, key))
        self.assertEqual(result.steps, config.steps)
        self.assertTrue(np.isfinite(result.patches).all())
        self.assertTrue((result.sizes > 0).all())
        np.testing.assert_array_equal(result.patches, repeated.patches)
        np.testing.assert_allclose(result.sizes[0], config.bump_size**2, rtol=1e-6)
        first_tip = result.patches[0, config.arc_points // 2]
        expected = np.exp(1j * result.angles[0]) * (
            np.sqrt(1 + result.sizes[0]) + np.sqrt(result.sizes[0])
        )
        np.testing.assert_allclose(first_tip, expected, atol=2e-6)
        self.assertGreater(abs(first_tip.real), 1e-3)
        self.assertGreater(abs(first_tip.imag), 1e-3)
        self.assertGreater(np.abs(result.patches).max(), abs(first_tip))
        _, log_derivative = map_exterior(result, jnp.array(1e5))
        np.testing.assert_allclose(
            log_derivative, 0.5 * np.sum(np.log1p(result.sizes)), atol=2e-6
        )


if __name__ == "__main__":
    unittest.main()
