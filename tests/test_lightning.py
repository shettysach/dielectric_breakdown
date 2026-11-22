import unittest

import jax
import jax.numpy as jnp
import numpy as np

from lightning_jax import SimulationConfig, electric_field, simulate, solve_potential
from lightning_jax.simulation import _sample_growth, initialize


class LaplaceTests(unittest.TestCase):
    def test_converges_to_linear_profile_with_insulated_edges(self):
        rows, cols = 9, 15
        expected = np.broadcast_to(np.linspace(0, 1, cols), (rows, cols))
        fixed = jnp.zeros((rows, cols), dtype=bool).at[:, [0, -1]].set(True)
        initial = jnp.zeros((rows, cols))
        solution = solve_potential(initial, fixed, jnp.asarray(expected), 1e-6, 5000)
        self.assertGreater(int(solution.iterations), 0)
        self.assertLessEqual(float(solution.defect), 1e-6)
        np.testing.assert_allclose(solution.potential, expected, atol=1e-4)
        ex, ey = electric_field(solution.potential, True, True)
        np.testing.assert_allclose(ex, -1 / (cols - 1), atol=3e-5)
        np.testing.assert_allclose(ey, 0, atol=1e-5)

    def test_iteration_limit_reports_remaining_defect(self):
        domain = initialize(SimulationConfig(rows=12, cols=20))
        solution = solve_potential(
            jnp.zeros_like(domain.potential), domain.fixed_mask, domain.fixed_values, 1e-6, 1
        )
        self.assertEqual(int(solution.iterations), 1)
        self.assertGreater(float(solution.defect), 1e-6)
        potential = np.asarray(solution.potential)
        np.testing.assert_array_equal(potential[:, 0], 0)
        np.testing.assert_array_equal(potential[:, -1], 1)
        laplacian = (
            np.pad(potential, ((1, 0), (0, 0)), mode="edge")[:-1, 1:-1]
            + np.pad(potential, ((0, 1), (0, 0)), mode="edge")[1:, 1:-1]
            + potential[:, :-2] + potential[:, 2:] - 4 * potential[:, 1:-1]
        )
        self.assertAlmostEqual(float(solution.defect), np.abs(laplacian).max() / 4)


class GrowthTests(unittest.TestCase):
    def test_sampling_matches_field_weights(self):
        config = SimulationConfig(rows=3, cols=5, eta=2)
        potential = jnp.broadcast_to(jnp.arange(5, dtype=float) ** 2, (3, 5))
        candidates = jnp.zeros_like(potential, dtype=bool).at[1, 1:3].set(True)
        keys = jax.random.split(jax.random.key(17), 4096)
        sample = jax.jit(jax.vmap(lambda key: _sample_growth(key, potential, candidates, config)))
        rows, cols = map(np.asarray, sample(keys))
        np.testing.assert_array_equal(rows, 1)
        self.assertTrue(np.all(np.isin(cols, [1, 2])))
        self.assertAlmostEqual(float(np.mean(cols == 2)), 0.8, delta=0.04)

    def test_seeded_growth_is_reproducible_and_connected(self):
        config = SimulationConfig(rows=18, cols=30, steps=24, top_voltage=0.2, bottom_voltage=0.8)
        key = jax.random.key(42)
        result = jax.device_get(simulate(config, key))
        repeated = jax.device_get(simulate(config, key))
        np.testing.assert_array_equal(result.growth, repeated.growth)
        self.assertEqual(result.steps, config.steps)
        self.assertEqual(result.unconverged_solves, 0)
        seen = {config.seed_cell}
        for row, col in result.growth[:result.steps]:
            self.assertTrue(0 < row < config.rows - 1 and 0 < col < config.cols - 1)
            self.assertNotIn((row, col), seen)
            adjacent = {(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)}
            self.assertTrue(adjacent & seen)
            seen.add((row, col))
        self.assertEqual(set(zip(*np.where(result.discharge))), seen)
        np.testing.assert_array_equal(result.potential[result.discharge], config.left_voltage)
        np.testing.assert_array_equal(result.potential[:, 0], config.left_voltage)
        np.testing.assert_array_equal(result.potential[:, -1], config.right_voltage)
        np.testing.assert_allclose(result.potential[0, 1:-1], config.top_voltage)
        np.testing.assert_allclose(result.potential[-1, 1:-1], config.bottom_voltage)

    def test_stops_on_contact_and_pads_unused_growth(self):
        config = SimulationConfig(
            rows=3, cols=4, steps=9, eta=0, top_voltage=0.0, bottom_voltage=0.0
        )
        result = jax.device_get(simulate(config, jax.random.key(2)))
        self.assertTrue(result.reached_electrode)
        self.assertFalse(result.stalled)
        self.assertEqual(result.steps, 1)
        np.testing.assert_array_equal(result.growth[0], [1, 2])
        np.testing.assert_array_equal(result.growth[1:], -1)
        self.assertEqual(result.potential[1, 2], config.left_voltage)

    def test_zero_growth_solving_preserves_the_seed(self):
        config = SimulationConfig(rows=8, cols=10, steps=0)
        result = jax.device_get(simulate(config, jax.random.key(0)))
        self.assertEqual(result.growth.shape, (0, 2))
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.discharge.sum(), 1)
        self.assertTrue(result.discharge[config.seed_cell])
        self.assertLessEqual(result.final_defect, config.tolerance)


if __name__ == "__main__":
    unittest.main()
