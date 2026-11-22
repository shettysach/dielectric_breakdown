from dataclasses import dataclass
from functools import partial
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax, random

from .solver import electric_field, neighbors, solve_potential


@dataclass(frozen=True)
class SimulationConfig:
    rows: int = 96
    cols: int = 128
    steps: int = 450
    eta: float = 3.0
    tolerance: float = 1e-5
    max_iterations: int = 5000
    left_voltage: float = 0.0
    right_voltage: float = 1.0
    top_voltage: float | None = None
    bottom_voltage: float | None = None

    def __post_init__(self):
        if self.rows < 3 or self.cols < 4:
            raise ValueError("the grid requires at least 3 rows and 4 columns")
        if self.steps < 0 or self.max_iterations < 1:
            raise ValueError("steps must be nonnegative and max_iterations positive")
        if not isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be finite and nonnegative")
        if not isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        voltages = (self.left_voltage, self.right_voltage, self.top_voltage, self.bottom_voltage)
        if any(v is not None and not isfinite(v) for v in voltages):
            raise ValueError("electrode voltages must be finite")
        if self.left_voltage == self.right_voltage:
            raise ValueError("the left and right voltages must differ")

    @property
    def seed_cell(self) -> tuple[int, int]:
        return self.rows // 2, 1


class Domain(NamedTuple):
    potential: Array
    fixed_mask: Array
    fixed_values: Array
    discharge: Array


class SimulationResult(NamedTuple):
    potential: Array
    discharge: Array
    growth: Array
    steps: Array
    reached_electrode: Array
    stalled: Array
    unconverged_solves: Array
    final_defect: Array


def initialize(config: SimulationConfig) -> Domain:
    """Build the boundary data and a warm start for the first solve."""
    shape = config.rows, config.cols
    profile = jnp.linspace(config.left_voltage, config.right_voltage, config.cols, dtype=jnp.float32)
    fixed_values = jnp.broadcast_to(profile, shape)
    fixed_mask = jnp.zeros(shape, dtype=bool).at[:, 0].set(True).at[:, -1].set(True)
    for row, voltage in ((0, config.top_voltage), (-1, config.bottom_voltage)):
        if voltage is not None:
            fixed_mask = fixed_mask.at[row, 1:-1].set(True)
            fixed_values = fixed_values.at[row, 1:-1].set(voltage)
    discharge = jnp.zeros(shape, dtype=bool).at[config.seed_cell].set(True)
    potential = jnp.where(discharge, config.left_voltage, fixed_values)
    return Domain(potential, fixed_mask, fixed_values, discharge)


def _sample_growth(key, potential, candidates, config):
    if config.eta == 0:
        logits = jnp.zeros_like(potential)
    else:
        ex, ey = electric_field(
            potential, config.top_voltage is None, config.bottom_voltage is None
        )
        magnitude = jnp.hypot(ex, ey)
        log_field = jnp.where(candidates, jnp.log(magnitude), -jnp.inf)
        max_log_field = jnp.max(log_field)
        # Normalize in log space, falling back to uniform growth for zero field.
        logits = jnp.where(
            jnp.isfinite(max_log_field),
            (log_field - max_log_field) * config.eta,
            0.0,
        )
    logits = jnp.where(candidates, logits, -jnp.inf)
    flat_index = random.categorical(key, logits.reshape(-1), mode="high")
    return flat_index // config.cols, flat_index % config.cols


@partial(jax.jit, static_argnames=("config",))
def simulate(config: SimulationConfig, key: Array) -> SimulationResult:
    """Grow a seeded discharge and return its final potential and ordered cells.

    The growth array has shape (config.steps, 2), padded with (-1, -1).
    Only the first result.steps rows are valid. The seed is implicit.
    """
    domain = initialize(config)

    def solve(potential, discharge):
        fixed = domain.fixed_mask | discharge
        values = jnp.where(discharge, config.left_voltage, domain.fixed_values)
        return solve_potential(potential, fixed, values, config.tolerance, config.max_iterations)

    def candidates(discharge):
        up, down, left, right = neighbors(discharge)
        return (up | down | left | right) & ~(discharge | domain.fixed_mask)

    first = solve(domain.potential, domain.discharge)
    growth = jnp.full((config.steps, 2), -1, dtype=jnp.int32)
    state = SimulationResult(
        first.potential, domain.discharge, growth, jnp.int32(0), jnp.bool_(False),
        ~jnp.any(candidates(domain.discharge)), jnp.int32(first.defect > config.tolerance),
        first.defect,
    )
    if config.steps == 0:
        return state

    def can_grow(state):
        return (state.steps < config.steps) & ~state.reached_electrode & ~state.stalled

    def advance(state):
        step_key = random.fold_in(key, state.steps)
        row, col = _sample_growth(step_key, state.potential, candidates(state.discharge), config)
        discharge = state.discharge.at[row, col].set(True)
        growth = state.growth.at[state.steps].set(jnp.array([row, col], dtype=jnp.int32))
        solution = solve(state.potential, discharge)
        reached = col == config.cols - 2
        return SimulationResult(
            solution.potential, discharge, growth, state.steps + 1, reached,
            ~reached & ~jnp.any(candidates(discharge)),
            state.unconverged_solves + (solution.defect > config.tolerance),
            solution.defect,
        )

    return lax.while_loop(can_grow, advance, state)
