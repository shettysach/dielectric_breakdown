from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax


class LaplaceSolution(NamedTuple):
    potential: Array
    iterations: Array
    defect: Array


def neighbors(grid: Array) -> tuple[Array, Array, Array, Array]:
    """Return up, down, left, and right neighbors with edge replication."""
    return (
        jnp.concatenate((grid[:1, :], grid[:-1, :]), axis=0),
        jnp.concatenate((grid[1:, :], grid[-1:, :]), axis=0),
        jnp.concatenate((grid[:, :1], grid[:, :-1]), axis=1),
        jnp.concatenate((grid[:, 1:], grid[:, -1:]), axis=1),
    )


@jax.jit
def solve_potential(
    initial: Array,
    fixed_mask: Array,
    fixed_values: Array,
    tolerance: float,
    max_iterations: int,
) -> LaplaceSolution:
    """Relax a floating-point grid with Dirichlet cells and insulated edges.

    The tolerance bounds the max Jacobi fixed-point defect in potential units.
    On a unit grid that defect is one quarter of the max Laplacian residual.
    """
    def jacobi(potential):
        up, down, left, right = neighbors(potential)
        # Scale before summing so the average stays in the potential range.
        average = (up * 0.25 + down * 0.25) + (left * 0.25 + right * 0.25)
        return jnp.where(fixed_mask, fixed_values, average)

    def propose(potential):
        proposal = jacobi(potential)
        defect = jnp.max(jnp.abs(proposal - potential))
        return proposal, defect

    def needs_update(state):
        iterations, _, _, defect = state
        return (iterations < max_iterations) & (defect > tolerance)

    def update(state):
        iterations, _, potential, _ = state
        proposal, defect = propose(potential)
        return iterations + 1, potential, proposal, defect

    potential = jnp.where(fixed_mask, fixed_values, initial)
    proposal, defect = propose(potential)
    iterations, potential, _, defect = lax.while_loop(
        needs_update, update, (0, potential, proposal, defect)
    )
    return LaplaceSolution(potential, iterations, defect)


@jax.jit
def electric_field(
    potential: Array, insulated_top: bool = False, insulated_bottom: bool = False
) -> tuple[Array, Array]:
    """Return (E_x, E_y), with y increasing along rows on a unit grid."""
    up, down, left, right = neighbors(potential)
    ex = (left - right) * 0.5
    ey = (up - down) * 0.5
    ex = ex.at[:, 0].multiply(2).at[:, -1].multiply(2)
    ey = ey.at[0, :].set(jnp.where(insulated_top, 0, ey[0, :] * 2))
    ey = ey.at[-1, :].set(jnp.where(insulated_bottom, 0, ey[-1, :] * 2))
    return ex, ey
