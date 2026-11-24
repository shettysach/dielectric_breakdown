"""Radial dielectric breakdown using semicircular Hastings–Levitov bumps."""

from dataclasses import dataclass
from functools import partial
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax, random


@dataclass(frozen=True)
class OffLatticeConfig:
    steps: int = 500
    eta: float = 2.5
    bump_size: float = 0.18
    candidates: int = 512
    arc_points: int = 17

    def __post_init__(self):
        if self.steps < 0 or self.candidates < 2 or self.arc_points < 3:
            raise ValueError("steps must be nonnegative, candidates at least 2, and arc points at least 3")
        if not isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be finite and nonnegative")
        if not isfinite(self.bump_size) or not 0 < self.bump_size <= 1:
            raise ValueError("bump size must be positive and at most 1 seed radius")


class OffLatticeResult(NamedTuple):
    angles: Array
    sizes: Array
    patches: Array
    steps: Array


def _bump_terms(z, direction, size):
    # The square root is normalized to approach +1 in inverse coordinates
    # at infinity. Its two branch points lie on the unit circle.
    inverse = direction / z
    delta = 1 - inverse
    root = jnp.sqrt(delta * delta + (4 * (size / (1 + size))) * inverse)
    factor = 0.5 * jnp.sqrt(1 + size) * (1 + inverse + root)
    return factor, root, delta


def _compose(z, directions, sizes, count, with_derivative):
    def apply_map(i, state):
        z, log_derivative = state
        index = count - i - 1
        factor, root, delta = _bump_terms(z, directions[index], sizes[index])
        if with_derivative:
            # Chain rule in log space avoids underflow deep in screened regions.
            derivative = 0.5 * factor * (1 + delta / root)
            log_derivative = log_derivative + jnp.log(jnp.abs(derivative))
        mapped = z * factor
        # Roundoff can place a mathematically exterior point just inside.
        mapped = mapped / jnp.minimum(jnp.abs(mapped), 1)
        return mapped, log_derivative

    if sizes.shape[0] == 0:
        return z, jnp.zeros_like(z.real)
    return lax.fori_loop(0, count, apply_map, (z, jnp.zeros_like(z.real)))


@jax.jit
def map_exterior(result: OffLatticeResult, coordinates: Array) -> tuple[Array, Array]:
    """Return F(w) and log|F'(w)| for complex coordinates with |w| >= 1."""
    directions = jnp.exp(1j * result.angles)
    coordinates = jnp.asarray(coordinates, dtype=jnp.result_type(coordinates, directions))
    return _compose(coordinates, directions, result.sizes, result.steps, with_derivative=True)


def _patch(direction, size, arcs, directions, sizes, count):
    opening = 2 * jnp.arctan(jnp.sqrt(size))
    edge = direction * jnp.exp(1j * opening * arcs)
    factor, _, _ = _bump_terms(edge, direction, size)
    bumped = edge * factor
    base = direction * jnp.exp(0.5j * opening * arcs[::-1])
    # The branch endpoints map to the ends of the old boundary arc.
    outline = jnp.concatenate((bumped.at[0].set(base[-1]).at[-1].set(base[0]), base[1:-1]))
    outline = outline / jnp.minimum(jnp.abs(outline), 1)
    polygon, _ = _compose(outline, directions, sizes, count, with_derivative=False)
    return polygon


@partial(jax.jit, static_argnames=("config",))
def simulate_off_lattice(config: OffLatticeConfig, key: Array) -> OffLatticeResult:
    """Grow a unit disk in the plane, with the driving field sourced at infinity.

    Patches are sampled boundaries of the added regions in growth order.
    Only entries before result.steps are valid if numerical growth stops early.
    """
    angles = jnp.zeros(config.steps, dtype=float)
    sizes = jnp.zeros_like(angles)
    directions = jnp.ones(config.steps, dtype=complex)
    patches = jnp.zeros((config.steps, 2 * config.arc_points - 2), dtype=complex)
    result = OffLatticeResult(angles, sizes, patches, jnp.int32(0))
    if config.steps == 0:
        return result

    bins = jnp.arange(config.candidates, dtype=angles.dtype)
    arcs = jnp.linspace(-1, 1, config.arc_points, dtype=angles.dtype)

    def can_grow(state):
        result, _, valid = state
        return (result.steps < config.steps) & valid

    def advance(state):
        result, directions, _ = state
        step = result.steps
        angle_key, choice_key = random.split(random.fold_in(key, step))
        # Jitter each stratum independently so the candidates have no fixed angles.
        proposals = 2 * jnp.pi * (
            (bins + random.uniform(angle_key, bins.shape, dtype=bins.dtype)) / config.candidates
        )
        candidates = jnp.exp(1j * proposals)
        _, log_derivative = _compose(candidates, directions, result.sizes, step, True)
        sizes = jnp.exp(2 * (jnp.log(config.bump_size) - log_derivative))
        valid = jnp.isfinite(log_derivative) & jnp.isfinite(sizes) & (sizes > 0)
        # E^eta per unit physical arclength becomes |F'|^(1-eta) per unit angle.
        logits = jnp.where(valid, (1 - config.eta) * log_derivative, -jnp.inf)
        usable = jnp.any(jnp.isfinite(logits))

        def add_bump(_):
            picked = random.categorical(choice_key, logits, mode="high")
            direction, size = candidates[picked], sizes[picked]
            polygon = _patch(direction, size, arcs, directions, result.sizes, step)
            finite = jnp.all(jnp.isfinite(polygon))
            updated = OffLatticeResult(
                result.angles.at[step].set(proposals[picked]),
                result.sizes.at[step].set(size),
                result.patches.at[step].set(polygon),
                step + 1,
            )
            return lax.cond(
                finite,
                lambda: (updated, directions.at[step].set(direction), finite),
                lambda: (result, directions, finite),
            )

        return lax.cond(usable, add_bump, lambda _: (result, directions, usable), operand=None)

    result, _, _ = lax.while_loop(can_grow, advance, (result, directions, jnp.bool_(True)))
    return result
