"""Run with python -m examples.lightning from the project root."""

from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from lightning_jax import SimulationConfig, electric_field, simulate


def plot(config, result, diagnostic):
    has_diagnostic = diagnostic is not None
    fig, axes = plt.subplots(
        1, 2 if has_diagnostic else 1, squeeze=False,
        figsize=(11 if has_diagnostic else 6, 5), layout="constrained",
    )
    ax = axes[0, 0]
    image = ax.imshow(
        result.discharge, cmap=ListedColormap(["#080e20", "#c8eaff"]),
        vmin=0, vmax=1, interpolation="nearest", origin="upper",
    )
    ax.axvline(0, color="#527696", linewidth=1)
    ax.axvline(config.cols - 1, color="#527696", linewidth=1)
    if config.top_voltage is not None:
        ax.axhline(0, color="#527696", linewidth=1)
    if config.bottom_voltage is not None:
        ax.axhline(config.rows - 1, color="#527696", linewidth=1)
    ax.set(xlabel="Grid column", ylabel="Grid row", title=f"Discharge after {result.steps} steps")

    if has_diagnostic:
        grid, title, cmap = diagnostic
        field_ax = axes[0, 1]
        field = field_ax.imshow(grid, cmap=cmap, interpolation="nearest", origin="upper")
        field_ax.set(xlabel="Grid column", ylabel="Grid row", title=title)
        fig.colorbar(field, ax=field_ax, shrink=0.75)
    return fig, image


def save_animation(fig, image, config, result, path, stride):
    steps = int(result.steps)
    arrival = np.full(result.discharge.shape, steps + 1)
    arrival[config.seed_cell] = 0
    rows, cols = result.growth[:steps].T
    arrival[rows, cols] = np.arange(1, steps + 1)
    frames = [*range(0, steps, stride), steps]

    def draw(step):
        image.set_data(arrival <= step)
        image.axes.set_title(f"Discharge after {step} steps")

    movie = FuncAnimation(fig, draw, frames=frames, interval=100, repeat=False)
    movie.save(path, writer=PillowWriter(fps=10))


def main():
    defaults = SimulationConfig()
    parser = ArgumentParser(description="Simulate a branching dielectric breakdown discharge.")
    parser.add_argument("--rows", type=int, default=defaults.rows)
    parser.add_argument("--cols", type=int, default=defaults.cols)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eta", type=float, default=defaults.eta)
    parser.add_argument("--tolerance", type=float, default=defaults.tolerance)
    parser.add_argument("--max-iterations", type=int, default=defaults.max_iterations)
    parser.add_argument("--left-voltage", type=float, default=defaults.left_voltage)
    parser.add_argument("--right-voltage", type=float, default=defaults.right_voltage)
    parser.add_argument("--top-voltage", type=float, help="fix the first row at this voltage")
    parser.add_argument("--bottom-voltage", type=float, help="fix the last row at this voltage")
    parser.add_argument("--diagnostic", choices=["potential", "field"])
    parser.add_argument("--output", type=Path, help="save the final plot, e.g. lightning.png")
    parser.add_argument("--animation", type=Path, help="save the evolution as a GIF")
    parser.add_argument("--frame-stride", type=int, default=5)
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be positive")
    if args.animation and args.animation.suffix.lower() != ".gif":
        parser.error("--animation requires a .gif file")
    try:
        config = SimulationConfig(**{name: getattr(args, name) for name in vars(defaults)})
    except ValueError as error:
        parser.error(str(error))

    result = simulate(config, jax.random.key(args.seed))
    diagnostic = None
    if args.diagnostic == "potential":
        diagnostic = result.potential, "Final electric potential (V)", "viridis"
    if args.diagnostic == "field":
        ex, ey = electric_field(
            result.potential, config.top_voltage is None, config.bottom_voltage is None
        )
        magnitude = jnp.where(result.discharge, jnp.nan, jnp.hypot(ex, ey))
        diagnostic = magnitude, "Final field magnitude (V per grid unit)", "magma"
    result = jax.device_get(result)
    if diagnostic is not None:
        grid, title, cmap = diagnostic
        diagnostic = jax.device_get(grid), title, cmap
    fig, image = plot(config, result, diagnostic)
    if args.output:
        fig.savefig(args.output, dpi=160)
    if args.animation:
        save_animation(fig, image, config, result, args.animation, args.frame_stride)
    if not (args.output or args.animation):
        plt.show()
    plt.close(fig)

    reason = "reached the opposing electrode" if result.reached_electrode else "step limit"
    if result.stalled:
        reason = "no eligible growth cells"
    print(f"{jax.default_backend()} · {result.steps} growth steps · {reason}")
    print(f"Final Jacobi defect {result.final_defect:.3g} · {result.unconverged_solves} solves above tolerance")


if __name__ == "__main__":
    main()
