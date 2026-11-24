"""Run with python -m examples.off_lattice from the project root."""

from argparse import ArgumentParser
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle

from lightning_jax import OffLatticeConfig, simulate_off_lattice


def plot(result):
    patches = result.patches[:result.steps]
    vertices = np.stack((patches.real, patches.imag), axis=-1)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#080e20")
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    ax.add_patch(
        Circle((0, 0), 1, facecolor="#233a56", edgecolor="#6d96b8", linewidth=0.5, zorder=2)
    )
    glow = PolyCollection(
        vertices, facecolors="none", edgecolors="#546af0", linewidths=1.1, alpha=0.12
    )
    discharge = PolyCollection(
        vertices, facecolors="#c8eaff", edgecolors="#c8eaff", linewidths=0.12, zorder=2
    )
    ax.add_collection(glow)
    ax.add_collection(discharge)

    lower = vertices.reshape(-1, 2).min(axis=0, initial=-1)
    upper = vertices.reshape(-1, 2).max(axis=0, initial=1)
    center = (lower + upper) * 0.5
    half_width = np.max(upper - lower) * 0.57
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)
    return fig, vertices, (glow, discharge)


def save_animation(fig, vertices, layers, path, stride):
    frames = [*range(0, len(vertices), stride), len(vertices)]

    def draw(step):
        for layer in layers:
            layer.set_verts(vertices[:step])

    movie = FuncAnimation(fig, draw, frames=frames, interval=100, repeat=False)
    movie.save(path, writer=PillowWriter(fps=10), dpi=100)


def main():
    defaults = OffLatticeConfig()
    parser = ArgumentParser(description="Simulate radial off-lattice dielectric breakdown.")
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--eta", type=float, default=defaults.eta)
    parser.add_argument("--bump-size", type=float, default=defaults.bump_size)
    parser.add_argument("--candidates", type=int, default=defaults.candidates)
    parser.add_argument("--arc-points", type=int, default=defaults.arc_points)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="save a final image, e.g. off-lattice.png")
    parser.add_argument("--animation", type=Path, help="save the evolution as a GIF")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.frame_stride < 1 or args.dpi < 1:
        parser.error("--frame-stride and --dpi must be positive")
    if args.animation and args.animation.suffix.lower() != ".gif":
        parser.error("--animation requires a .gif file")
    try:
        config = OffLatticeConfig(
            steps=args.steps, eta=args.eta, bump_size=args.bump_size,
            candidates=args.candidates, arc_points=args.arc_points,
        )
    except ValueError as error:
        parser.error(str(error))

    result = jax.device_get(simulate_off_lattice(config, jax.random.key(args.seed)))
    fig, vertices, layers = plot(result)
    if args.output:
        fig.savefig(args.output, dpi=args.dpi)
    if args.animation:
        save_animation(fig, vertices, layers, args.animation, args.frame_stride)
    if not (args.output or args.animation):
        plt.show()
    plt.close(fig)

    print(f"{jax.default_backend()} · {result.steps} off-lattice growth steps · seed {args.seed}")
    if result.steps < config.steps:
        raise SystemExit("Stopped early because the conformal map became numerically invalid.")


if __name__ == "__main__":
    main()
