"""Run with python -m benchmarks.timing from the project root."""

from argparse import ArgumentParser
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp

from lightning_jax import SimulationConfig, simulate, solve_potential
from lightning_jax.simulation import initialize


def elapsed_ms(start):
    return (perf_counter() - start) * 1000


def time_calls(function, args, repeats):
    measurements = []
    for _ in range(repeats):
        start = perf_counter()
        result = jax.block_until_ready(function(*args))
        measurements.append(elapsed_ms(start))
    return median(measurements), result


def benchmark(name, function, args, static_args, size, repeats, eager_repeats, ir_dir):
    jax.block_until_ready(args)
    start = perf_counter()
    lowered = function.lower(*static_args, *args)
    lower_ms = elapsed_ms(start)
    start = perf_counter()
    compiled = lowered.compile()
    compile_ms = elapsed_ms(start)
    jax.block_until_ready(compiled(*args))
    jit_ms, result = time_calls(compiled, args, repeats)

    eager = "—"
    if eager_repeats:
        with jax.disable_jit():
            eager_function = lambda *inputs: function(*static_args, *inputs)
            jax.block_until_ready(eager_function(*args))
            eager_ms, _ = time_calls(eager_function, args, eager_repeats)
        eager = f"{eager_ms:.2f}"
    print(
        f"{size:>5}  {name:<10}  {lower_ms:>10.2f}  {compile_ms:>11.2f}"
        f"  {jit_ms:>10.2f}  {eager:>11}"
    )

    if ir_dir is not None:
        stem = f"{name}-{size}"
        stablehlo = lowered.as_text(dialect="stablehlo")
        (ir_dir / f"{stem}.stablehlo.mlir").write_text(stablehlo)
        hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
        (ir_dir / f"{stem}.hlo.txt").write_text(hlo)
        optimized = compiled.as_text()
        if optimized is not None:
            (ir_dir / f"{stem}.optimized-hlo.txt").write_text(optimized)
            print(
                f"  {stablehlo.count('stablehlo.while')} StableHLO while ops, "
                f"{optimized.count(' fusion(')} optimized HLO fusion instructions"
            )
    return jax.device_get(result)


def main():
    parser = ArgumentParser(description="Benchmark the Laplace solver and DBM simulation.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[24, 48, 96])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eager-repeats", type=int, default=0, help="enable unjitted timings")
    parser.add_argument("--ir-dir", type=Path, help="export StableHLO, HLO, and optimized HLO")
    args = parser.parse_args()
    if min(args.sizes) < 4 or args.steps < 0 or args.repeats < 1 or args.eager_repeats < 0:
        parser.error("sizes must be at least 4, repeats positive, and steps and eager repeats nonnegative")
    if args.ir_dir is not None:
        args.ir_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(0)
    print(f"JAX {jax.__version__} on {jax.devices()[0]} · median elapsed time in ms")
    print(f"{'Grid':>5}  {'Workload':<10}  {'Lower':>10}  {'Compile':>11}  {'JIT':>10}  {'Unjitted':>11}")
    for size in args.sizes:
        config = SimulationConfig(rows=size, cols=size, steps=args.steps)
        domain = initialize(config)
        solve_args = (
            domain.potential,
            domain.fixed_mask | domain.discharge,
            jnp.where(domain.discharge, config.left_voltage, domain.fixed_values),
            config.tolerance,
            config.max_iterations,
        )
        solution = benchmark(
            "laplace", solve_potential, solve_args, (), size, args.repeats, args.eager_repeats,
            args.ir_dir,
        )
        print(f"  {solution.iterations} Jacobi iterations, final defect {solution.defect:.3g}")
        result = benchmark(
            "simulation", simulate, (key,), (config,), size, args.repeats, args.eager_repeats,
            args.ir_dir,
        )
        print(
            f"  {result.steps} growth steps, {result.unconverged_solves} solves above tolerance"
        )


if __name__ == "__main__":
    main()
