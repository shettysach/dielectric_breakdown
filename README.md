# JAX lightning simulation

This project simulates a two-dimensional dielectric breakdown discharge. JAX compiles the Jacobi relaxation and stochastic growth loops with XLA and runs them on the selected device.

## Run the demo

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[demo]'
python -m examples.lightning --output lightning.png --diagnostic potential
python -m examples.lightning --animation lightning.gif --seed 7
```

Run the example and benchmark commands from the project root. Run the demo without output flags to open an interactive Matplotlib window. With both flags, it saves the final plot and the GIF from the same simulation. The optional potential or field panel shows the final solved state throughout a GIF.

Set `--rows`, `--cols`, `--steps`, `--seed`, and `--eta` to explore different grids and growth patterns. Use `--tolerance` and `--max-iterations` to control Jacobi convergence. Run `python -m examples.lightning --help` for all options.

To run on a compatible NVIDIA host, install the CUDA-enabled JAX wheel and select the CUDA platform. For example, on a system supported by the CUDA 13 wheel:

```sh
python -m pip install -U 'jax[cuda13]'
JAX_PLATFORMS=cuda python -m examples.lightning --output lightning.png
```

See [JAX installation](https://docs.jax.dev/en/latest/installation.html) for supported platforms and driver requirements.

## Model and API

The grid has unit spacing. The first and last columns are electrodes fixed at 0 and 1 V by default. The first and last rows are insulating by default, with zero normal flux. Supply `--top-voltage` or `--bottom-voltage` to pin that row. The side electrodes determine the corner voltages. The discharge starts one cell in from the first column and stays at that electrode's potential as it grows.

Each growth step samples a free cell along the four-neighbor front. Its weight is the local magnitude of the electric field raised to `eta`, using centered differences in the interior. Growth is uniform when `eta` is zero or every candidate has zero field. It stops at the requested step limit, when the cluster touches the opposing electrode, or when no candidates remain.

The solver warm-starts from the preceding potential and uses a five-point stencil with replicated ghost cells on insulating edges. The tolerance bounds the maximum Jacobi fixed-point defect, one quarter of the maximum discrete Laplacian residual on the unit grid. The result includes the defect of the final potential and the number of solves that exceeded the tolerance at the iteration cap. Tighten the tolerance and raise the cap to check sensitivity at higher resolutions.

```python
import jax
from lightning_jax import SimulationConfig, simulate

config = SimulationConfig(rows=64, cols=96, steps=250, eta=3.0)
result = simulate(config, jax.random.key(0))
```

`result` contains the final potential, the discharge mask, and an ordered growth array. The first `result.steps` coordinate pairs record growth after the initial seed, and any remaining entries are `(-1, -1)`. The key controls reproducibility for a fixed configuration and numerical backend. Outputs stay as JAX arrays and transfer to the host when needed for plotting or export.

## Validation and benchmarks

```sh
python -m unittest discover -s tests -v
python -m benchmarks.timing --sizes 24 48 96 --ir-dir /tmp/lightning-ir
python -m benchmarks.timing --sizes 12 24 --steps 4 --eager-repeats 1
```

The benchmark times a Laplace solve for the initial seed and a full simulation, separating lowering, compilation, and warmed execution. Each execution timing waits for the device. The unjitted run uses `jax.disable_jit()` and can be much slower. See [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) and the [JAX JIT debugging flags](https://docs.jax.dev/en/latest/debugging/flags.html).

`--ir-dir` exports StableHLO, unoptimized HLO, and optimized HLO where the backend provides it. Inspect the StableHLO `while` operations for the Jacobi and growth loops and the optimized HLO `fusion` instructions for grouped operations. Fusion instruction counts describe the compiled program. GPU launches and utilization can be measured with a device trace. To request backend compiler dumps on a CUDA system, run the benchmark with the XLA dump flags. Output depends on the backend and XLA build. See [OpenXLA compilation dumps](https://openxla.org/xla/hlo_dumps) and [JAX profiling](https://docs.jax.dev/en/latest/profiling.html).

```sh
JAX_PLATFORMS=cuda XLA_FLAGS="--xla_dump_to=/tmp/lightning-xla --xla_dump_hlo_as_text" \
  python -m benchmarks.timing --sizes 128 256 --ir-dir /tmp/lightning-ir
```
