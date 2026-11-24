# JAX lightning simulation

This project simulates a two-dimensional dielectric breakdown discharge. JAX compiles the Jacobi relaxation and stochastic growth loops with XLA and runs them on the selected device.

## Off-lattice discharge

With the demo dependencies installed, run these commands from the project root:

```sh
python -m examples.off_lattice --output off-lattice.png
python -m examples.off_lattice --steps 1400 --candidates 768 --eta 2.75 --seed 17 --dpi 400 --output off-lattice-large.png
python -m examples.off_lattice --steps 1500 --candidates 768 --eta 3.15 --bump-size 0.085 --seed 17 --dpi 400 --output off-lattice-thin.png
python -m examples.off_lattice --steps 250 --animation off-lattice.gif
```

This is the radial conformal-map DBM. It starts from a conducting unit disk with a logarithmic far-field potential and adds dielectric breakdown bumps at continuously sampled angles. It solves the exterior Laplace problem by mapping the exterior of that disk to the exterior of the growing cluster. This geometry models a discharge growing out from the seed in an unbounded plane. The grid demo below models a gap between opposing electrodes.

For the map `z = F(w)`, the normalized potential at `F(w)` is `log|w|`. On the mapped boundary the field magnitude is `1/|F'(w)|`. The growth density per unit angle is proportional to `|F'|^(1 − eta)` because physical arclength contributes one factor of `|F'|`. Each step estimates that density from stratified random angular candidates and rescales its bump so the nominal growth increment has the same physical size to first order. See [Mathiesen, Jensen, and Bakke](https://arxiv.org/abs/0712.1739) and the [Hastings–Levitov elementary maps](https://pdodds.w3.uvm.edu/files/papers/others/1998/hastings1998a.pdf).

```python
import jax
import jax.numpy as jnp
from lightning_jax import OffLatticeConfig, map_exterior, simulate_off_lattice

result = simulate_off_lattice(OffLatticeConfig(), jax.random.key(7))
w = jnp.array([1.5 + 0j, 2.0 + 1j])
position, log_jacobian = map_exterior(result, w)
potential = jnp.log(jnp.abs(w))
field_magnitude = jnp.exp(-potential - log_jacobian)
```

`result.patches[:result.steps]` holds sampled outlines of the added regions, and the result also stores the map angles and bump parameters. `--arc-points` adjusts rendering detail. The simulation reevaluates the accumulated map at every candidate, so increasing the number of growth steps has roughly quadratic cost at a fixed candidate count. For quantitative studies, check sensitivity to `--candidates` and `--bump-size` across multiple seeds.

The thin-channel example uses smaller bumps to resolve finer branches and a larger `eta` to favor growing tips. The seed disk is shown in a muted color.

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
