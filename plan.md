# Project Plan: GPU-Accelerated Lightning Simulation

## Overview

Build a GPU-accelerated scientific simulation of electrical discharge branching using the **Dielectric Breakdown Model (DBM)**.

The primary goal is to demonstrate how modern deep learning frameworks, particularly **JAX/XLA**, can be used for traditional numerical scientific computing.

We will eventually compare JAX against PyTorch and Tinygrad in terms of performance, compilation, GPU utilization, and generated GPU kernels.

**For now, focus exclusively on implementing the JAX backend.**

## Project Structure

```text
.
├── src/
│   ├── jax/          # Primary implementation
│   ├── torch/        # Future PyTorch implementation
│   └── tinygrad/     # Future Tinygrad implementation
├── examples/         # Simulation demos
├── benchmarks/       # Performance comparisons
├── tests/            # Numerical correctness tests
└── README.md
```

Keep implementations independent but expose similar APIs to enable fair comparisons.

## Physics and Numerical Model

Implement a 2D Dielectric Breakdown Model.

1. Represent the simulation domain as a 2D grid with electrodes, insulating boundaries, and a growing conductive discharge.
2. Solve Laplace's equation in the nonconducting region:

   \[
   \nabla^2\phi=0
   \]

3. Compute the electric field:

   \[
   \mathbf{E}=-\nabla\phi
   \]

4. Identify candidate cells adjacent to the discharge.
5. Sample the next growth location using an electric-field-dependent probability:

   \[
   P_i\propto|\mathbf E_i|^\eta
   \]

6. Update the conductive region and repeat.

The model should produce branching, lightning-like discharge patterns.

## JAX Implementation

Implement everything inside `src/jax/` initially.

- Use `jax.numpy` for grid operations.
- Implement a basic finite-difference Laplace solver (Jacobi initially).
- Use `jax.jit` and `jax.lax` control flow to compile iterative computations.
- Use JAX's PRNG for reproducible stochastic growth.
- Keep core computations on the GPU, minimizing CPU synchronization and host-device transfers.
- Design the solver and growth logic as separate components to allow future numerical methods.
- Support configurable grid sizes, boundary conditions, solver tolerances, and growth parameters.

Prioritize **correctness and a working end-to-end simulation** before optimization.

## Demonstration and Visualization

Provide a runnable example that:

- Generates a branching lightning discharge.
- Visualizes the evolving discharge and optionally the electric potential/field.
- Supports different resolutions and random seeds.
- Can export images or animations of the simulation.

Visualization can initially use Matplotlib. NVIDIA Warp or another GPU-based visualization backend can be considered later.

## Performance and Compiler Analysis

Once the simulation works:

- Benchmark PDE solve time and overall simulation time at different grid resolutions.
- Compare JIT-compiled versus non-JIT execution.
- Inspect JAX's generated StableHLO/HLO and GPU compiler output where available.
- Explore the effects of JIT compilation, loop optimization, and kernel fusion.
- Report compilation time separately from steady-state execution time.
- Validate numerical convergence and solver correctness.

## Future Extensions

Do not implement these yet:

- PyTorch and Tinygrad backends with equivalent numerical behavior.
- Alternative PDE solvers and optimization techniques.
- 3D dielectric breakdown.
- More physically detailed discharge models.
- NVIDIA Warp visualization.
- Cross-framework benchmarking.

## Immediate Goal

Deliver a **working 2D JAX lightning simulator** that solves Laplace's equation, grows a stochastic discharge, and produces a convincing visualization.

Keep the code modular, reasonably minimal, and extensible. Avoid premature abstractions for backends that do not exist yet.
