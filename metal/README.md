# HOOMD-metal technical guide

This guide covers installation options, validation, benchmarking, runtime settings, and how the
Metal backend works. For an overview, see the [main README](../README.md).

## Requirements

- An Apple silicon Mac (M1 or newer; any variant).
- macOS 15 (Sequoia) or newer.
- Apple's command line tools: `xcode-select --install`.
- An arm64 conda, such as [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda for
  Apple silicon. An Intel conda running under Rosetta does not work, and the installer says so.

Intel Macs cannot use the GPU backend: on an Intel Mac `hoomd.device.GPU()` raises an error that
says to use `hoomd.device.CPU()`.

## Install

```bash
git clone --recursive https://github.com/aldorvasq/HOOMD-metal.git
cd HOOMD-metal
metal/install.sh --validate
conda activate hoomd-metal
```

[`install.sh`](install.sh) checks the Mac, creates (or updates) the conda environment from
[`environment.yml`](environment.yml), fetches missing git submodules, builds HOOMD-metal,
installs it into the environment, and runs a short GPU test. The build takes about 5 minutes on an
M4 Mac mini.

| Option | Effect |
|---|---|
| `--name ENV` | Install into the conda environment `ENV` instead of `hoomd-metal`. |
| `--jobs N` | Compile with `N` parallel jobs (the default is chosen from memory and cores). |
| `--build-dir DIR` | Build in `DIR` instead of `build/hoomd-metal`. |
| `--no-mpcd` | Skip the MPCD component. |
| `--validate` | Run the validation checks after installing. |

To update, `git pull` and run `metal/install.sh` again. To uninstall, run
`conda env remove -n hoomd-metal`.

To configure a build by hand, pass `-DENABLE_GPU=ON -DHOOMD_GPU_PLATFORM=Metal` to CMake. Metal
builds are always single precision, build without HPMC and MPI, and target macOS 15.

## Check this Mac

```bash
python metal/validate/validate.py             # 10 checks, about 2 minutes
python metal/validate/validate.py polymer     # selected checks
python metal/validate/validate.py --pytest    # also HOOMD-blue's test suite (30 s more)
```

Each check runs the same simulation with `hoomd.device.CPU()` and `hoomd.device.GPU()`, in
separate processes, and compares the results:

- Single-step forces, torques, energies, and virials must match to 1e-4 of their largest value
  (a few forms with near-singular geometry allow up to 1e-2).
- Short trajectories that use the same random numbers must match closely.
- Long runs must agree within five standard errors, and NVE runs must conserve energy.

| Check | Contents |
|---|---|
| `forces` | 25 pair potentials, DPD, bonds, angles, dihedrals, impropers, special pairs, external fields, walls |
| `aniso` | Gay-Berne, dipole, YLZ, ALJ, patchy, friction, Tersoff, square density, RevCross |
| `nlist` | Cell, Stencil, and Tree neighbor lists |
| `pppm` | PPPM electrostatics (orders 3, 5, 7) |
| `polymer` | Kremer-Grest melt with NVE, Langevin, NPT, and Bussi dynamics |
| `lj` | LJ liquid, NVE |
| `rigid` | Rigid bodies |
| `manifold` | RATTLE integrators on seven manifolds |
| `flow` | Reverse perturbation (Mueller-Plathe) flow |
| `mpcd` | MPCD streaming, collisions, walls, fillers, flows, embedded chain |

`--pytest` runs HOOMD-blue's test suite on both devices and reports tests that fail on the GPU but
pass on the CPU. Many tests fail on both devices because they assume a double precision build;
those do not count. A short list of known single precision cases is reported separately, each
with its explanation (see `PYTEST_KNOWN_GPU_ONLY` in [`validate.py`](validate/validate.py)).

## Benchmark

```bash
python metal/bench/benchmark.py                  # bead-spring melts, 6,400 to 409,600 beads
python metal/bench/benchmark.py --system lj      # LJ liquid
python metal/bench/benchmark.py --device cpu     # one CPU core, for comparison
python metal/bench/benchmark.py --sustain 15     # 15 minutes, to see thermal throttling
```

Close other GPU-heavy applications (trajectory viewers, games, video) before measuring.
Measurements on an M4 Mac mini (10-core GPU, 120 GB/s):

| System | Particles | GPU steps/s | CPU steps/s | Speedup |
|---|---|---|---|---|
| Kremer-Grest melt (50-bead chains), Langevin | 6,400 | 5,064 | 959 | 5.3x |
| | 25,600 | 2,505 | 245 | 10x |
| | 102,400 | 984 | 60 | 16x |
| | 409,600 | 302 | 14 | 21x |
| LJ liquid (r_cut = 2.5), Bussi NVT | 32,768 | 1,449 | 57 | 25x |

Large systems are limited by memory bandwidth and scale with it across Macs. Small systems are
limited by about one CPU-GPU synchronization per time step (the neighbor list check), which a
bigger GPU does not shorten. MacBook Airs have no fan and slow down after a few minutes of full
load; keep them on power and start long runs with `caffeinate -i` so the Mac does not sleep.

## Runtime settings

| Environment variable | Effect |
|---|---|
| `HOOMD_METAL_SPIN_US` | Microseconds the CPU busy-waits for the GPU before sleeping (default 5000). Lower values use less CPU power, which may help on a MacBook Air, at some cost in speed. |
| `HOOMD_METAL_PROFILE=1` | Print GPU time per kernel at exit (runs each kernel separately, so slower). |
| `HOOMD_METAL_STATS=1` | Print command buffer and synchronization statistics at exit. |
| `HOOMD_METAL_SYNC=1` | Wait for every kernel and check errors (slow; for debugging). |

## How it works

HOOMD-blue's C++ engine talks to the GPU through the HIP runtime API. HOOMD-metal implements the
part of that API HOOMD uses on top of Apple's Metal ([`hoomd/mtl`](../hoomd/mtl)), so the engine
runs unchanged:

- **Memory.** Apple silicon has unified memory, so every allocation is shared between CPU and GPU
  and most host-device copies become free.
- **Kernel launches** are encoded into Metal command buffers, ordered with fences, and committed
  early so the GPU works while the CPU prepares the next step.
- **Kernels.** Each CUDA kernel file (`.cu`) has a `.metal` counterpart with Metal Shading
  Language kernels and their C++ launchers. CUDA templates become Metal function constants. The
  kernels are embedded in the library and compiled for the Mac's GPU on first use.
- **Libraries** that have no Metal version are reimplemented: sorting, scans, and reductions
  (`Primitives.h`, replacing CUB and thrust), FFTs for PPPM (`HipFFT.metal`), a bounding volume
  hierarchy for the tree neighbor list, and a Philox random number generator that reproduces
  HOOMD-blue's random streams bit for bit.

## License

New HOOMD-metal code is released under the MIT License ([LICENSE-MIT](../LICENSE-MIT)): the
Metal runtime and tools in `hoomd/mtl` (except `WarpTools.h`), everything in `metal/`, and
`CMake/hoomd/HOOMDMetalSetup.cmake`. `hoomd/mtl/Philox.h` follows OpenRAND's Philox generator
(MIT License, Copyright (c) 2023, Michigan State University).

HOOMD-blue's code remains under its BSD 3-Clause License ([LICENSE](../LICENSE)). That includes
the Metal versions of HOOMD-blue's CUDA kernels (the `.metal` and `.metal.inc` files outside
`hoomd/mtl`) and every change to an existing HOOMD-blue file. Each file's header states which
license applies.

The HOOMD-metal logos and social preview image (`sphinx-doc/hoomdmetal-*`) are derived from the
HOOMD-blue logo, with the letter t from the hoomd-rs logo, and remain under the BSD 3-Clause
License.
