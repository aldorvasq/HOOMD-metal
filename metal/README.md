# HOOMD-metal

HOOMD-metal is [HOOMD-blue](https://github.com/glotzerlab/hoomd-blue) with a GPU backend for
Apple silicon Macs. Scripts written for HOOMD-blue run unchanged: `hoomd.device.GPU()` selects
the Mac's GPU. It is an unofficial fork and is not supported by the HOOMD-blue developers.

## Requirements

- An Apple silicon Mac (M1 or newer; any variant).
- macOS 15 (Sequoia) or newer.
- Apple's command line tools: `xcode-select --install`.
- An arm64 conda, such as [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda for
  Apple silicon. An Intel conda running under Rosetta does not work.

Intel Macs cannot use the GPU backend. On an Intel Mac `hoomd.device.GPU()` raises an error that
says to use `hoomd.device.CPU()`.

## Install

```bash
git clone --recursive https://github.com/aldorvasq/HOOMD-metal.git
cd HOOMD-metal
metal/install.sh --validate
conda activate hoomd-metal
```

`metal/install.sh` creates the conda environment `hoomd-metal` from
[`environment.yml`](environment.yml), builds HOOMD-metal, installs it into the environment, runs a
short GPU test, and with `--validate` runs the CPU vs GPU checks. The build takes about 5 minutes
on an M4 Mac mini. Options: `--name ENV` for another environment name, `--jobs N` to limit
parallel compiles on machines with little memory, `--no-mpcd` to skip MPCD.

To update, `git pull` and run `metal/install.sh` again. To uninstall, run
`conda env remove -n hoomd-metal`.

## Check this Mac

```bash
python metal/validate/validate.py           # 10 checks, 2-3 minutes
python metal/validate/validate.py --pytest  # also HOOMD's test suite (30 s more)
```

Each check runs the same simulation with `hoomd.device.CPU()` and `hoomd.device.GPU()` and
compares the results. Single-step forces, torques, energies, and virials must match to about
1e-4 or better. Short trajectories that use the same random numbers must match closely. Long
runs must agree within their statistical error, and NVE runs must conserve energy. The checks
cover:

| Check | Contents |
|---|---|
| `forces` | 25 pair potentials, DPD, bonds, angles, dihedrals, impropers, special pairs, external fields, walls |
| `aniso` | Gay-Berne, dipole, YLZ, ALJ, patchy, friction, Tersoff, square density, RevCross |
| `nlist` | Cell, Stencil, and Tree neighbor lists |
| `pppm` | PPPM electrostatics |
| `polymer` | Kremer-Grest melt with NVE, Langevin, NPT, and Bussi dynamics |
| `lj` | LJ liquid, NVE |
| `rigid` | Rigid bodies |
| `manifold` | RATTLE integrators on seven manifolds |
| `flow` | Reverse perturbation (Mueller-Plathe) flow |
| `mpcd` | MPCD streaming, collisions, walls, fillers, flows, embedded chain |

`--pytest` runs HOOMD's own test suite on both devices and reports tests that fail on the GPU but
pass on the CPU. Many tests fail on both devices because they assume a double precision build;
those do not count.

## Performance

Measured with `metal/bench/benchmark.py` on an M4 Mac mini (10-core GPU, 120 GB/s memory
bandwidth), against one core of the same Mac's CPU:

| System | Particles | GPU steps/s | CPU steps/s | Speedup |
|---|---|---|---|---|
| Kremer-Grest melt (50-bead chains), Langevin | 6,400 | 5,064 | 959 | 5.3x |
| | 25,600 | 2,505 | 245 | 10x |
| | 102,400 | 984 | 60 | 16x |
| | 409,600 | 302 | 14 | 21x |
| LJ liquid (r_cut = 2.5), Bussi NVT | 32,768 | 1,449 | 57 | 25x |

Measure your own Mac with `python metal/bench/benchmark.py` (`--device cpu` for the CPU column,
`--sustain 15` to see whether throughput drops during long runs).

What to expect on other Macs:

- **Large systems** are limited by memory bandwidth, so they speed up roughly with it: M Pro
  chips have 150-273 GB/s, M Max 300-546 GB/s, and M Ultra 800-819 GB/s. An Ultra appears as one
  GPU and HOOMD uses all of its cores.
- **Small systems** (below about 10,000 particles) are limited by a fixed cost per step, so faster
  GPUs help little. Run several independent simulations at once to use a large GPU fully.
- **MacBook Air** models have no fan and slow down after a few minutes of full GPU load. Results
  are unchanged. Keep the Mac on power, and start long runs with `caffeinate -i` so it does not
  sleep.
- **Memory:** a bead-spring melt uses about 1.2 KB per bead, so 8 GB holds about 3 million beads.

## Limitations

- Single precision only. Apple GPUs have no double precision arithmetic.
- No HPMC. HOOMD-blue requires a double precision build for hard particle Monte Carlo.
- No MPI. One HOOMD-metal process uses one GPU, which on Apple silicon is the whole GPU.

## Runtime settings

| Environment variable | Effect |
|---|---|
| `HOOMD_METAL_SPIN_US` | Microseconds the CPU busy-waits for the GPU before sleeping (default 5000). Lower values use less CPU power, which may help on a MacBook Air, at some cost in speed. |
| `HOOMD_METAL_PROFILE=1` | Print GPU time per kernel at exit (runs each kernel separately, so slower). |
| `HOOMD_METAL_STATS=1` | Print command buffer and synchronization statistics at exit. |
| `HOOMD_METAL_SYNC=1` | Wait for every kernel and check errors (slow; for debugging). |

## How it works

`hoomd/mtl` implements the part of the HIP runtime that HOOMD uses on top of Metal. Each CUDA
kernel file (`.cu`) has a `.metal` counterpart with the Metal Shading Language kernels and their
C++ launchers. The kernels are compiled for the Mac's GPU when a simulation first uses them.
