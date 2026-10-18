"""Measure HOOMD-metal throughput (time steps per second) on this Mac.

Usage::

    python metal/bench/benchmark.py                   # bead-spring melts, 6.4k to 410k beads
    python metal/bench/benchmark.py --sizes 102400    # one system size
    python metal/bench/benchmark.py --system lj       # Lennard-Jones liquid (r_cut = 2.5)
    python metal/bench/benchmark.py --device cpu      # one CPU core, for comparison
    python metal/bench/benchmark.py --sustain 15      # 15 minutes, report every 30 s

``--sustain`` shows whether the Mac slows down under a long run (fanless laptops do). Close
other GPU-heavy apps (trajectory viewers, games, video) before measuring.
"""

import argparse
import os
import platform
import subprocess
import time

import hoomd
import numpy as np

DEFAULT_SIZES = {"kg": [6400, 25600, 102400, 409600], "lj": [4096, 32768, 262144]}


def kremer_grest(sim, n_target):
    """Melt of 50-bead chains (WCA + FENE) at bead density 0.85, Langevin dynamics."""
    beads, bond, spacing = 50, 0.97, 1.12
    n_side = max(1, round(np.sqrt(n_target / (2 * beads))))
    n_x, n_y, n_z = 2, n_side, n_side
    Lx, Ly, Lz = n_x * beads * bond, n_y * spacing, n_z * spacing
    N = n_x * n_y * n_z * beads
    index = np.arange(N)
    chain, b = index // beads, index % beads
    cx, rest = chain // (n_y * n_z), chain % (n_y * n_z)
    iy, iz = rest // n_z, rest % n_z
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = ["A"]
    snap.configuration.box = [Lx, Ly, Lz, 0, 0, 0]
    snap.particles.position[:] = np.stack(
        [(cx * beads + b + 0.5) * bond - Lx / 2, (iy + 0.5) * spacing - Ly / 2,
         (iz + 0.5) * spacing - Lz / 2], axis=1)
    snap.bonds.N = N - n_x * n_y * n_z
    snap.bonds.types = ["backbone"]
    snap.bonds.group[:] = np.stack([index[b > 0] - 1, index[b > 0]], axis=1)
    sim.create_state_from_snapshot(snap)
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=1.0)
    nl = hoomd.md.nlist.Cell(buffer=0.4, exclusions=("bond",))
    wca = hoomd.md.pair.LJ(nlist=nl, default_r_cut=2 ** (1 / 6), mode="shift")
    wca.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
    fene = hoomd.md.bond.FENEWCA()
    fene.params["backbone"] = dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0, default_gamma=0.5)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.01, methods=[langevin],
                                                    forces=[wca, fene])
    return N


def lennard_jones(sim, n_target):
    """LJ liquid at density 0.84 and kT = 1.0 (r_cut = 2.5, buffer 0.4), NVT dynamics."""
    n_side = max(2, round(n_target ** (1 / 3)))
    N = n_side**3
    L = (N / 0.84) ** (1 / 3)
    x = (np.arange(n_side) + 0.5) * L / n_side - L / 2
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = ["A"]
    snap.configuration.box = [L, L, L, 0, 0, 0]
    snap.particles.position[:] = np.array(np.meshgrid(x, x, x)).reshape(3, -1).T
    sim.create_state_from_snapshot(snap)
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=1.0)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=2.5)
    lj.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
    nvt = hoomd.md.methods.ConstantVolume(
        filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.0))
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005, methods=[nvt], forces=[lj])
    return N


def warm_up(sim):
    """Equilibrate briefly and let the kernel autotuners finish (they time kernels)."""
    sim.run(500)
    for _ in range(100):
        if sim.operations.is_tuning_complete:
            break
        sim.run(100)


def timed_run(sim, steps):
    start = time.perf_counter()
    sim.run(steps)
    return steps / (time.perf_counter() - start)


def describe(device):
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                          text=True).stdout.strip()
    memory = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                text=True).stdout or 0)
    print(f"{chip}, {memory / 2**30:.0f} GB, macOS {platform.mac_ver()[0]}, "
          f"hoomd {hoomd.version.version} ({hoomd.version.gpu_platform or 'CPU'})")
    print(f"device: {device.device.strip()}")
    spin = os.environ.get("HOOMD_METAL_SPIN_US")
    if spin is not None:
        print(f"HOOMD_METAL_SPIN_US={spin}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--system", choices=("kg", "lj"), default="kg",
                        help="kg: Kremer-Grest bead-spring melt (default), lj: LJ liquid")
    parser.add_argument("--sizes", type=int, nargs="+",
                        help="approximate particle counts (default: several)")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="target length of each measurement (default 10)")
    parser.add_argument("--sustain", type=float, metavar="MINUTES",
                        help="run the first size for this many minutes and report every 30 s")
    args = parser.parse_args()

    device = hoomd.device.GPU() if args.device == "gpu" else hoomd.device.CPU()
    describe(device)
    build = kremer_grest if args.system == "kg" else lennard_jones
    sizes = args.sizes or DEFAULT_SIZES[args.system]
    if args.sustain:
        sizes = sizes[:1]

    if args.sustain:
        print(f"\nRunning {args.sustain:g} minutes:")
    else:
        print(f"\n{'particles':>10s} {'steps/s':>10s} {'M particle-steps/s':>19s}")
    for n_target in sizes:
        sim = hoomd.Simulation(device=device, seed=3)
        N = build(sim, n_target)
        warm_up(sim)
        # size the measurement from a short trial run
        tps = timed_run(sim, 200)
        steps = max(200, int(tps * args.seconds))
        if not args.sustain:
            tps = timed_run(sim, steps)
            print(f"{N:10d} {tps:10.1f} {N * tps / 1e6:19.1f}", flush=True)
            continue

        steps = max(100, int(tps * 5))
        start = time.time()
        report, samples = start + 30, []
        while time.time() - start < args.sustain * 60:
            samples.append((time.time() - start, timed_run(sim, steps)))
            if time.time() >= report:
                recent = [t for when, t in samples if when >= report - 30 - start]
                print(f"{(time.time() - start) / 60:5.1f} min {np.mean(recent):10.1f} steps/s",
                      flush=True)
                report += 30
        first = np.mean([t for when, t in samples if when < 60])
        last = np.mean([t for when, t in samples if when >= samples[-1][0] - 60])
        print(f"\n{N} particles: {first:.1f} steps/s in the first minute, {last:.1f} in the last "
              f"minute ({100 * (last / first - 1):+.0f}%)")


if __name__ == "__main__":
    main()
