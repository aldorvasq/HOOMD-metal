# Copyright (c) 2026 Aldo Vásquez-Briceño.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

"""Kremer-Grest bead-spring polymer melt on the Mac GPU.

This is an ordinary HOOMD-blue script: nothing in it is specific to HOOMD-metal. The same file
runs on an NVIDIA or AMD GPU with upstream HOOMD-blue, and on the Mac's GPU with HOOMD-metal.

    python polymer_melt.py                     # 500 chains of 50 beads on the GPU
    python polymer_melt.py --device cpu        # the same run on one CPU core, for comparison
    python polymer_melt.py --chains 2000 --steps 20000

It writes the trajectory to melt.gsd, which OVITO, VMD (with the GSD plugin), and freud read.
"""

import argparse
import time

import hoomd
import numpy as np

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--chains", type=int, default=500, help="number of chains (default 500)")
parser.add_argument("--length", type=int, default=50, help="beads per chain (default 50)")
parser.add_argument("--steps", type=int, default=10000, help="time steps (default 10000)")
parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
args = parser.parse_args()

# Pick the device. hoomd.device.auto_select() would also choose the Mac GPU.
device = hoomd.device.GPU() if args.device == "gpu" else hoomd.device.CPU()
print("Running on:", device.device.strip())

# --- Build straight chains on a lattice at bead density 0.85 -----------------------------------
n_chains, n_beads = args.chains, args.length
N = n_chains * n_beads
L = (N / 0.85) ** (1 / 3)
n_side = int(np.ceil(n_chains ** (1 / 2)))
rng = np.random.default_rng(42)
positions, bonds = [], []
for c in range(n_chains):
    # each chain is a random walk with unit steps that starts at a lattice site in the xy plane
    start = np.array([(c % n_side + 0.5) * L / n_side, (c // n_side + 0.5) * L / n_side,
                      rng.uniform(0, L)])
    steps = rng.normal(size=(n_beads, 3))
    steps /= np.linalg.norm(steps, axis=1)[:, None]
    chain = start + 0.97 * np.cumsum(steps, axis=0)
    first = len(positions)
    positions.extend(chain)
    bonds.extend([first + b, first + b + 1] for b in range(n_beads - 1))
positions = np.array(positions) % L - L / 2

snapshot = hoomd.Snapshot()
snapshot.configuration.box = [L, L, L, 0, 0, 0]
snapshot.particles.N = N
snapshot.particles.types = ["A"]
snapshot.particles.position[:] = positions
snapshot.bonds.N = len(bonds)
snapshot.bonds.types = ["backbone"]
snapshot.bonds.group[:] = bonds

sim = hoomd.Simulation(device=device, seed=1)
sim.create_state_from_snapshot(snapshot)

# --- Force field: WCA repulsion between beads, FENE bonds along the chains --------------------
nlist = hoomd.md.nlist.Cell(buffer=0.4, exclusions=["bond"])
wca = hoomd.md.pair.LJ(nlist=nlist, default_r_cut=2 ** (1 / 6), mode="shift")
wca.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
fene = hoomd.md.bond.FENEWCA()
fene.params["backbone"] = dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)

# The random walks overlap. Push the beads apart in two stages before switching on WCA + FENE:
# 1. a soft repulsion that stays finite at any distance, with harmonic bonds holding the chains
#    together (FENE bonds would break while beads move apart);
# 2. an energy minimization (FIRE) with WCA, which removes the last close contacts.
harmonic = hoomd.md.bond.Harmonic()
harmonic.params["backbone"] = dict(k=100.0, r0=0.97)
soft = hoomd.md.pair.DPDConservative(nlist=nlist, default_r_cut=1.0)
soft.params[("A", "A")] = dict(A=100.0)
sim.operations.integrator = hoomd.md.Integrator(
    dt=0.005, forces=[soft, harmonic],
    methods=[hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0, default_gamma=1.0)])
sim.run(3000)
fire = hoomd.md.minimize.FIRE(dt=0.002, force_tol=0.1, angmom_tol=0.1, energy_tol=1e-6,
                              forces=[wca, harmonic],
                              methods=[hoomd.md.methods.ConstantVolume(hoomd.filter.All())])
sim.operations.integrator = fire
while not fire.converged and sim.timestep < 50000:
    sim.run(500)
fire.forces.clear()

# --- Production: Kremer-Grest (WCA + FENE) with Langevin dynamics at kT = 1 ---------------------
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0, default_gamma=0.5)
sim.operations.integrator = hoomd.md.Integrator(dt=0.01, methods=[langevin], forces=[wca, fene])
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)

thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermo)
logger = hoomd.logging.Logger(categories=["scalar"])
logger.add(thermo, quantities=["kinetic_temperature", "pressure", "potential_energy"])
logger.add(sim, quantities=["timestep", "tps"])
sim.operations.writers.append(hoomd.write.Table(trigger=hoomd.trigger.Periodic(args.steps // 10),
                                                logger=logger))
sim.operations.writers.append(hoomd.write.GSD(filename="melt.gsd", mode="wb",
                                              trigger=hoomd.trigger.Periodic(args.steps // 20)))

start = time.perf_counter()
sim.run(args.steps)
elapsed = time.perf_counter() - start
print(f"\n{N} beads, {args.steps} steps in {elapsed:.1f} s: {args.steps / elapsed:.0f} steps/s")
