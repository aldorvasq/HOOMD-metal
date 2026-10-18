# Copyright (c) 2026 Aldo Vasquez-Briceno.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

"""Kremer-Grest bead-spring melt (WCA + FENE + harmonic angles) with four integration methods.

Runs 5760 beads (288 chains of 20) for 21000 steps with NVE, Langevin, NPT (MTTK), and Bussi
dynamics. The first steps must match closely; the later samples must agree statistically. NVE
must also conserve energy.
"""

import numpy as np

from .common import device

MODES = ("nve", "langevin", "npt", "bussi")
COLUMNS = ("kinetic_energy", "pair_energy", "bond_energy", "angle_energy", "pressure", "volume")
# run lengths between samples: 5 early samples, then 20 samples 1000 steps apart
EARLY = [1, 4, 5, 10, 20]
LATE = [60, 100, 300, 500] + [1000] * 20
N_LATE_SAMPLES = 20


def make_snapshot(hoomd):
    n_chain_x, n_y, n_z, n_beads = 2, 12, 12, 20
    bond, spacing = 0.97, 1.12
    Lx = n_chain_x * n_beads * bond
    Ly, Lz = n_y * spacing, n_z * spacing
    pos, bonds, angles = [], [], []
    for cx in range(n_chain_x):
        for iy in range(n_y):
            for iz in range(n_z):
                start = len(pos)
                for b in range(n_beads):
                    pos.append([(cx * n_beads + b + 0.5) * bond - Lx / 2,
                                (iy + 0.5) * spacing - Ly / 2, (iz + 0.5) * spacing - Lz / 2])
                    if b > 0:
                        bonds.append([start + b - 1, start + b])
                    if b > 1:
                        angles.append([start + b - 2, start + b - 1, start + b])
    N = len(pos)
    rng = np.random.default_rng(7)
    vel = rng.normal(scale=1.0, size=(N, 3))
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = ["A"]
    snap.configuration.box = [Lx, Ly, Lz, 0, 0, 0]
    snap.particles.position[:] = np.array(pos) + rng.normal(scale=0.02, size=(N, 3))
    snap.particles.velocity[:] = vel - vel.mean(axis=0)
    snap.bonds.N = len(bonds)
    snap.bonds.types = ["backbone"]
    snap.bonds.group[:] = bonds
    snap.angles.N = len(angles)
    snap.angles.types = ["bend"]
    snap.angles.group[:] = angles
    return snap


def simulate(hoomd, dev, snap, mode):
    sim = hoomd.Simulation(device=dev, seed=11)
    sim.create_state_from_snapshot(snap)
    sim.always_compute_pressure = True
    md = hoomd.md
    nl = md.nlist.Cell(buffer=0.4, exclusions=("bond",))
    wca = md.pair.LJ(nlist=nl, default_r_cut=2 ** (1 / 6), mode="shift")
    wca.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
    fene = md.bond.FENEWCA()
    fene.params["backbone"] = dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)
    angle = md.angle.Harmonic()
    angle.params["bend"] = dict(k=1.5, t0=np.pi)
    everything = hoomd.filter.All()
    if mode == "nve":
        method = md.methods.ConstantVolume(filter=everything)
    elif mode == "langevin":
        method = md.methods.Langevin(filter=everything, kT=1.0, default_gamma=0.5)
    elif mode == "npt":
        method = md.methods.ConstantPressure(
            filter=everything, tauS=1.0, S=5.0, couple="xyz",
            thermostat=md.methods.thermostats.MTTK(kT=1.0, tau=0.5))
    else:
        method = md.methods.ConstantVolume(filter=everything,
                                           thermostat=md.methods.thermostats.Bussi(kT=1.0))
    sim.operations.integrator = md.Integrator(dt=0.005, methods=[method],
                                              forces=[wca, fene, angle])
    thermo = md.compute.ThermodynamicQuantities(filter=everything)
    sim.operations.computes.append(thermo)

    def sample():
        return [thermo.kinetic_energy, wca.energy, fene.energy, angle.energy, thermo.pressure,
                thermo.volume]

    early, late = [], []
    for steps in EARLY:
        sim.run(steps)
        early.append(sample())
    for steps in LATE:
        sim.run(steps)
        late.append(sample())
    return np.array(early), np.array(late)[-N_LATE_SAMPLES:]


def run(device_name):
    import hoomd

    dev = device(device_name)
    snap = make_snapshot(hoomd)
    results = {}
    for mode in MODES:
        try:
            results[f"{mode}/early"], results[f"{mode}/late"] = simulate(hoomd, dev, snap, mode)
        except Exception as e:
            results[f"{mode}/error"] = np.array(repr(e))
    return results


def compare(check):
    for mode in MODES:
        # identical initial conditions and random streams: the first 40 steps match closely
        check.value(f"{mode}/early", rtol=1e-4)
        for column, name in enumerate(COLUMNS):
            check.mean(f"{mode}/late", column=column, nsigma=5, rel_floor=1e-3,
                       label=f"{mode}/{name}")
    for device_name in ("cpu", "gpu"):
        late = getattr(check, device_name).get("nve/late")
        if late is not None:
            energy = late[:, 0] + late[:, 1] + late[:, 2] + late[:, 3]
            n_beads = 5760
            check.below(f"nve/energy_drift_{device_name} (per bead)",
                        np.max(np.abs(energy - energy[0])) / n_beads, 5e-3)
