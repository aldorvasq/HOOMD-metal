# Copyright (c) 2026 Aldo Vasquez-Briceno.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

"""Binary Lennard-Jones liquid (r_cut = 2.5) with NVE dynamics.

Runs 4096 particles for 5050 steps. The first 50 steps must match closely, the later samples
must agree statistically, and the total energy must be conserved.
"""

import numpy as np

from .common import device, lattice

N_SIDE = 16
N = N_SIDE**3


def run(device_name):
    import hoomd

    L = (N / 0.8) ** (1 / 3)
    rng = np.random.default_rng(42)
    vel = rng.normal(scale=1.0, size=(N, 3))
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = ["A", "B"]
    snap.particles.typeid[:] = np.arange(N) % 2
    snap.configuration.box = [L, L, L, 0, 0, 0]
    snap.particles.position[:] = lattice(N_SIDE, L) + rng.normal(scale=0.05, size=(N, 3))
    snap.particles.velocity[:] = vel - vel.mean(axis=0)

    sim = hoomd.Simulation(device=device(device_name), seed=1)
    sim.create_state_from_snapshot(snap)
    sim.always_compute_pressure = True
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.3), default_r_cut=2.5,
                          mode="shift")
    lj.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
    lj.params[("A", "B")] = dict(epsilon=0.8, sigma=0.9)
    lj.params[("B", "B")] = dict(epsilon=0.5, sigma=0.85)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(dt=0.002, methods=[nve], forces=[lj])
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    def sample():
        return [thermo.kinetic_energy, thermo.potential_energy, thermo.pressure]

    early, late = [], []
    for _ in range(10):
        sim.run(5)
        early.append(sample())
    for _ in range(20):
        sim.run(250)
        late.append(sample())
    return {"early": np.array(early), "late": np.array(late)}


def compare(check):
    check.value("early", rtol=1e-4)
    for column, name in enumerate(("kinetic_energy", "potential_energy", "pressure")):
        check.mean("late", column=column, nsigma=5, rel_floor=1e-3, label=name)
    for device_name in ("cpu", "gpu"):
        data = getattr(check, device_name)
        if "early" in data and "late" in data:
            energy = np.concatenate([data["early"], data["late"]])[:, :2].sum(axis=1)
            check.below(f"energy_drift_{device_name} (per particle)",
                        np.max(np.abs(energy - energy[0])) / N, 1e-3)
