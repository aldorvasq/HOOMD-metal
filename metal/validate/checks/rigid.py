# Copyright (c) 2026 Aldo Vasquez-Briceno.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

"""Rigid bodies (three-bead rods) mixed with free beads, integrated with NVE.

Compares the step-0 forces, torques, and pressure tensor, the energies over 200 steps, and the
final positions and orientations of 343 rods and 343 free beads.
"""

import numpy as np

from .common import device, lattice


def run(device_name):
    import hoomd

    rng = np.random.default_rng(3)
    n_bodies = n_free = 343
    L = 22.0
    centers = lattice(7, L)
    free = centers + L / 14
    free[free >= L / 2] -= L
    snap = hoomd.Snapshot()
    snap.particles.N = n_bodies + n_free
    snap.particles.types = ["R", "A", "B"]
    snap.configuration.box = [L, L, L, 0, 0, 0]
    snap.particles.position[:] = np.concatenate([centers, free])
    snap.particles.typeid[:] = [0] * n_bodies + [2] * n_free
    snap.particles.mass[:] = [3.0] * n_bodies + [1.0] * n_free
    snap.particles.moment_inertia[:n_bodies] = [2.0, 2.0, 0.1]
    q = rng.normal(size=(n_bodies, 4))
    snap.particles.orientation[:n_bodies] = q / np.linalg.norm(q, axis=1)[:, None]
    snap.particles.velocity[:] = rng.normal(scale=0.5, size=(n_bodies + n_free, 3))
    snap.particles.angmom[:n_bodies] = np.concatenate(
        [np.zeros((n_bodies, 1)), rng.normal(scale=0.3, size=(n_bodies, 3))], axis=1)

    sim = hoomd.Simulation(device=device(device_name), seed=11)
    sim.create_state_from_snapshot(snap)
    rigid = hoomd.md.constrain.Rigid()
    rigid.body["R"] = {
        "constituent_types": ["A", "A", "A"],
        "positions": [(-0.9, 0, 0), (0, 0, 0), (0.9, 0, 0)],
        "orientations": [(1, 0, 0, 0)] * 3,
    }
    rigid.create_bodies(sim.state)
    sim.always_compute_pressure = True
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4, exclusions=["body"]),
                          default_r_cut=2.5)
    for a in ("R", "A", "B"):
        for b in ("R", "A", "B"):
            lj.params[(a, b)] = dict(epsilon=0.0 if "R" in (a, b) else 1.0, sigma=1.0)
    centers_and_free = hoomd.filter.Rigid(("center", "free"))
    method = hoomd.md.methods.ConstantVolume(filter=centers_and_free)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.001, methods=[method], forces=[lj],
                                                    rigid=rigid, integrate_rotational_dof=True)
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=centers_and_free)
    sim.operations.computes.append(thermo)
    sim.run(0)
    results = {
        "step0/forces": np.array(lj.forces),
        "step0/torques": np.array(lj.torques),
        "step0/pressure_tensor": np.array(thermo.pressure_tensor),
    }
    energies = []
    for _ in range(20):
        sim.run(10)
        energies.append([thermo.kinetic_energy, thermo.potential_energy,
                         thermo.translational_kinetic_energy, thermo.rotational_kinetic_energy])
    results["energies"] = np.array(energies)
    s = sim.state.get_snapshot()
    results["positions_200"] = s.particles.position
    results["orientations_200"] = s.particles.orientation
    return results


def compare(check):
    check.close("step0/forces", 1e-4)
    check.close("step0/torques", 1e-4)
    check.close("step0/pressure_tensor", 1e-4)
    check.value("energies", rtol=1e-3)
    check.close("positions_200", 1e-4)
    check.close("orientations_200", 1e-3)
    for device_name in ("cpu", "gpu"):
        e = getattr(check, device_name).get("energies")
        if e is not None:
            total = e[:, 0] + e[:, 1]
            check.below(f"energy_drift_{device_name} (relative)",
                        np.max(np.abs(total - total[0])) / abs(total[0]), 1e-2)
