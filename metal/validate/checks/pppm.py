# Copyright (c) 2026 Aldo Vasquez-Briceno.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

"""PPPM long-range electrostatics (assignment orders 3, 5, and 7) in a triclinic box.

Compares the reciprocal-space and real-space forces, the pressure tensor, and the energy of 4000
charges.
"""

import numpy as np

from .common import device

N = 4000
ORDERS = (3, 5, 7)


def run(device_name):
    import hoomd

    rng = np.random.default_rng(7)
    L = 20.0
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = ["A"]
    snap.configuration.box = [L, L * 1.1, L * 0.9, 0.1, 0.0, 0.05]
    n = int(np.ceil(N ** (1 / 3)))
    frac = (np.array(np.meshgrid(*(np.arange(n),) * 3)).reshape(3, -1).T[:N] + 0.5) / n - 0.5
    frac += rng.uniform(-0.1, 0.1, size=frac.shape) / n
    box = hoomd.Box.from_box(snap.configuration.box)
    snap.particles.position[:] = frac @ box.to_matrix().T
    q = rng.choice([-1.0, 1.0], N)
    snap.particles.charge[:] = q - q.mean()
    snap.wrap()

    dev = device(device_name)
    results = {}
    for order in ORDERS:
        try:
            sim = hoomd.Simulation(device=dev, seed=1)
            sim.create_state_from_snapshot(snap)
            sim.always_compute_pressure = True
            ewald, coulomb = hoomd.md.long_range.pppm.make_pppm_coulomb_forces(
                nlist=hoomd.md.nlist.Cell(buffer=0.4), resolution=(32, 36, 30), order=order,
                r_cut=3.0)
            method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.Null())
            sim.operations.integrator = hoomd.md.Integrator(dt=0.001, methods=[method],
                                                            forces=[ewald, coulomb])
            thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
            sim.operations.computes.append(thermo)
            sim.run(0)
            results[f"order{order}/forces"] = np.array(coulomb.forces)
            results[f"order{order}/ewald_forces"] = np.array(ewald.forces)
            results[f"order{order}/pressure_tensor"] = np.array(thermo.pressure_tensor)
            results[f"order{order}/energy"] = np.array([coulomb.energy])
        except Exception as e:
            results[f"order{order}/error"] = np.array(repr(e))
    return results


def compare(check):
    for order in ORDERS:
        check.close(f"order{order}/forces", 1e-4)
        check.close(f"order{order}/ewald_forces", 1e-4)
        check.close(f"order{order}/pressure_tensor", 1e-3)
        # The total is a small difference of large self and reciprocal energies, so single
        # precision limits its absolute accuracy: compare per particle.
        check.value(f"order{order}/energy", atol=1e-4 * N)
