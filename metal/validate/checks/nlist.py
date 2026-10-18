# Copyright (c) 2026 Aldo Vásquez-Briceño.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

"""Cell, Stencil, and Tree neighbor lists with very different cutoffs (colloid + solvent).

Compares LJ forces from each neighbor list type at step 0 and after 200 steps, and checks that
all three GPU neighbor lists give the same forces.
"""

import numpy as np

from .common import device, lattice

NLISTS = ("Cell", "Stencil", "Tree")


def make_snapshot(hoomd):
    rng = np.random.default_rng(5)
    L = 30.0
    n_big = 200
    big = lattice(6, L)[:n_big]
    small = lattice(26, L, jitter=0.05, rng=rng)
    d = small[:, None, :] - big[None, :, :]
    d -= L * np.round(d / L)
    small = small[np.min(np.sum(d * d, axis=2), axis=1) > 3.2**2][:12000]
    N = n_big + len(small)
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = ["big", "small"]
    snap.configuration.box = [L, L, L, 0, 0, 0]
    snap.particles.position[:] = np.concatenate([big, small])
    snap.particles.typeid[:] = [0] * n_big + [1] * len(small)
    snap.particles.velocity[:] = rng.normal(scale=0.5, size=(N, 3))
    snap.bonds.N = 100
    snap.bonds.types = ["b"]
    snap.bonds.group[:] = np.arange(n_big, n_big + 200).reshape(-1, 2)
    snap.wrap()
    return snap


def run(device_name):
    import hoomd

    dev = device(device_name)
    snap = make_snapshot(hoomd)
    results = {}
    for name in NLISTS:
        try:
            sim = hoomd.Simulation(device=dev, seed=3)
            sim.create_state_from_snapshot(snap)
            if name == "Cell":
                nl = hoomd.md.nlist.Cell(buffer=0.3, exclusions=["bond"])
            elif name == "Stencil":
                nl = hoomd.md.nlist.Stencil(buffer=0.3, cell_width=1.0, exclusions=["bond"])
            else:
                nl = hoomd.md.nlist.Tree(buffer=0.3, exclusions=["bond"])
            lj = hoomd.md.pair.LJ(nlist=nl, mode="shift")
            lj.params[("big", "big")] = dict(epsilon=1.0, sigma=5.0)
            lj.r_cut[("big", "big")] = 12.0
            lj.params[("big", "small")] = dict(epsilon=1.0, sigma=3.0)
            lj.r_cut[("big", "small")] = 7.0
            lj.params[("small", "small")] = dict(epsilon=1.0, sigma=1.0)
            lj.r_cut[("small", "small")] = 2.5
            harmonic = hoomd.md.bond.Harmonic()
            harmonic.params["b"] = dict(k=1.0, r0=1.0)
            method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
            sim.operations.integrator = hoomd.md.Integrator(dt=0.0005, methods=[method],
                                                            forces=[lj, harmonic])
            sim.run(0)
            results[f"{name}/forces"] = np.array(lj.forces)
            results[f"{name}/energies"] = np.array(lj.energies)
            sim.run(200)
            results[f"{name}/forces_200"] = np.array(lj.forces)
            results[f"{name}/positions_200"] = sim.state.get_snapshot().particles.position
        except Exception as e:
            results[f"{name}/error"] = np.array(repr(e))
    return results


def compare(check):
    for name in NLISTS:
        check.close(f"{name}/forces", 1e-4)
        check.close(f"{name}/energies", 1e-4)
        check.close(f"{name}/forces_200", 1e-3)
        check.close(f"{name}/positions_200", 1e-5)
    # every GPU neighbor list must find the same neighbors as the GPU cell list
    gpu = check.gpu
    for name in ("Stencil", "Tree"):
        if f"{name}/forces" in gpu and "Cell/forces" in gpu:
            a = np.asarray(gpu["Cell/forces"], dtype=np.float64)
            b = np.asarray(gpu[f"{name}/forces"], dtype=np.float64)
            check.below(f"gpu {name} vs gpu Cell forces (max diff / max)",
                        np.max(np.abs(a - b)) / np.max(np.abs(a)), 1e-4)
