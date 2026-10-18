# Copyright (c) 2026 Aldo Vásquez-Briceño.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

"""Anisotropic, patchy, friction, and many-body potentials.

Evaluates each potential on 2000 randomly oriented particles and compares per-particle forces,
torques, energies, and virials.
"""

import numpy as np

from .common import device, lattice

N = 2000
L = 16.0
PAIRS = [("A", "A"), ("A", "B"), ("B", "B")]
TOLERANCE = 1e-4

CUBE_VERTS = [(x, y, z) for x in (-0.2, 0.2) for y in (-0.2, 0.2) for z in (-0.2, 0.2)]
CUBE_FACES = [[0, 2, 6], [6, 4, 0], [5, 0, 4], [5, 1, 0], [5, 4, 6], [5, 6, 7], [3, 2, 0],
              [3, 0, 1], [3, 6, 2], [3, 7, 6], [3, 1, 5], [3, 5, 7]]


def make_snapshot(hoomd):
    rng = np.random.default_rng(11)
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = ["A", "B"]
    snap.configuration.box = [L, L, L, 0, 0, 0]
    snap.particles.position[:] = lattice(13, L, jitter=0.1, rng=rng)[:N]
    snap.particles.typeid[:] = rng.integers(0, 2, N)
    q = rng.normal(size=(N, 4))
    snap.particles.orientation[:] = q / np.linalg.norm(q, axis=1)[:, None]
    snap.particles.moment_inertia[:] = [1.0, 1.0, 1.0]
    snap.particles.velocity[:] = rng.normal(scale=0.5, size=(N, 3))
    snap.particles.angmom[:] = np.concatenate(
        [np.zeros((N, 1)), rng.normal(scale=0.3, size=(N, 3))], axis=1)
    snap.particles.charge[:] = rng.choice([-1.0, 1.0], N)
    snap.particles.diameter[:] = 1.0
    return snap


def cases(hoomd):
    md = hoomd.md
    aniso = md.pair.aniso
    result = {}

    def nlist():
        return md.nlist.Cell(buffer=0.3)

    def gay_berne():
        p = aniso.GayBerne(nlist=nlist(), default_r_cut=3.0)
        for pr in PAIRS:
            p.params[pr] = dict(epsilon=1.0, lperp=0.45, lpar=0.6)
        return p

    def dipole():
        p = aniso.Dipole(nlist=nlist(), default_r_cut=3.0)
        for pr in PAIRS:
            p.params[pr] = dict(A=1.0, kappa=1.0)
        p.mu["A"] = (1.0, 0.5, 0.0)
        p.mu["B"] = (0.0, 0.0, 1.0)
        return p

    def ylz():
        p = aniso.YLZ(nlist=nlist(), default_r_cut=2.6)
        for pr in PAIRS:
            p.params[pr] = dict(eps=1.0, phi=0.2, beta=1.774532, rmin=1.12, twozeta=2)
        p.mu["A"] = (1.0, 0.0, 0.0)
        p.mu["B"] = (0.0, 1.0, 0.0)
        return p

    def alj():
        p = aniso.ALJ(nlist=nlist(), default_r_cut=2.5)
        for pr in PAIRS:
            p.params[pr] = dict(epsilon=1.0, sigma_i=0.4, sigma_j=0.4, alpha=1)
        for t in ("A", "B"):
            p.shape[t] = dict(vertices=CUBE_VERTS, faces=CUBE_FACES, rounding_radii=0.0)
        return p

    def patchy(cls, pair_params):
        def build():
            p = cls(nlist=nlist(), default_r_cut=2.5)
            for pr in PAIRS:
                p.params[pr] = dict(pair_params=pair_params,
                                    envelope_params=dict(alpha=np.pi / 3, omega=20))
            p.directors["A"] = [(1, 0, 0), (0, 0, 1)]
            p.directors["B"] = [(0, 1, 0)]
            return p

        return build

    def friction(cls, params, full_nlist=False):
        def build():
            nl = nlist()
            p = cls(nlist=nl, default_r_cut=2.0 ** (1 / 6))
            for pr in PAIRS:
                p.params[pr] = params
            if not full_nlist:
                return p
            # A zero-strength triplet force makes the shared neighbor list full on both devices,
            # so that the random forces at kT > 0 are drawn for the same pairs.
            sq = md.many_body.SquareDensity(nlist=nl, default_r_cut=0.1)
            for pr in PAIRS:
                sq.params[pr] = dict(A=0.0, B=0.0)
            return [p, sq]

        return build

    def tersoff():
        p = md.many_body.Tersoff(nlist=nlist(), default_r_cut=1.6)
        for pr in PAIRS:
            p.params[pr] = dict(magnitudes=(2.0, 1.0), lambda3=5.0)
        return p

    def square_density():
        p = md.many_body.SquareDensity(nlist=nlist(), default_r_cut=1.6)
        for pr in PAIRS:
            p.params[pr] = dict(A=1.0, B=2.0)
        return p

    def revcross():
        p = md.many_body.RevCross(nlist=nlist(), default_r_cut=1.6)
        p.params[("A", "A")] = dict(sigma=0, n=0, epsilon=0, lambda3=0)
        p.params[("B", "B")] = dict(sigma=0, n=0, epsilon=0, lambda3=0)
        p.params[("A", "B")] = dict(sigma=1.0, n=10, epsilon=1.0, lambda3=1.0)
        return p

    friction_lj = md.pair.friction
    result["GayBerne"] = gay_berne
    result["Dipole"] = dipole
    result["YLZ"] = ylz
    result["ALJ"] = alj
    result["PatchyLJ"] = patchy(aniso.PatchyLJ, dict(epsilon=1.0, sigma=1.0))
    result["PatchyGaussian"] = patchy(aniso.PatchyGaussian, dict(epsilon=1.0, sigma=1.0))
    result["PatchyYukawa"] = patchy(aniso.PatchyYukawa, dict(epsilon=1.0, kappa=1.0))
    result["PatchyMie"] = patchy(aniso.PatchyMie, dict(epsilon=1.0, sigma=1.0, n=12, m=6))
    result["PatchyExpandedLJ"] = patchy(aniso.PatchyExpandedLJ,
                                        dict(epsilon=1.0, sigma=1.0, delta=0.2))
    result["FrictionLJLinear_kT0"] = friction(
        friction_lj.FrictionLJLinear, dict(epsilon=1.0, sigma=1.0, gamma_f=2.0, kT=0.0))
    result["FrictionLJCoulomb_kT0"] = friction(
        friction_lj.FrictionLJCoulomb, dict(epsilon=1.0, sigma=1.0, kappa_f=3.0, kT=0.0))
    result["FrictionLJCoulombNewton_kT0"] = friction(
        friction_lj.FrictionLJCoulombNewton,
        dict(epsilon=1.0, sigma=1.0, kappa_f=3.0, gamma_f=2.0, kT=0.0))
    result["FrictionLJLinear_kT1"] = friction(
        friction_lj.FrictionLJLinear, dict(epsilon=1.0, sigma=1.0, gamma_f=2.0, kT=1.0), True)
    result["FrictionLJCoulomb_kT1"] = friction(
        friction_lj.FrictionLJCoulomb, dict(epsilon=1.0, sigma=1.0, kappa_f=3.0, kT=1.0), True)
    result["Tersoff"] = tersoff
    result["SquareDensity"] = square_density
    result["RevCross"] = revcross
    return result


def run(device_name):
    import hoomd

    dev = device(device_name)
    snap = make_snapshot(hoomd)
    results = {}
    for name, build in cases(hoomd).items():
        try:
            sim = hoomd.Simulation(device=dev, seed=5)
            sim.create_state_from_snapshot(snap)
            sim.always_compute_pressure = True
            forces = build()
            forces = forces if isinstance(forces, list) else [forces]
            method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.Null())
            sim.operations.integrator = hoomd.md.Integrator(
                dt=0.005, methods=[method], forces=forces, integrate_rotational_dof=True)
            sim.run(0)
            force = forces[0]
            results[name + "/forces"] = np.array(force.forces)
            results[name + "/energies"] = np.array(force.energies)
            results[name + "/torques"] = np.array(force.torques)
            results[name + "/virials"] = np.array(force.virials)
        except Exception as e:  # record the failure and continue with the other forces
            results[name + "/error"] = np.array(repr(e))
    return results


def compare(check):
    import hoomd

    for name in cases(hoomd):
        for quantity in ("forces", "torques", "energies", "virials"):
            check.close(f"{name}/{quantity}", TOLERANCE)
