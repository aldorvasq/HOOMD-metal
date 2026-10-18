# Copyright (c) 2026 Aldo Vasquez-Briceno.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

"""Pair, bond, angle, dihedral, improper, special pair, external, and wall forces.

Evaluates every force on one configuration of 300 ten-bead chains in a triclinic box and compares
per-particle forces, energies, and virials.
"""

import numpy as np

from .common import device

N = 3000
L = 16.0

# Single precision rounding is amplified near the singular geometries of these forms.
TOLERANCE = {"DLVO": 1e-2, "improper_harmonic": 5e-3, "dihedral_table": 1e-3}
DEFAULT_TOLERANCE = 1e-4


def make_snapshot(hoomd):
    rng = np.random.default_rng(2024)
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = ["A", "B"]
    snap.configuration.box = [L, L, L, 0.1, 0.0, 0.05]
    # self-avoiding random walk chains of 10 (well-conditioned bond, angle, and dihedral geometry)
    pos = np.zeros((N, 3))
    for i in range(N):
        for _ in range(1000):
            if i % 10 == 0:
                trial = rng.uniform(-L / 2, L / 2, 3)
            else:
                v = rng.normal(size=3)
                v /= np.linalg.norm(v)
                if i % 10 >= 2:
                    prev = pos[i - 1] - pos[i - 2]
                    prev /= np.linalg.norm(prev)
                    if np.dot(v, prev) < -0.3:
                        continue
                trial = pos[i - 1] + 0.97 * v
            d = pos[:i] - trial
            d -= L * np.round(d / L)
            if i == 0 or np.min(np.sum(d * d, axis=1)) > 0.85**2:
                break
        pos[i] = trial
    snap.particles.position[:] = pos
    snap.particles.velocity[:] = rng.normal(size=(N, 3))
    snap.particles.typeid[:] = rng.integers(0, 2, N)
    snap.particles.charge[:] = rng.choice([-1.0, 1.0], N)
    snap.particles.diameter[:] = rng.uniform(0.8, 1.2, N)
    chains = np.arange(N).reshape(-1, 10)
    bonds = np.concatenate([chains[:, i : i + 2] for i in range(9)])
    angles = np.concatenate([chains[:, i : i + 3] for i in range(8)])
    dihedrals = np.concatenate([chains[:, i : i + 4] for i in range(7)])
    snap.bonds.N = len(bonds)
    snap.bonds.types = ["b"]
    snap.bonds.group[:] = bonds
    snap.angles.N = len(angles)
    snap.angles.types = ["a"]
    snap.angles.group[:] = angles
    snap.dihedrals.N = len(dihedrals)
    snap.dihedrals.types = ["d"]
    snap.dihedrals.group[:] = dihedrals
    snap.impropers.N = len(dihedrals)
    snap.impropers.types = ["i"]
    snap.impropers.group[:] = dihedrals
    snap.pairs.N = len(chains) * 2
    snap.pairs.types = ["p"]
    snap.pairs.group[:] = np.concatenate([chains[:, [0, 5]], chains[:, [2, 9]]])
    snap.wrap()
    return snap


def cases(hoomd):
    md = hoomd.md
    result = {}

    def nlist():
        return md.nlist.Cell(buffer=0.3, exclusions=())

    def pair(name, cls, params, r_cut=2.5, mode="shift", **kwargs):
        def build():
            if mode is not None:
                kwargs["mode"] = mode
            p = cls(nlist=nlist(), default_r_cut=r_cut, **kwargs)
            for types in [("A", "A"), ("A", "B"), ("B", "B")]:
                p.params[types] = params(types)
            return p

        result[name] = build

    eps = {("A", "A"): 1.0, ("A", "B"): 0.8, ("B", "B"): 0.6}
    pair("LJ", md.pair.LJ, lambda p: dict(epsilon=eps[p], sigma=0.9))
    pair("LJ_xplor", md.pair.LJ, lambda p: dict(epsilon=eps[p], sigma=0.9), mode="xplor",
         default_r_on=2.0)
    pair("Gaussian", md.pair.Gaussian, lambda p: dict(epsilon=eps[p], sigma=0.7))
    pair("ExpandedLJ", md.pair.ExpandedLJ, lambda p: dict(epsilon=eps[p], sigma=0.8, delta=0.1))
    pair("Yukawa", md.pair.Yukawa, lambda p: dict(epsilon=eps[p], kappa=1.2))
    pair("Ewald", md.pair.Ewald, lambda p: dict(kappa=1.0, alpha=0.0), mode=None)
    pair("Morse", md.pair.Morse, lambda p: dict(D0=eps[p], alpha=3.0, r0=1.0))
    pair("DPDConservative", md.pair.DPDConservative, lambda p: dict(A=25.0 * eps[p]),
         r_cut=1.0, mode=None)
    pair("ForceShiftedLJ", md.pair.ForceShiftedLJ, lambda p: dict(epsilon=eps[p], sigma=0.9),
         mode=None)
    pair("Mie", md.pair.Mie, lambda p: dict(epsilon=eps[p], sigma=0.9, n=12, m=6))
    pair("ExpandedMie", md.pair.ExpandedMie,
         lambda p: dict(epsilon=eps[p], sigma=0.8, n=10, m=5, delta=0.1))
    pair("ReactionField", md.pair.ReactionField,
         lambda p: dict(epsilon=1.0, eps_rf=10.0, use_charge=True))
    pair("DLVO", md.pair.DLVO, lambda p: dict(A=1.0, kappa=1.0, Z=0.5, a1=0.5, a2=0.5),
         mode="none")
    pair("Buckingham", md.pair.Buckingham, lambda p: dict(A=100.0, rho=0.3, C=1.0))
    pair("LJ1208", md.pair.LJ1208, lambda p: dict(epsilon=eps[p], sigma=0.9))
    pair("LJ0804", md.pair.LJ0804, lambda p: dict(epsilon=eps[p], sigma=0.9))
    pair("Fourier", md.pair.Fourier, lambda p: dict(a=[0.5, 1.0, 1.5], b=[0.25, 0.034, 0.7]),
         mode="none")
    pair("OPP", md.pair.OPP,
         lambda p: dict(C1=1.0, C2=1.0, eta1=15, eta2=3, k=1.0, phi=np.pi / 4))
    pair("TWF", md.pair.TWF, lambda p: dict(epsilon=eps[p], sigma=1.0, alpha=50.0))
    pair("LJGauss", md.pair.LJGauss, lambda p: dict(epsilon=eps[p], sigma=0.1, r0=1.5))
    pair("Moliere", md.pair.Moliere, lambda p: dict(qi=1.0, qj=1.0, aF=0.5), mode="none")
    pair("ZBL", md.pair.ZBL, lambda p: dict(qi=1.0, qj=1.0, aF=0.5), mode=None)
    pair("ExpandedGaussian", md.pair.ExpandedGaussian,
         lambda p: dict(epsilon=eps[p], sigma=0.7, delta=0.2))
    pair("WangFrenkel", md.pair.WangFrenkel,
         lambda p: dict(epsilon=eps[p], sigma=0.9, R=1.8, mu=1, nu=1), r_cut=1.8, mode="none")
    pair("Zetterling", md.pair.Zetterling,
         lambda p: dict(A=1.0, alpha=1.0, kf=2.0, B=1.0, n=12, sigma=1.0), mode="none")

    def table():
        r = np.linspace(0.5, 2.5, 201)
        p = md.pair.Table(nlist=nlist(), default_r_cut=2.5)
        for types in [("A", "A"), ("A", "B"), ("B", "B")]:
            p.params[types] = dict(r_min=0.5, U=4 * (r**-12 - r**-6),
                                   F=4 * (12 * r**-13 - 6 * r**-7))
        return p

    result["Table"] = table

    def dpd():
        p = md.pair.DPD(nlist=nlist(), kT=1.0, default_r_cut=1.0)
        for types in [("A", "A"), ("A", "B"), ("B", "B")]:
            p.params[types] = dict(A=25.0, gamma=4.5)
        return p

    result["DPD"] = dpd

    def bonded(name, cls, type_name, params, **kwargs):
        def build():
            force = cls(**kwargs)
            force.params[type_name] = params
            return force

        result[name] = build

    bonded("bond_harmonic", md.bond.Harmonic, "b", dict(k=50.0, r0=1.4))
    bonded("bond_fenewca", md.bond.FENEWCA, "b",
           dict(k=30.0, r0=3.0, epsilon=1.0, sigma=1.0, delta=0.0))
    bonded("bond_tether", md.bond.Tether, "b",
           dict(k_b=10.0, l_min=0.5, l_c1=1.0, l_c0=1.5, l_max=3.0))
    bonded("angle_harmonic", md.angle.Harmonic, "a", dict(k=3.0, t0=2.0))
    bonded("angle_cosinesq", md.angle.CosineSquared, "a", dict(k=3.0, t0=2.0))
    theta = np.linspace(0, np.pi, 101)
    bonded("angle_table", md.angle.Table, "a",
           dict(U=2 * (theta - 2.0) ** 2, tau=-4 * (theta - 2.0)), width=101)
    bonded("dihedral_periodic", md.dihedral.Periodic, "d", dict(k=2.0, d=1, n=3, phi0=0.5))
    bonded("dihedral_opls", md.dihedral.OPLS, "d", dict(k1=1.0, k2=0.5, k3=0.25, k4=0.1))
    phi = np.linspace(-np.pi, np.pi, 101)
    bonded("dihedral_table", md.dihedral.Table, "d",
           dict(U=np.cos(3 * phi), tau=3 * np.sin(3 * phi)), width=101)
    bonded("improper_harmonic", md.improper.Harmonic, "i", dict(k=2.0, chi0=0.3))
    bonded("improper_periodic", md.improper.Periodic, "i", dict(k=2.0, d=1, n=2, chi0=0.3))

    def special(cls, params):
        def build():
            force = cls()
            force.params["p"] = params
            force.r_cut["p"] = 5.0
            return force

        return build

    result["special_lj"] = special(md.special_pair.LJ, dict(epsilon=1.0, sigma=1.0))
    result["special_coulomb"] = special(md.special_pair.Coulomb, dict(alpha=0.5))

    def periodic():
        force = md.external.field.Periodic()
        force.params["A"] = dict(A=1.0, i=0, w=0.1, p=3)
        force.params["B"] = dict(A=0.5, i=1, w=0.2, p=2)
        return force

    def electric():
        force = md.external.field.Electric()
        force.E["A"] = (1.0, 0.5, 0.2)
        force.E["B"] = (0.0, -1.0, 0.3)
        return force

    def walls():
        w = [hoomd.wall.Sphere(radius=7.0, origin=(0, 0, 0), inside=True),
             hoomd.wall.Plane(origin=(0, 0, -7.5), normal=(0, 0, 1)),
             hoomd.wall.Cylinder(radius=7.5, axis=(1, 0, 0), inside=True)]
        force = md.external.wall.LJ(walls=w)
        force.params[["A", "B"]] = dict(epsilon=1.0, sigma=1.0, r_cut=2.5)
        return force

    result["external_periodic"] = periodic
    result["external_electric"] = electric
    result["wall_lj"] = walls
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
            force = build()
            method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.Null())
            sim.operations.integrator = hoomd.md.Integrator(dt=0.001, methods=[method],
                                                            forces=[force])
            sim.run(0)
            results[name + "/forces"] = np.array(force.forces)
            results[name + "/energies"] = np.array(force.energies)
            results[name + "/virials"] = np.array(force.virials)
        except Exception as e:  # record the failure and continue with the other forces
            results[name + "/error"] = np.array(repr(e))
    return results


def compare(check):
    import hoomd

    for name in cases(hoomd):
        tol = TOLERANCE.get(name, DEFAULT_TOLERANCE)
        for quantity in ("forces", "energies", "virials"):
            check.close(f"{name}/{quantity}", tol)
