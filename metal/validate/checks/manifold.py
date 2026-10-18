"""RATTLE integrators (NVE, Langevin, Brownian, active Brownian) on seven manifolds.

Compares positions, velocities, and orientations after 100 steps, and checks that the particles
stay on the manifold.
"""

import numpy as np

from .common import device

L = 20.0
N = 150
MANIFOLDS = ("Sphere", "Ellipsoid", "Cylinder", "Plane", "Primitive", "Gyroid", "Diamond")
METHODS = ("NVE", "Langevin", "Brownian", "Active")


def implicit(name):
    """Implicit function F(x) = 0 of each test manifold (matching hoomd.md.manifold)."""
    k = 2 * np.pi / L
    functions = {
        "Sphere": lambda x: np.sum(x**2, axis=1) - 6.0**2,
        "Ellipsoid": lambda x: (x[:, 0] / 7) ** 2 + (x[:, 1] / 5) ** 2 + (x[:, 2] / 6) ** 2 - 1,
        "Cylinder": lambda x: x[:, 0] ** 2 + x[:, 1] ** 2 - 6.0**2,
        "Plane": lambda x: x[:, 2] - 0.5,
        "Primitive": lambda x: np.sum(np.cos(k * x), axis=1),
        "Gyroid": lambda x: (np.sin(k * x[:, 0]) * np.cos(k * x[:, 1])
                             + np.sin(k * x[:, 1]) * np.cos(k * x[:, 2])
                             + np.sin(k * x[:, 2]) * np.cos(k * x[:, 0])),
        "Diamond": lambda x: (np.prod(np.cos(np.pi / L * x), axis=1)
                              - np.prod(np.sin(np.pi / L * x), axis=1)),
    }
    return functions[name]


def manifold(hoomd, name):
    m = hoomd.md.manifold
    return {
        "Sphere": lambda: m.Sphere(r=6.0),
        "Ellipsoid": lambda: m.Ellipsoid(a=7.0, b=5.0, c=6.0),
        "Cylinder": lambda: m.Cylinder(r=6.0),
        "Plane": lambda: m.Plane(shift=0.5),
        "Primitive": lambda: m.Primitive(N=1),
        "Gyroid": lambda: m.Gyroid(N=1),
        "Diamond": lambda: m.Diamond(N=1),
    }[name]()


def points_on(name, rng):
    """N points on the manifold, at least 1.1 apart."""
    F = implicit(name)
    points = []
    while len(points) < N:
        x = rng.uniform(-L / 2 + 0.5, L / 2 - 0.5, size=(4000, 3))
        for _ in range(50):  # Newton projection onto F = 0
            h = 1e-6
            grad = np.stack([(F(x + h * e) - F(x - h * e)) / (2 * h) for e in np.eye(3)], axis=1)
            x = x - (F(x) / np.sum(grad**2, axis=1))[:, None] * grad
        x = x[np.all(np.abs(x) < L / 2 - 0.1, axis=1) & (np.abs(F(x)) < 1e-6)]
        for p in x:
            if len(points) == N:
                break
            if points:
                d = np.array(points) - p
                d -= L * np.round(d / L)
                if np.min(np.sum(d * d, axis=1)) <= 1.1**2:
                    continue
            points.append(p)
    return np.array(points)


def simulate(hoomd, dev, snap, name, method_name):
    sim = hoomd.Simulation(device=dev, seed=21)
    sim.create_state_from_snapshot(snap)
    wca = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=2 ** (1 / 6),
                           mode="shift")
    wca.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
    constraint = manifold(hoomd, name)
    forces = [wca]
    rattle = hoomd.md.methods.rattle
    everything = hoomd.filter.All()
    if method_name == "NVE":
        sim.state.thermalize_particle_momenta(everything, kT=1.0)
        method = rattle.NVE(filter=everything, manifold_constraint=constraint, tolerance=1e-6)
        dt = 0.002
    elif method_name == "Langevin":
        method = rattle.Langevin(filter=everything, kT=1.0, manifold_constraint=constraint,
                                 tolerance=1e-6)
        dt = 0.002
    else:
        method = rattle.Brownian(filter=everything, kT=1.0, manifold_constraint=constraint,
                                 tolerance=1e-6)
        dt = 0.0005
        if method_name == "Active":
            active = hoomd.md.force.ActiveOnManifold(filter=everything,
                                                     manifold_constraint=constraint)
            active.active_force["A"] = (5.0, 0, 0)
            forces.append(active)
            sim.operations.updaters.append(
                active.create_diffusion_updater(trigger=1, rotational_diffusion=1.0))
    sim.operations.integrator = hoomd.md.Integrator(dt=dt, methods=[method], forces=forces)
    sim.run(100)
    return sim.state.get_snapshot()


def run(device_name):
    import hoomd

    rng = np.random.default_rng(4)
    dev = device(device_name)
    results = {}
    for name in MANIFOLDS:
        snap = hoomd.Snapshot()
        snap.particles.N = N
        snap.particles.types = ["A"]
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.particles.position[:] = points_on(name, rng)
        snap.particles.moment_inertia[:] = [1, 1, 1]
        q = rng.normal(size=(N, 4))
        snap.particles.orientation[:] = q / np.linalg.norm(q, axis=1)[:, None]
        for method_name in METHODS:
            key = f"{name}/{method_name}"
            try:
                s = simulate(hoomd, dev, snap, name, method_name)
                results[key + "/positions"] = s.particles.position
                results[key + "/velocities"] = s.particles.velocity
                results[key + "/orientations"] = s.particles.orientation
                results[key + "/residual"] = implicit(name)(
                    s.particles.position.astype(np.float64))
            except Exception as e:
                results[key + "/error"] = np.array(repr(e))
    return results


def compare(check):
    for name in MANIFOLDS:
        for method_name in METHODS:
            key = f"{name}/{method_name}"
            check.close(key + "/positions", 1e-4)
            check.close(key + "/velocities", 1e-3)
            check.close(key + "/orientations", 1e-4)
            residual = check.gpu.get(key + "/residual")
            if residual is not None:
                check.below(key + "/gpu max |F(x)| (on the manifold)",
                            np.max(np.abs(residual)), 1e-4)
