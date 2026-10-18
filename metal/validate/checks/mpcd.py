"""Multiparticle collision dynamics: streaming, collisions, walls, fillers, sorting, and a
bead-spring chain embedded in an SRD solvent.

Short runs must match closely. Poiseuille flow, reverse nonequilibrium shear flow, and the
embedded chain must agree statistically.
"""

import numpy as np

from .common import device

REQUIRES = "hoomd.mpcd"


def solvent(hoomd, rng, L, H=None, density=5, kT=1.0, n_md=0):
    snap = hoomd.Snapshot()
    snap.configuration.box = [L, L, L, 0, 0, 0]
    snap.particles.N = n_md
    snap.particles.types = ["A"]
    n = int(density * L**3) if H is None else int(density * L * L * 2 * H)
    snap.mpcd.N = n
    snap.mpcd.types = ["S"]
    pos = rng.uniform(-L / 2, L / 2, size=(n, 3))
    if H is not None:
        pos[:, 1] = rng.uniform(-H, H, size=n)
    snap.mpcd.position[:] = pos
    vel = rng.normal(scale=np.sqrt(kT), size=(n, 3))
    snap.mpcd.velocity[:] = vel - vel.mean(axis=0)
    snap.mpcd.mass = 1.0
    return snap


def run(device_name):
    import hoomd
    import hoomd.mpcd

    mpcd = hoomd.mpcd
    rng = np.random.default_rng(8)
    dev = device(device_name)
    results = {}

    def simulate(snap, steps, integrator, updaters=()):
        sim = hoomd.Simulation(device=dev, seed=17)
        sim.create_state_from_snapshot(snap)
        sim.operations.integrator = integrator
        for u in updaters:
            sim.operations.updaters.append(u)
        sim.run(steps)
        return sim

    def store(name, sim, md=False):
        s = sim.state.get_snapshot()
        results[name + "/solvent_positions"] = s.mpcd.position
        results[name + "/solvent_velocities"] = s.mpcd.velocity
        if md:
            results[name + "/positions"] = s.particles.position
            results[name + "/velocities"] = s.particles.velocity

    def section(name, body):
        try:
            body()
        except Exception as e:
            results[name + "/error"] = np.array(repr(e))

    def stream_plates():
        geometry = mpcd.geometry.ParallelPlates(separation=8.0, no_slip=True)
        stream = mpcd.stream.BounceBack(
            period=1, geometry=geometry,
            mpcd_particle_force=mpcd.force.ConstantForce((0.5, 0, 0)))
        sim = simulate(solvent(hoomd, rng, 10.0, H=4.0), 40,
                       mpcd.Integrator(dt=0.05, streaming_method=stream))
        store("stream_plates", sim)

    def stream_bulk():
        stream = mpcd.stream.Bulk(period=1,
                                  mpcd_particle_force=mpcd.force.SineForce(0.3, 2 * np.pi / 8))
        sim = simulate(solvent(hoomd, rng, 8.0), 40,
                       mpcd.Integrator(dt=0.05, streaming_method=stream))
        store("stream_bulk", sim)

    def andersen():
        integrator = mpcd.Integrator(
            dt=0.1, streaming_method=mpcd.stream.Bulk(period=1),
            collision_method=mpcd.collide.AndersenThermostat(period=1, kT=1.0))
        store("andersen_3steps", simulate(solvent(hoomd, rng, 8.0), 3, integrator))

    def srd_short():
        for name, kT in (("srd_3steps", None), ("srd_thermostat_3steps", 1.0)):
            integrator = mpcd.Integrator(
                dt=0.1, streaming_method=mpcd.stream.Bulk(period=1),
                collision_method=mpcd.collide.StochasticRotationDynamics(period=1, angle=130,
                                                                         kT=kT))
            store(name, simulate(solvent(hoomd, rng, 8.0), 3, integrator))

    def walls_short():
        geometry = mpcd.geometry.ParallelPlates(separation=8.0, speed=0.5, no_slip=True)
        integrator = mpcd.Integrator(
            dt=0.1, streaming_method=mpcd.stream.BounceBack(period=1, geometry=geometry),
            collision_method=mpcd.collide.StochasticRotationDynamics(period=1, angle=130),
            virtual_particle_fillers=[mpcd.fill.GeometryFiller(type="S", density=5.0, kT=1.0,
                                                               geometry=geometry)])
        store("walls_filler_srd_3steps", simulate(solvent(hoomd, rng, 10.0, H=4.0), 3,
                                                  integrator))

    def poiseuille():
        L, H = 12.0, 5.0
        geometry = mpcd.geometry.ParallelPlates(separation=2 * H, no_slip=True)
        stream = mpcd.stream.BounceBack(
            period=1, geometry=geometry,
            mpcd_particle_force=mpcd.force.ConstantForce((0.05, 0, 0)))
        integrator = mpcd.Integrator(
            dt=0.1, streaming_method=stream,
            collision_method=mpcd.collide.StochasticRotationDynamics(period=1, angle=130,
                                                                     kT=1.0),
            virtual_particle_fillers=[mpcd.fill.GeometryFiller(type="S", density=5.0, kT=1.0,
                                                               geometry=geometry)],
            mpcd_particle_sorter=mpcd.tune.ParticleSorter(trigger=20))
        sim = simulate(solvent(hoomd, rng, L, H=H), 1000, integrator)
        bins = np.linspace(-H, H, 11)
        profiles, kTs = [], []
        for _ in range(100):
            sim.run(20)
            s = sim.state.get_snapshot()
            y, v = s.mpcd.position[:, 1], s.mpcd.velocity
            profile = np.array([v[(y >= bins[i]) & (y < bins[i + 1]), 0].mean()
                                for i in range(10)])
            dv = v.copy()
            dv[:, 0] -= profile[np.clip(np.digitize(y, bins) - 1, 0, 9)]
            profiles.append(profile)
            kTs.append(np.mean(dv**2))
        results["poiseuille/vx_profile"] = np.array(profiles)
        results["poiseuille/kT"] = np.array(kTs)

    def shear_flow():
        integrator = mpcd.Integrator(
            dt=0.1, streaming_method=mpcd.stream.Bulk(period=1),
            collision_method=mpcd.collide.StochasticRotationDynamics(period=1, angle=130))
        rnes = mpcd.update.ReverseNonequilibriumShearFlow(trigger=1, num_swaps=10,
                                                          slab_width=1.0)
        sim = simulate(solvent(hoomd, rng, 10.0), 500, integrator, updaters=[rnes])
        s = sim.state.get_snapshot()
        results["shear_flow/exchanged_momentum"] = np.array([rnes.summed_exchanged_momentum])
        results["shear_flow/kinetic_energy"] = np.array([0.5 * np.sum(s.mpcd.velocity**2)])
        bins = np.linspace(-5, 5, 11)
        profiles = []
        for _ in range(10):
            sim.run(20)
            s = sim.state.get_snapshot()
            y, vx = s.mpcd.position[:, 1], s.mpcd.velocity[:, 0]
            profiles.append([vx[(y >= bins[i]) & (y < bins[i + 1])].mean() for i in range(10)])
        results["shear_flow/vx_profile"] = np.mean(profiles, axis=0)

    def chain():
        L, n_beads = 10.0, 20
        snap = solvent(hoomd, rng, L, n_md=n_beads)
        positions = np.zeros((n_beads, 3))
        for i in range(n_beads):
            row, col = divmod(i, 5)
            col = col if row % 2 == 0 else 4 - col
            positions[i] = (col * 0.97 - 2.0, row * 0.97 - 1.5, 0.0)
        snap.particles.position[:] = positions
        snap.particles.mass[:] = 5.0
        snap.bonds.N = n_beads - 1
        snap.bonds.types = ["b"]
        snap.bonds.group[:] = np.stack([np.arange(n_beads - 1), np.arange(1, n_beads)], axis=1)
        nl = hoomd.md.nlist.Cell(buffer=0.4, exclusions=["bond"])
        wca = hoomd.md.pair.LJ(nlist=nl, default_r_cut=2 ** (1 / 6), mode="shift")
        wca.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
        fene = hoomd.md.bond.FENEWCA()
        fene.params["b"] = dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)
        srd = mpcd.collide.StochasticRotationDynamics(period=20, angle=130, kT=1.0,
                                                      embedded_particles=hoomd.filter.All())
        integrator = mpcd.Integrator(
            dt=0.005, streaming_method=mpcd.stream.Bulk(period=20), collision_method=srd,
            forces=[wca, fene],
            methods=[hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())])
        sim = simulate(snap, 60, integrator)
        store("chain_60steps", sim, md=True)
        sim.run(3940)
        rg2, kT = [], []
        for _ in range(200):
            sim.run(200)
            s = sim.state.get_snapshot()
            p = s.particles.position + s.particles.image * L
            rg2.append(np.mean(np.sum((p - p.mean(axis=0)) ** 2, axis=1)))
            kT.append(np.mean(s.particles.velocity**2) * 5.0)
        results["chain/Rg2"] = np.array(rg2)
        results["chain/kT"] = np.array(kT)

    for name, body in (("stream_plates", stream_plates), ("stream_bulk", stream_bulk),
                       ("andersen_3steps", andersen), ("srd_3steps", srd_short),
                       ("walls_filler_srd_3steps", walls_short), ("poiseuille", poiseuille),
                       ("shear_flow", shear_flow), ("chain_60steps", chain)):
        section(name, body)
    return results


def compare(check):
    # deterministic short runs (identical random streams on both devices)
    for name in ("stream_plates", "stream_bulk", "andersen_3steps", "srd_3steps",
                 "srd_thermostat_3steps", "walls_filler_srd_3steps"):
        check.close(name + "/solvent_positions", 1e-4)
        check.close(name + "/solvent_velocities", 1e-4)
    for quantity in ("positions", "velocities", "solvent_positions", "solvent_velocities"):
        check.close("chain_60steps/" + quantity, 1e-4)
    # statistics (samples are correlated, so estimate errors from block averages)
    check.mean("poiseuille/vx_profile", blocks=10, nsigma=5, rel_floor=0.02)
    check.mean("poiseuille/kT", blocks=10, nsigma=5, rel_floor=0.01)
    check.value("shear_flow/kinetic_energy", rtol=1e-3)
    check.value("shear_flow/exchanged_momentum", rtol=0.05)
    check.close("shear_flow/vx_profile", 0.25)
    check.mean("chain/kT", blocks=10, nsigma=5, rel_floor=0.02)
    # a single 20-bead chain relaxes slowly, so its size is only a coarse check
    check.mean("chain/Rg2", blocks=5, nsigma=5, rel_floor=0.25)
