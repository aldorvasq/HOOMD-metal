"""Reverse perturbation (Mueller-Plathe) shear flow in an LJ liquid.

Compares the exchanged momentum and the resulting velocity profile of 8000 particles over 700
steps.
"""

import numpy as np

from .common import device, lattice

L = 21.0


def run(device_name):
    import hoomd

    rng = np.random.default_rng(1)
    N = 8000
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = ["A"]
    snap.configuration.box = [L, L, L, 0, 0, 0]
    snap.particles.position[:] = lattice(20, L)
    snap.particles.velocity[:] = rng.normal(size=(N, 3))
    sim = hoomd.Simulation(device=device(device_name), seed=2)
    sim.create_state_from_snapshot(snap)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=2.5)
    lj.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
    method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005, methods=[method], forces=[lj])
    flow = hoomd.md.update.ReversePerturbationFlow(
        filter=hoomd.filter.All(), flow_target=hoomd.variant.Ramp(0, 30, 0, 2000),
        slab_direction="Z", flow_direction="X", n_slabs=20)
    flow.trigger = hoomd.trigger.Periodic(10)
    sim.operations.updaters.append(flow)
    exchanged = []
    for _ in range(10):
        sim.run(50)
        exchanged.append(flow.summed_exchanged_momentum)
    bins = np.linspace(-L / 2, L / 2, 21)
    profiles = []
    for _ in range(10):
        sim.run(20)
        s = sim.state.get_snapshot()
        z, vx = s.particles.position[:, 2], s.particles.velocity[:, 0]
        profiles.append([vx[(z >= bins[i]) & (z < bins[i + 1])].mean() for i in range(20)])
    return {"exchanged_momentum": np.array(exchanged), "vx_profile": np.mean(profiles, axis=0)}


def compare(check):
    # The swaps pick the fastest particles, so trajectories diverge, but the updater drives both
    # runs toward the same target momentum flux.
    check.value("exchanged_momentum", rtol=0.05, atol=0.5)
    check.close("vx_profile", 0.25)
