# Copyright (c) 2026 Aldo Vasquez-Briceno.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

"""Comparison helpers shared by the validation checks."""

import numpy as np


class Checker:
    """Collect pass/fail results that compare CPU and GPU outputs.

    Every comparison records ``(name, passed, detail)`` in ``results``. Keys that are missing
    from either result set, or that hold an error message, fail the comparison.
    """

    def __init__(self, cpu, gpu):
        self.cpu = cpu
        self.gpu = gpu
        self.results = []

    def _record(self, name, passed, detail):
        self.results.append((name, bool(passed), detail))
        return passed

    def _get(self, key):
        """Return the CPU and GPU arrays for key, or None after recording a failure."""
        parts = key.split("/")
        errors = ["/".join(parts[:i]) + "/error" for i in range(1, len(parts))]
        for device, data in (("cpu", self.cpu), ("gpu", self.gpu)):
            for error in errors:
                if error in data:
                    self._record(key, False, f"{device} raised: {data[error]}")
                    return None
            if key not in data:
                self._record(key, False, f"missing from the {device} results")
                return None
        return (np.asarray(self.cpu[key], dtype=np.float64),
                np.asarray(self.gpu[key], dtype=np.float64))

    def close(self, key, rtol, scale=None):
        """Pass when max |cpu - gpu| <= rtol * scale (default scale: max |cpu|).

        Non-finite values must appear at the same places with the same values.
        """
        pair = self._get(key)
        if pair is None:
            return False
        a, b = pair
        if a.shape != b.shape:
            return self._record(key, False, f"shape {a.shape} (cpu) vs {b.shape} (gpu)")
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        mismatch = np.sum(finite_a != finite_b) + np.sum(
            (~finite_a & ~finite_b) & ~((a == b) | (np.isnan(a) & np.isnan(b))))
        if mismatch:
            return self._record(key, False, f"{mismatch} non-finite values differ")
        ok = finite_a & finite_b
        if not ok.any():
            return self._record(key, True, "no finite values")
        if scale is None:
            scale = np.max(np.abs(a[ok]))
        scale = max(scale, 1e-30)
        rel = np.max(np.abs(a[ok] - b[ok])) / scale
        return self._record(key, rel <= rtol, f"max diff / scale {rel:.1e} (tol {rtol:g})")

    def value(self, key, rtol=0.0, atol=0.0):
        """Pass when |cpu - gpu| <= atol + rtol * |cpu| element-wise."""
        pair = self._get(key)
        if pair is None:
            return False
        a, b = pair
        bad = np.abs(a - b) > atol + rtol * np.abs(a)
        detail = (f"cpu {np.array2string(a, precision=4)} gpu {np.array2string(b, precision=4)}"
                  if a.size <= 4 else f"{np.sum(bad)} of {a.size} values outside tolerance")
        return self._record(key, not np.any(bad) and np.all(np.isfinite(b)), detail)

    def mean(self, key, column=None, start=0, nsigma=5.0, blocks=None, rel_floor=0.0, label=None):
        """Compare the means of two sampled series within their statistical error.

        Args:
            key: Series stored as ``(samples,)`` or ``(samples, columns)``.
            column: Column to compare (all columns when None).
            start: First sample to use (skip equilibration).
            nsigma: Allowed difference in units of the combined standard error.
            blocks: Average consecutive samples into this many blocks before estimating the
                error (use for correlated samples).
            rel_floor: Always allow this relative difference.
            label: Name to report (default: key and column).
        """
        pair = self._get(key)
        if pair is None:
            return False
        a, b = pair
        a, b = a[start:], b[start:]
        if column is not None:
            a, b = a[:, column], b[:, column]
        if blocks:
            n = len(a) // blocks * blocks
            a = a[:n].reshape(blocks, -1, *a.shape[1:]).mean(axis=1)
            b = b[:n].reshape(blocks, -1, *b.shape[1:]).mean(axis=1)
        mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
        error = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
        allowed = nsigma * error + rel_floor * np.abs(mean_a)
        diff = np.abs(mean_a - mean_b)
        passed = np.all(diff <= allowed + 1e-12) and np.all(np.isfinite(mean_b))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.max(np.where(error > 0, diff / error, 0.0))
        name = label or (key if column is None else f"{key}[{column}]")
        detail = (f"cpu {np.array2string(np.atleast_1d(mean_a)[:3], precision=4)} "
                  f"gpu {np.array2string(np.atleast_1d(mean_b)[:3], precision=4)} "
                  f"(max {z:.1f} sigma, limit {nsigma:g})")
        return self._record(name, passed, detail)

    def below(self, name, value, limit):
        """Pass when value <= limit."""
        value = float(value)
        return self._record(name, np.isfinite(value) and value <= limit,
                            f"{value:.2e} (limit {limit:.0e})")


def device(name):
    """Create the hoomd device for "cpu" or "gpu"."""
    import hoomd

    return hoomd.device.GPU() if name == "gpu" else hoomd.device.CPU()


def lattice(n_side, L, jitter=0.0, rng=None):
    """Simple cubic lattice of n_side**3 points filling a cubic box of side L."""
    g = (np.arange(n_side) + 0.5) * (L / n_side) - L / 2
    points = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
    if jitter:
        points = points + rng.uniform(-jitter, jitter, size=points.shape)
    return points
