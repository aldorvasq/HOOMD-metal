# Copyright (c) 2026 Aldo Vásquez-Briceño.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

"""Check that HOOMD-metal gives the same results on the GPU as on the CPU of this Mac.

Usage::

    python metal/validate/validate.py                 # run every check (2-3 minutes on an M4)
    python metal/validate/validate.py polymer forces  # run selected checks
    python metal/validate/validate.py --list          # list the checks
    python metal/validate/validate.py --pytest        # also compare HOOMD's test suite results

Each check runs once with ``hoomd.device.CPU()`` and once with ``hoomd.device.GPU()``, in
separate processes at the same time, then compares the results. The tolerances allow for single
precision rounding and, in long runs, for statistical error. The exit status is 0 when every
comparison passes.
"""

import argparse
import importlib
import importlib.util
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ElementTree

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CHECKS = ("forces", "aniso", "nlist", "pppm", "polymer", "lj", "rigid", "manifold", "flow",
          "mpcd")

# pytest cases that may fail on the GPU while passing on the CPU, and why that is not a GPU error.
_RANDOM_ATTRIBUTES = ("draws random attribute values from one shared generator, so the CPU and "
                      "GPU variants test different values; both devices fail when a value does "
                      "not survive the single precision round trip")
_RIGID_MPCD = ("conserves momentum to about 2e-5 in single precision on both devices; the "
               "GPU summation order lands just outside the test tolerance")
PYTEST_KNOWN_GPU_ONLY = {
    "md.pytest.test_meshpotential::test_forces_and_energies": (
        "the Helfrich energy cancels large terms; single precision rounding differs between the "
        "CPU and GPU summation order"),
    "md.pytest.test_methods.TestMethods::test_attributes_attached": _RANDOM_ATTRIBUTES,
    "md.pytest.test_methods.TestThermostats::test_attributes_attached": _RANDOM_ATTRIBUTES,
    "mpcd.pytest.test_collide.TestCollisionMethod::test_rigid_collide": _RIGID_MPCD,
    "mpcd.pytest.test_collide.TestCollisionMethod::test_rigid_collide_free": _RIGID_MPCD,
}


def load_check(name):
    sys.path.insert(0, str(HERE))
    return importlib.import_module(f"checks.{name}")


def worker(name, device_name, output):
    """Run one check on one device and save its results (runs in a subprocess)."""
    results = load_check(name).run(device_name)
    np.savez(output, **results)


def describe_system():
    import hoomd

    lines = [f"hoomd {hoomd.version.version} ({hoomd.version.gpu_platform or 'no GPU'}) "
             f"from {pathlib.Path(hoomd.__file__).parent}"]
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                          text=True).stdout.strip()
    memory = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                text=True).stdout or 0)
    lines.append(f"{chip}, {memory / 2**30:.0f} GB, macOS {platform.mac_ver()[0]}")
    available = hoomd.device.GPU.get_available_devices()
    if not available:
        reasons = "; ".join(hoomd.device.GPU.get_unavailable_device_reasons())
        return lines, f"No usable GPU: {reasons}"
    lines.append(f"GPU: {available[0].strip()}")
    return lines, None


def run_check(name, workdir, verbose):
    module = load_check(name)
    requires = getattr(module, "REQUIRES", None)
    if requires and importlib.util.find_spec(requires) is None:
        print(f"{name:10s} skipped ({requires} is not built)")
        return 0, 0

    start = time.time()
    processes = {}
    for device_name in ("cpu", "gpu"):
        log = open(workdir / f"{name}_{device_name}.log", "w")
        processes[device_name] = (subprocess.Popen(
            [sys.executable, str(HERE / "validate.py"), "--worker", name, device_name,
             str(workdir / f"{name}_{device_name}.npz")],
            cwd=workdir, stdout=log, stderr=subprocess.STDOUT), log, time.time())
    elapsed = {}
    while len(elapsed) < len(processes):
        for device_name, (process, log, began) in processes.items():
            if device_name not in elapsed and process.poll() is not None:
                elapsed[device_name] = time.time() - began
                log.close()
        time.sleep(0.1)

    timing = f"cpu {elapsed['cpu']:.0f} s, gpu {elapsed['gpu']:.0f} s"
    crashed = [d for d, (p, _, _) in processes.items() if p.returncode != 0]
    if crashed:
        print(f"{name:10s} FAILED: the {' and '.join(crashed)} run exited with an error "
              f"({timing})")
        for device_name in crashed:
            tail = (workdir / f"{name}_{device_name}.log").read_text().strip().splitlines()[-15:]
            print("\n".join(f"    {device_name}| {line}" for line in tail))
        return 0, 1

    data = {}
    for device_name in ("cpu", "gpu"):
        with np.load(workdir / f"{name}_{device_name}.npz") as npz:
            data[device_name] = {key: npz[key] for key in npz.files}
    sys.path.insert(0, str(HERE))
    from checks.common import Checker

    check = Checker(data["cpu"], data["gpu"])
    module.compare(check)
    passed = sum(ok for _, ok, _ in check.results)
    failed = len(check.results) - passed
    status = "ok" if failed == 0 else "FAILED"
    print(f"{name:10s} {status:6s} {passed}/{len(check.results)} comparisons passed ({timing}, "
          f"{time.time() - start:.0f} s total)")
    for key, ok, detail in check.results:
        if verbose or not ok:
            print(f"    {'pass' if ok else 'FAIL'}  {key}: {detail}")
    return passed, failed


def pytest_parity(workdir, verbose):
    """Run HOOMD's pytest suite and report tests that fail on the GPU but pass on the CPU.

    Many tests fail on both devices because they assume a double precision build. Those are
    reported but do not count as failures.
    """
    report = workdir / "pytest.xml"
    print("\npytest: running HOOMD's test suite on both devices (this takes a while)...")
    start = time.time()
    with open(workdir / "pytest.log", "w") as log:
        subprocess.run([sys.executable, "-m", "pytest", "--pyargs", "hoomd", "-q",
                        "-p", "no:cacheprovider", f"--junitxml={report}"],
                       cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    if not report.exists():
        print("pytest FAILED to run; see", workdir / "pytest.log")
        return 0, 1

    outcome = {}
    for case in ElementTree.parse(report).getroot().iter("testcase"):
        test = f"{case.get('classname')}::{case.get('name')}"
        if case.find("failure") is not None or case.find("error") is not None:
            outcome[test] = "failed"
        elif case.find("skipped") is not None:
            outcome[test] = "skipped"
        else:
            outcome[test] = "passed"

    def parameters(test):
        """Split "module::test[GPU-params3]" into ("module::test", ["GPU", "params3"])."""
        head, _, params = test.partition("[")
        return head, params[:-1].split("-") if params else []

    gpu_only, known = [], []
    for test in outcome:
        head, tokens = parameters(test)
        if "GPU" not in tokens:
            continue
        cpu_test = f"{head}[{'-'.join('CPU' if t == 'GPU' else t for t in tokens)}]"
        if outcome[test] == "failed" and outcome.get(cpu_test) == "passed":
            short = re.sub(r"\[.*\]$", "", test)
            (known if short in PYTEST_KNOWN_GPU_ONLY else gpu_only).append(test)
    counts = {k: sum(v == k for v in outcome.values()) for k in ("passed", "failed", "skipped")}
    print(f"pytest {counts['passed']} passed, {counts['failed']} failed, {counts['skipped']} "
          f"skipped in {time.time() - start:.0f} s (failures on both devices come from tests "
          f"that assume double precision)")
    print(f"pytest {'ok    ' if not gpu_only else 'FAILED'} {len(gpu_only)} tests fail only on "
          f"the GPU ({len(known)} more are known, explained cases)")
    for test in gpu_only:
        print(f"    FAIL  {test}")
    if verbose:
        for test in known:
            reason = PYTEST_KNOWN_GPU_ONLY[re.sub(r"\[.*\]$", "", test)]
            print(f"    known {test}: {reason}")
    return (1, 0) if not gpu_only else (0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checks", nargs="*", help="checks to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list the checks and exit")
    parser.add_argument("--pytest", action="store_true",
                        help="also run HOOMD's pytest suite and flag GPU-only failures")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every comparison")
    parser.add_argument("--keep", metavar="DIR", help="keep results and logs in DIR")
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(*args.worker)
        return 0

    unknown = [name for name in args.checks if name not in CHECKS]
    if unknown:
        parser.error(f"unknown checks {unknown}; choose from {list(CHECKS)}")

    if args.list:
        for name in CHECKS:
            doc = (load_check(name).__doc__ or "").strip().split("\n")[0]
            print(f"{name:10s} {doc}")
        return 0

    lines, problem = describe_system()
    print("\n".join(lines))
    if problem:
        print(problem)
        return 2
    print()

    if args.keep:
        workdir = pathlib.Path(args.keep).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = pathlib.Path(tempfile.mkdtemp(prefix="hoomd-metal-validate-"))

    passed = failed = 0
    for name in args.checks or CHECKS:
        p, f = run_check(name, workdir, args.verbose)
        passed, failed = passed + p, failed + f
    if args.pytest:
        p, f = pytest_parity(workdir, args.verbose)
        passed, failed = passed + p, failed + f

    print()
    if failed:
        print(f"FAILED: {failed} of {passed + failed} comparisons failed. "
              f"Results and logs: {workdir}")
        return 1
    print(f"PASSED: all {passed} comparisons agree between the CPU and the GPU.")
    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
