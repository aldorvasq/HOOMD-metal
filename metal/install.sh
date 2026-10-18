#!/usr/bin/env bash
# Copyright (c) 2026 Aldo Vásquez-Briceño.
# Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

# Build HOOMD-metal and install it into a conda environment (created when needed).
#
# Usage: metal/install.sh [options]
#
#   -n, --name NAME       conda environment name (default: hoomd-metal)
#   -j, --jobs N          parallel compile jobs (default: chosen from memory and cores)
#   -b, --build-dir DIR   build directory (default: build/hoomd-metal in the repository)
#       --no-mpcd         do not build the MPCD component
#       --validate        run the CPU vs GPU validation checks after installing
#   -h, --help            show this help
#
# Afterwards: conda activate hoomd-metal

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="hoomd-metal"
JOBS=""
BUILD_DIR="$REPO/build/hoomd-metal"
BUILD_MPCD="ON"
VALIDATE=0

die() {
    echo "error: $*" >&2
    exit 1
}

step() {
    echo
    echo "==> $*"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n | --name) ENV_NAME="$2"; shift 2 ;;
        -j | --jobs) JOBS="$2"; shift 2 ;;
        -b | --build-dir) BUILD_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"; shift 2 ;;
        --no-mpcd) BUILD_MPCD="OFF"; shift ;;
        --validate) VALIDATE=1; shift ;;
        -h | --help) sed -n '5,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option '$1' (see --help)" ;;
    esac
done

step "Checking this Mac"
[[ "$(uname -s)" == "Darwin" ]] || die "HOOMD-metal runs on macOS only."
[[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]] ||
    die "This Mac does not have Apple silicon. The Metal backend requires an M1 or newer;" \
        "build upstream HOOMD-blue for the CPU instead."
macos="$(sw_vers -productVersion)"
(( ${macos%%.*} >= 15 )) || die "macOS $macos is too old: HOOMD-metal requires macOS 15 or newer."
xcrun --find clang++ > /dev/null 2>&1 ||
    die "Apple's command line tools are missing. Install them with: xcode-select --install"
[[ -d "$(xcrun --show-sdk-path)/System/Library/Frameworks/Metal.framework" ]] ||
    die "The macOS SDK has no Metal framework. Reinstall the command line tools."
chip="$(sysctl -n machdep.cpu.brand_string)"
memory_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
echo "$chip, ${memory_gb} GB, macOS $macos"

CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA" ]]; then
    for candidate in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" \
        /opt/miniforge3 /opt/miniconda3 /opt/anaconda3 /opt/homebrew/Caskroom/miniforge/base; do
        if [[ -x "$candidate/bin/conda" ]]; then
            CONDA="$candidate/bin/conda"
            break
        fi
    done
fi
[[ -n "$CONDA" ]] ||
    die "conda was not found. Install Miniforge from https://github.com/conda-forge/miniforge"
conda_platform="$("$CONDA" info 2> /dev/null | awk '$1 == "platform" {print $3}')"
[[ "$conda_platform" == "osx-arm64" ]] ||
    die "$CONDA is a '$conda_platform' conda. On Apple silicon it must be osx-arm64, not the" \
        "Intel build running under Rosetta. Install the arm64 Miniforge."
echo "conda: $CONDA ($conda_platform)"

env_prefix() {
    "$CONDA" env list | awk -v name="$1" '$1 == name {print $NF}'
}

PREFIX="$(env_prefix "$ENV_NAME")"
if [[ -n "$PREFIX" ]]; then
    if compgen -G "$PREFIX/conda-meta/hoomd-[0-9]*.json" > /dev/null; then
        die "The environment '$ENV_NAME' contains the conda-forge hoomd package. Remove it" \
            "(conda remove -n $ENV_NAME hoomd) or choose another name with --name."
    fi
    step "Updating the conda environment '$ENV_NAME' ($PREFIX)"
    "$CONDA" env update --quiet --name "$ENV_NAME" --file "$REPO/metal/environment.yml"
else
    step "Creating the conda environment '$ENV_NAME'"
    "$CONDA" env create --quiet --name "$ENV_NAME" --file "$REPO/metal/environment.yml"
    PREFIX="$(env_prefix "$ENV_NAME")"
    [[ -n "$PREFIX" ]] || die "conda did not create the environment '$ENV_NAME'."
fi
PY="$PREFIX/bin/python"
[[ "$("$PY" -c 'import platform; print(platform.machine())')" == "arm64" ]] ||
    die "$PY is not an arm64 Python."

if [[ -d "$REPO/.git" && -z "$(ls -A "$REPO/hoomd/extern/nano-signal-slot" 2> /dev/null)" ]]; then
    step "Fetching git submodules"
    git -C "$REPO" submodule update --init --recursive -- hoomd/extern
fi
[[ -n "$(ls -A "$REPO/hoomd/extern/nano-signal-slot" 2> /dev/null)" ]] ||
    die "The git submodules are missing. Clone with: git clone --recursive <url>"

if [[ -z "$JOBS" ]]; then
    # about 2 GB of memory per compile job, at most one per core
    cores="$(sysctl -n hw.ncpu)"
    JOBS=$(( memory_gb / 2 ))
    (( JOBS > cores )) && JOBS=$cores
    (( JOBS < 1 )) && JOBS=1
fi

# Keep the Python of an activated virtual environment, or an older HOOMD on PYTHONPATH, from
# leaking into the build and the tests.
unset VIRTUAL_ENV PYTHONPATH PYTHONHOME

step "Configuring in $BUILD_DIR"
mkdir -p "$BUILD_DIR"
site_dir="$("$PY" -c 'import os, sys, sysconfig
print(os.path.relpath(sysconfig.get_paths()["platlib"], sys.prefix))')/hoomd"
"$PREFIX/bin/cmake" -S "$REPO" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GPU=ON -DHOOMD_GPU_PLATFORM=Metal \
    -DHOOMD_LONGREAL_SIZE=32 -DHOOMD_SHORTREAL_SIZE=32 \
    -DBUILD_HPMC=OFF -DENABLE_MPI=OFF -DBUILD_MPCD="$BUILD_MPCD" -DBUILD_TESTING=OFF \
    -DPython_EXECUTABLE="$PY" -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DPYTHON_SITE_INSTALL_DIR="$site_dir" \
    > "$BUILD_DIR/configure.log" 2>&1 ||
    { tail -30 "$BUILD_DIR/configure.log"; die "configuration failed (log: $BUILD_DIR/configure.log)"; }

step "Building with $JOBS jobs (about 5 minutes on an M4, longer on smaller Macs)"
start=$SECONDS
"$PREFIX/bin/cmake" --build "$BUILD_DIR" -j "$JOBS"
echo "built in $(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s"

step "Installing into $PREFIX/$site_dir"
"$PREFIX/bin/cmake" --install "$BUILD_DIR" > /dev/null

step "Testing the installation"
smoke_dir="$(mktemp -d)"
(cd "$smoke_dir" && "$PY" - << 'EOF'
import time

import hoomd
import numpy as np

print(f"hoomd {hoomd.version.version} ({hoomd.version.gpu_platform}) from {hoomd.__file__}")
device = hoomd.device.GPU()
print("GPU:", device.device)
n = 10
L = n * 1.2
x = (np.arange(n) + 0.5) * L / n - L / 2
snapshot = hoomd.Snapshot()
snapshot.particles.N = n**3
snapshot.particles.types = ["A"]
snapshot.configuration.box = [L, L, L, 0, 0, 0]
snapshot.particles.position[:] = np.array(np.meshgrid(x, x, x)).reshape(3, -1).T
sim = hoomd.Simulation(device=device, seed=1)
sim.create_state_from_snapshot(snapshot)
lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=2.5)
lj.params[("A", "A")] = dict(epsilon=1, sigma=1)
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
sim.operations.integrator = hoomd.md.Integrator(dt=0.005, methods=[langevin], forces=[lj])
sim.run(100)
start = time.time()
sim.run(1000)
energy = lj.energy
assert np.isfinite(energy), "the LJ energy is not finite"
print(f"1000-particle LJ test: {1000 / (time.time() - start):.0f} steps/s, energy {energy:.1f}")
EOF
) || { rm -rf "$smoke_dir"; die "the installed hoomd failed its smoke test."; }
rm -rf "$smoke_dir"

if (( VALIDATE )); then
    step "Validating the GPU against the CPU"
    "$PY" "$REPO/metal/validate/validate.py" || die "validation failed (see above)."
fi

cat << EOF

HOOMD-metal is installed in the conda environment '$ENV_NAME'. To use it:

    conda activate $ENV_NAME
    python your_script.py            # with hoomd.device.GPU()

Check the GPU against the CPU on this Mac:  python $REPO/metal/validate/validate.py
Uninstall:  conda env remove -n $ENV_NAME
            (or keep the environment: xargs rm < $BUILD_DIR/install_manifest.txt)
EOF
