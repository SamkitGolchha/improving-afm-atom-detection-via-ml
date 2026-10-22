#!/usr/bin/env bash
# Build ppafm's C++ helpers on macOS (Apple clang has no built-in OpenMP).
# Points the build at an existing libomp (default: Anaconda's) and compiles in the venv.
# Separately, Apple's OpenCL compiler keeps pyopencl's quotes around ppafm's "-I <kernel dir>"
# option; our scripts work around that by building kernels from inside ppafm/ocl/cl.
# Usage: VENV=~/.venvs/afm-spinoff scripts/build_ppafm_macos.sh [OMP_PREFIX]
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/afm-spinoff}"
OMP_PREFIX="${1:-$HOME/anaconda3}"
CPP_DIR="$("$VENV/bin/python" -I -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("ppafm").origin).parent / "cpp")')"

test -f "$OMP_PREFIX/include/omp.h" && test -f "$OMP_PREFIX/lib/libomp.dylib" || {
  echo "libomp not found under $OMP_PREFIX" >&2; exit 1; }

cd "$CPP_DIR"
make clean
make all \
  CPPFLAGS="-fPIC -std=c++11 -O2 -Xpreprocessor -fopenmp -I$OMP_PREFIX/include" \
  LDFLAGS="-shared -L$OMP_PREFIX/lib -lomp -Wl,-rpath,$OMP_PREFIX/lib"
ls -l "$CPP_DIR"/*_lib.so
