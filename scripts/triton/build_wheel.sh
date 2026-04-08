#!/usr/bin/env bash
# build_wheel.sh — Build a pip wheel for Triton from a pre-built source tree.
#
# Wraps pack_wheel.py which handles all file collection, symlink dereferencing,
# and wheel metadata generation.
#
# Usage:
#   ./build_wheel.sh                # output to ./dist/
#   ./build_wheel.sh /path/to/dist  # custom output directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="${1:-${SCRIPT_DIR}/dist}"

if [ ! -f "${SCRIPT_DIR}/python/triton/_C/libtriton.so" ]; then
    echo "ERROR: python/triton/_C/libtriton.so not found."
    echo "Build Triton first, then re-run this script."
    exit 1
fi

python3 "${SCRIPT_DIR}/pack_wheel.py" -o "${DIST_DIR}"
