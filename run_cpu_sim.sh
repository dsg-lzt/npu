#!/bin/bash
# Build and run CPU simulation of furthest_point_sample operator
# Target: Ascend 310P, CANN 8.3RC1

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ASCEND_HOME="/home/lzt/Ascend/ascend-toolkit/8.3.RC1"
ARCH_DIR="${ASCEND_HOME}/x86_64-linux"

export LD_LIBRARY_PATH="${ARCH_DIR}/simulator/Ascend310P1/lib:${ARCH_DIR}/devlib:${ARCH_DIR}/lib64:${ASCEND_HOME}/tools/tikicpulib/lib:${ASCEND_HOME}/tools/tikicpulib/lib/Ascend310P1:${LD_LIBRARY_PATH}"

echo "=== Building furthest_point_sample ==="
mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j$(nproc)

echo ""
echo "=== Running CPU simulation ==="
"${BUILD_DIR}/furthest_point_sample"
