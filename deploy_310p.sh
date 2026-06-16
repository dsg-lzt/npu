#!/bin/bash
# Deploy and compile FPS kernel for Ascend 310P
# Run this script on the 310P server with CANN 8.3RC1 installed.
#
# Usage:
#   1. Copy the entire fps/ directory to the 310P server
#   2. Source CANN environment:  source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   3. cd fps && bash deploy_310p.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANN_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
OPP_PATH="${CANN_HOME}/opp"
DYN_DIR="${OPP_PATH}/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic"

echo "=== Deploying FurthestPointSample ==="
echo "CANN: ${CANN_HOME}"
mkdir -p "${DYN_DIR}"

# 1. op_proto
echo "[1/6] op_proto"
mkdir -p "${OPP_PATH}/vendors/customize/op_proto/inc"
cp -v "${SCRIPT_DIR}/op_proto/inc/furthest_point_sample_proto.h" \
      "${OPP_PATH}/vendors/customize/op_proto/inc/"

# 2. op_api
echo "[2/6] op_api"
mkdir -p "${OPP_PATH}/vendors/customize/op_api/include"
cp -v "${SCRIPT_DIR}/op_api/include/aclnn_furthest_point_sample.h" \
      "${OPP_PATH}/vendors/customize/op_api/include/"

# 3. tiling headers + op_host
echo "[3/7] tiling + op_host"
cp -v "${SCRIPT_DIR}/op_host/furthest_point_sampling_tiling.h"   "${DYN_DIR}/"
cp -v "${SCRIPT_DIR}/op_host/furthest_point_sampling.cpp"        "${DYN_DIR}/"

# 4. tiling data header (for kernel)
echo "[4/6] tiling data"
cp -v "${SCRIPT_DIR}/op_impl/ai_core/tbe/customize_impl/dynamic/fps_tiling_data.h" "${DYN_DIR}/"

# 5. kernel source
echo "[5/6] kernel"
cp -v "${SCRIPT_DIR}/op_impl/ai_core/tbe/customize_impl/dynamic/furthest_point_sample_kernel.cpp" "${DYN_DIR}/"

# 6. Python build script
echo "[6/6] build script"
cp -v "${SCRIPT_DIR}/op_impl/ai_core/tbe/customize_impl/dynamic/furthest_point_sample.py" "${DYN_DIR}/"

# 7. kernel config
echo "[7/7] config"
mkdir -p "${OPP_PATH}/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend310p"
cp -v "${SCRIPT_DIR}/op_impl/ai_core/tbe/kernel/config/ascend310p/furthest_point_sample.json" \
      "${OPP_PATH}/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend310p/"

echo ""
echo "=== Deployment complete ==="
echo "Files installed to: ${DYN_DIR}"
echo ""
echo "Next step: trigger compilation via the CANN operator build system."
