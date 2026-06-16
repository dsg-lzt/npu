#ifndef FURTHEST_POINT_SAMPLE_PROTO_H_
#define FURTHEST_POINT_SAMPLE_PROTO_H_
#include "graph/operator_reg.h"
namespace ge {
REG_OP(FurthestPointSample)
    .INPUT(points, ge::TensorType::ALL())
    .OUTPUT(sampled, ge::TensorType::ALL())
    .REQUIRED_ATTR(m, ge::AttrValue::INT)
    .OP_END_FACTORY_REG(FurthestPointSample);
}
#endif

# Sync and build
cp /home/lzt/ascend_project/fps/src/math/furthest_point_sample/op_kernel/furthest_point_sample.cpp \
   /home/lzt/ascend_project/cann-ops/src/math/furthest_point_sample/op_kernel/

source /home/lzt/miniconda3/bin/activate ascend_env
export LD_LIBRARY_PATH="/home/lzt/Ascend/ascend-toolkit/8.3.RC1/lib64:/home/lzt/Ascend/ascend-toolkit/8.3.RC1/x86_64-linux/lib64:/home/lzt/Ascend/ascend-toolkit/8.3.RC1/x86_64-linux/devlib:${LD_LIBRARY_PATH}"
cd /home/lzt/ascend_project/cann-ops && rm -rf build output && \
bash build.sh -n furthest_point_sample -c ascend310p 2>&1 | tail -5
