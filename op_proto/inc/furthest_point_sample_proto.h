#ifndef FURTHEST_POINT_SAMPLE_PROTO_H_
#define FURTHEST_POINT_SAMPLE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(FurthestPointSample)
    .INPUT(points, ge::TensorType::ALL())
    .INPUT(temp, ge::TensorType::ALL())
    .OUTPUT(sampled, ge::TensorType::ALL())
    .REQUIRED_ATTR(m, ge::AttrValue::INT)
    .OP_END_FACTORY_REG(FurthestPointSample);

}  // namespace ge

#endif  // FURTHEST_POINT_SAMPLE_PROTO_H_
