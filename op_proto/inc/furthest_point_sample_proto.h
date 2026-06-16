/**
 * Furthest Point Sample operator registration
 * Input:  points  (B, N, 3)
 * Output: sampled (B, M, 3)
 * Attr:   M (int32_t) - number of points to sample
 */
#ifndef FURTHEST_POINT_SAMPLE_PROTO_H_
#define FURTHEST_POINT_SAMPLE_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(FurthestPointSample)
    .INPUT(points, ge::TensorType::ALL())
    .OUTPUT(sampled, ge::TensorType::ALL())
    .REQUIRED_ATTR(m, ge::AttrValue::INT)
    .OP_END_FACTORY_REG(FurthestPointSample);

}  // namespace ge

#endif  // FURTHEST_POINT_SAMPLE_PROTO_H_
