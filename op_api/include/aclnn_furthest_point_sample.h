/**
 * Host-side ACLNN launcher for FurthestPointSample
 *   - aclnnFurthestPointSampleGetWorkspaceSize: query workspace
 *   - aclnnFurthestPointSample: launch kernel
 *
 * Input  tensors: points   = [B, N, 3]  (float32 / float16)
 * Output tensors: sampled  = [B, M, 3]
 * Attr           : m       (int64_t)
 */
#ifndef ACLNN_FURTHEST_POINT_SAMPLE_H_
#define ACLNN_FURTHEST_POINT_SAMPLE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Query workspace size required by FurthestPointSample.
 *
 * @param points        [IN]  input tensor, shape [B, N, 3], dtype float32/float16
 * @param sampled       [IN]  output tensor, shape [B, M, 3]
 * @param m             [IN]  number of output points (int64_t)
 * @param workspaceSize [OUT] required workspace size in bytes
 * @param executor      [OUT] op executor handle
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFurthestPointSampleGetWorkspaceSize(
    const aclTensor *points,
    const aclTensor *sampled,
    int64_t m,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * Execute FurthestPointSample on the current stream.
 *
 * @param workspace     [IN]  workspace buffer
 * @param workspaceSize [IN]  workspace size in bytes
 * @param executor      [IN]  executor handle from GetWorkspaceSize
 * @param stream        [IN]  acl stream
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFurthestPointSample(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_FURTHEST_POINT_SAMPLE_H_
