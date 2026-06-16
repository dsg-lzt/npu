/**
 * FurthestPointSample - Host-side Op Definition & Tiling Computation
 *
 * Input:  points  [B, N, 3]   float32 | float16
 * Output: sampled [B, M, 3]
 * Attr:   m       (int)
 */

#include "furthest_point_sample_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

constexpr int32_t COORD_DIM = 3;

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // ── 1. Extract problem shape ──────────────────────────────────────────
    const gert::StorageShape *ss = context->GetInputShape(0);
    const gert::Shape &shape = ss->GetOriginShape();
    int32_t B = static_cast<int32_t>(shape.GetDim(0));
    int32_t N = static_cast<int32_t>(shape.GetDim(1));

    // Read M from op attribute (index 0 = first attr)
    int32_t M = 0;
    auto attrs = context->GetAttrs();
    const int64_t *mPtr = attrs->GetInt(0);
    if (mPtr) M = static_cast<int32_t>(*mPtr);

    // ── 2. Data type size ─────────────────────────────────────────────────
    uint32_t dataTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(
        context->GetInputDesc(0)->GetDataType(), dataTypeLength);

    // ── 3. Platform info ──────────────────────────────────────────────────
    auto platform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = platform.GetCoreNum();
    if (coreNum == 0) coreNum = 1;

    // ── 4. Compute UB tile sizing ─────────────────────────────────────────
    // UB budget: 3 coord queues(d2) + mdBuf(N) + dist+tmp+sca = 9*tileN + N
    uint64_t ubElm = ubSize / dataTypeLength;
    uint32_t tileN = static_cast<uint32_t>((ubElm - N) / 9);
    if (tileN > static_cast<uint32_t>(N)) tileN = static_cast<uint32_t>(N);
    if (tileN < 64) tileN = 64;  // minimum reasonable tile
    uint32_t numTiles = (N + tileN - 1) / tileN;

    // ── 5. Multi-core batch distribution ──────────────────────────────────
    uint32_t batchesPerCore = (B + coreNum - 1) / coreNum;
    uint32_t coreRemainder   = B % coreNum;

    // ── 6. No workspace needed (minDist in UB) ─────────────────────
    constexpr float kFp32Init = 3.402823e+38f;
    constexpr float kFp16Init = 65504.0f;
    uint32_t totalWsBytes = B * N * dataTypeLength;

    // ── 7. Tiling key ─────────────────────────────────────────────────────
    int32_t tilingKey = (dataTypeLength == 4) ? 0 : 1;

    // ── 8. Populate tiling data ───────────────────────────────────────────
    FpsTilingData tiling;
    tiling.set_B(B);
    tiling.set_N(N);
    tiling.set_M(M);
    tiling.set_C(COORD_DIM);
    tiling.set_dataTypeLength(dataTypeLength);
    tiling.set_ubPointsNum(tileN);
    tiling.set_ubMinDistNum(tileN);
    tiling.set_tileLoopNum(numTiles);
    tiling.set_batchPerCore(batchesPerCore);
    tiling.set_coreRemainder(coreRemainder);
    tiling.set_wsOffset(0);
    tiling.set_wsStride(N);  // elements, not bytes
    tiling.set_initVal((dataTypeLength == 4) ? kFp32Init : kFp16Init);

    // ── 9. Set blockDim & tiling key ──────────────────────────────────────
    context->SetTilingKey(static_cast<uint64_t>(tilingKey));
    context->SetBlockDim(coreNum);

    // ── 10. Serialize tiling ──────────────────────────────────────────────
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // ── 11. Workspace size ────────────────────────────────────────────────
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = totalWsBytes;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

// ─── Shape / Dtype Inference ──────────────────────────────────────────────
namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    int32_t B = static_cast<int32_t>(inputShape->GetDim(0));

    int32_t M = 0;
    auto attrs = context->GetAttrs();
    const int64_t *mPtr = attrs->GetInt(0);
    if (mPtr) M = static_cast<int32_t>(*mPtr);

    outputShape->SetDim(0, B);
    outputShape->SetDim(1, M);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

// ─── Op Definition ────────────────────────────────────────────────────────
namespace ops {

class FurthestPointSample : public OpDef {
public:
    explicit FurthestPointSample(const char *name) : OpDef(name)
    {
        this->Input("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("sampled")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("m").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310p");
    }
};

OP_ADD(FurthestPointSample);

}  // namespace ops
