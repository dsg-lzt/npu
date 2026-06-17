#include "furthest_point_sampling_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

const uint32_t BUFFER_NUM = 2;
const uint32_t BLOCK_SIZE = 32;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    FurthestPointSamplingTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    auto aivNum = ascendcPlatform.GetCoreNum();

    const gert::StorageShape* x1_shape = context->GetInputShape(0);

    uint32_t B = x1_shape->GetStorageShape().GetDim(0);
    uint32_t N = x1_shape->GetStorageShape().GetDim(2);
    uint32_t npoint = *context->GetAttrs()->GetInt(0);

    uint32_t sizeofdatatype;
    auto dt = context->GetInputTensor(0)->GetDataType();
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_FLOAT) {
        sizeofdatatype = 2;
    } else {
        sizeofdatatype = 4;
    }

    int32_t localtensor_NUM = 10;
    aivNum = (aivNum <= B) ? aivNum : B;
    aivNum = aivNum >= 1 ? aivNum : 1;

    uint32_t b_num = B / aivNum;
    uint32_t remain = B - aivNum * b_num;

    uint32_t idx_size = (npoint + 7) / 8 * 8;
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t tiling_size =
        ((ubSize - idx_size * 4) / BLOCK_SIZE / BUFFER_NUM) / localtensor_NUM;
    tiling_size =
        tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;
    uint32_t block_size = tiling_size * ALIGN_NUM;

    block_size =
        block_size < N ? block_size : (N + 7) / 8 * 8;

    uint32_t length = N / block_size;
    length = N % block_size == 0 ? length : length + 1;

    tiling.set_N(N);
    tiling.set_npoint(npoint);
    tiling.set_b_num(b_num);
    tiling.set_remain(remain);
    tiling.set_block_size(block_size);
    tiling.set_idx_size(idx_size);
    tiling.set_length(length);

    context->SetBlockDim(aivNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class FurthestPointSampling : public OpDef {
public:
    explicit FurthestPointSampling(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("tmp")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("npoint").Int();

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(FurthestPointSampling);
}