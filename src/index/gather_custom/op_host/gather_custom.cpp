/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

namespace optiling {
const uint64_t BLOCK_SIZE = 32;

static int32_t GetAxis(gert::TilingContext* context)
{
    int32_t xDimNum = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    return (xDimNum >= 3) ? 1 : 0;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GatherCustomTilingData tiling;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubLength = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto coreNum = ascendcPlatform.GetCoreNum();

    uint64_t numIndices = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    int32_t axis = GetAxis(context);
    int32_t xDimNum = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint64_t outerLength = 1;
    for (int32_t d = 0; d < axis; ++d) {
        outerLength *= context->GetInputShape(0)->GetStorageShape().GetDim(d);
    }
    uint64_t gatherDimSize = context->GetInputShape(0)->GetStorageShape().GetDim(axis);
    uint64_t sliceLength = 1;
    for (int32_t d = axis + 1; d < xDimNum; ++d) {
        sliceLength *= context->GetInputShape(0)->GetStorageShape().GetDim(d);
    }

    uint32_t dataTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), dataTypeLength);
    if (dataTypeLength == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t minBufLen = BLOCK_SIZE / dataTypeLength;
    uint64_t ubSliceLen = (ubLength / 4) / dataTypeLength;
    ubSliceLen = (ubSliceLen / minBufLen) * minBufLen;
    if (ubSliceLen < minBufLen) {
        ubSliceLen = minBufLen;
    }

    uint64_t sliceLoopNum;
    uint64_t sliceTailLen;
    if (ubSliceLen >= sliceLength) {
        sliceLoopNum = 0;
        sliceTailLen = sliceLength;
    } else {
        sliceLoopNum = sliceLength / ubSliceLen;
        sliceTailLen = sliceLength - sliceLoopNum * ubSliceLen;
    }

    uint64_t indicesPerCore = numIndices / coreNum;
    uint64_t tailIndices = numIndices % coreNum;
    uint64_t smallCoreIndicesNum = indicesPerCore;
    uint64_t bigCoreIndicesNum = indicesPerCore + (tailIndices > 0 ? 1 : 0);

    tiling.set_numIndices(numIndices);
    tiling.set_sliceLength(sliceLength);
    tiling.set_outerLength(outerLength);
    tiling.set_gatherDimSize(gatherDimSize);
    tiling.set_sliceLoopNum(sliceLoopNum);
    tiling.set_ubSliceLen(ubSliceLen);
    tiling.set_sliceTailLen(sliceTailLen);
    tiling.set_smallCoreIndicesNum(smallCoreIndicesNum);
    tiling.set_bigCoreIndicesNum(bigCoreIndicesNum);
    tiling.set_tailBlockNum(tailIndices);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static int32_t InferGetAxis(gert::InferShapeContext* context)
{
    int32_t xDimNum = context->GetInputShape(0)->GetDimNum();
    return (xDimNum >= 3) ? 1 : 0;
}

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* xShape = context->GetInputShape(0);
    const auto* idxShape = context->GetInputShape(1);
    auto* yShape = context->GetOutputShape(0);

    int32_t axis = InferGetAxis(context);
    int64_t xDimNum = xShape->GetDimNum();
    int64_t idxDimNum = idxShape->GetDimNum();

    yShape->SetDimNum(xDimNum);
    for (int64_t i = 0; i < axis; ++i) {
        yShape->SetDim(i, xShape->GetDim(i));
    }
    for (int64_t j = 0; j < idxDimNum; ++j) {
        yShape->SetDim(axis + j, idxShape->GetDim(j));
    }
    for (int64_t i = axis + 1; i < xDimNum; ++i) {
        yShape->SetDim(idxDimNum + i - 1, xShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class GatherCustom : public OpDef {
public:
    explicit GatherCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310p");
    }
};

OP_ADD(GatherCustom);
}
