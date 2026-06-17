/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename T>
class KernelGatherCustom {
public:
    __aicore__ inline KernelGatherCustom() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR indices, GM_ADDR y,
        uint64_t numIndices, uint64_t sliceLength,
        uint64_t sliceLoopNum, uint64_t ubSliceLen, uint64_t sliceTailLen,
        uint64_t smallCoreIndicesNum, uint64_t bigCoreIndicesNum, uint64_t tailBlockNum,
        TPipe* pipeIn)
    {
        pipe = pipeIn;
        this->numIndices = numIndices;
        this->sliceLength = sliceLength;
        this->sliceLoopNum = sliceLoopNum;
        this->ubSliceLen = ubSliceLen;
        this->sliceTailLen = sliceTailLen;

        uint64_t coreIdx = AscendC::GetBlockIdx();
        if (coreIdx < tailBlockNum) {
            this->myIndicesNum = bigCoreIndicesNum;
        } else {
            this->myIndicesNum = smallCoreIndicesNum;
        }

        this->indicesStart = 0;
        for (uint64_t i = 0; i < coreIdx; ++i) {
            this->indicesStart += (i < tailBlockNum) ? bigCoreIndicesNum : smallCoreIndicesNum;
        }
        this->indicesEnd = this->indicesStart + this->myIndicesNum;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        this->indicesGm = (__gm__ int32_t*)indices;

        pipe->InitBuffer(copyQueue, BUFFER_NUM, this->ubSliceLen * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint64_t idx = this->indicesStart; idx < this->indicesEnd; ++idx) {
            int32_t gatherIdx = this->indicesGm[idx];
            uint64_t srcOffset = gatherIdx * this->sliceLength;
            uint64_t dstOffset = idx * this->sliceLength;

            for (uint64_t tile = 0; tile < this->sliceLoopNum; ++tile) {
                CopyIn(srcOffset + tile * this->ubSliceLen, this->ubSliceLen);
                CopyOut(dstOffset + tile * this->ubSliceLen, this->ubSliceLen);
            }
            if (this->sliceLoopNum * this->ubSliceLen < this->sliceLength) {
                uint64_t tailLen = this->sliceTailLen;
                if (this->sliceLoopNum == 0) {
                    tailLen = this->sliceLength;
                }
                CopyIn(srcOffset + this->sliceLoopNum * this->ubSliceLen, tailLen);
                CopyOut(dstOffset + this->sliceLoopNum * this->ubSliceLen, tailLen);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint64_t len)
    {
        LocalTensor<T> local = copyQueue.AllocTensor<T>();
        DataCopy(local, xGm[offset], len);
        copyQueue.EnQue(local);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint64_t len)
    {
        LocalTensor<T> local = copyQueue.DeQue<T>();
        DataCopy(yGm[offset], local, len);
        copyQueue.FreeTensor(local);
    }

private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    __gm__ int32_t* indicesGm;
    TPipe* pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> copyQueue;

    uint64_t numIndices;
    uint64_t sliceLength;
    uint64_t sliceLoopNum;
    uint64_t ubSliceLen;
    uint64_t sliceTailLen;
    uint64_t myIndicesNum;
    uint64_t indicesStart;
    uint64_t indicesEnd;
};

extern "C" __global__ __aicore__ void gather_custom(
    GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelGatherCustom<DTYPE_X> op;
    op.Init(x, indices, y,
            tiling_data.numIndices, tiling_data.sliceLength,
            tiling_data.sliceLoopNum, tiling_data.ubSliceLen, tiling_data.sliceTailLen,
            tiling_data.smallCoreIndicesNum, tiling_data.bigCoreIndicesNum, tiling_data.tailBlockNum,
            &pipe);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gather_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* indices, uint8_t* y,
    uint8_t* workspace, uint8_t* tiling)
{
    gather_custom<<<blockDim, l2ctrl, stream>>>(x, indices, y, workspace, tiling);
}
#endif
