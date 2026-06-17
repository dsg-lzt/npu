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
constexpr int32_t ONE_BLK_BYTES = 32;

template<typename T>
class KernelGatherCustom {
public:
    __aicore__ inline KernelGatherCustom() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR indices, GM_ADDR y,
        uint64_t numIndices, uint64_t sliceLength,
        uint64_t outerLength, uint64_t gatherDimSize,
        uint64_t sliceLoopNum, uint64_t ubSliceLen, uint64_t sliceTailLen,
        uint64_t smallCoreIndicesNum, uint64_t bigCoreIndicesNum, uint64_t tailBlockNum,
        TPipe* pipeIn, bool usePad)
    {
        pipe = pipeIn;
        this->numIndices = numIndices;
        this->sliceLength = sliceLength;
        this->outerLength = outerLength;
        this->gatherDimSize = gatherDimSize;
        this->sliceLoopNum = sliceLoopNum;
        this->ubSliceLen = ubSliceLen;
        this->sliceTailLen = sliceTailLen;
        this->usePad = usePad;

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

        if (usePad) {
            pipe->InitBuffer(padBuf, this->ubSliceLen * sizeof(T));
        } else {
            pipe->InitBuffer(copyQueue, BUFFER_NUM, this->ubSliceLen * sizeof(T));
        }

        PRINTF("[GatherCustom] coreIdx=%lu blockNum=%lu usePad=%d\n",
               coreIdx, AscendC::GetBlockNum(), usePad ? 1 : 0);
        PRINTF("[GatherCustom] outerLength=%lu gatherDimSize=%lu sliceLength=%lu\n",
               this->outerLength, this->gatherDimSize, this->sliceLength);
        PRINTF("[GatherCustom] ubSliceLen=%lu sliceLoopNum=%lu sliceTailLen=%lu\n",
               this->ubSliceLen, this->sliceLoopNum, this->sliceTailLen);
        PRINTF("[GatherCustom] indicesStart=%lu indicesEnd=%lu myIndicesNum=%lu\n",
               this->indicesStart, this->indicesEnd, this->myIndicesNum);
    }

    __aicore__ inline void Process()
    {
        uint64_t groupSize = this->gatherDimSize * this->sliceLength;
        uint64_t outGroupSize = this->numIndices * this->sliceLength;

        for (uint64_t outer = 0; outer < this->outerLength; ++outer) {
            uint64_t outerSrcBase = outer * groupSize;
            uint64_t outerDstBase = outer * outGroupSize;

            for (uint64_t idx = this->indicesStart; idx < this->indicesEnd; ++idx) {
                int32_t gatherIdx = this->indicesGm[idx];
                uint64_t srcBase = outerSrcBase + gatherIdx * this->sliceLength;
                uint64_t dstBase = outerDstBase + idx * this->sliceLength;

                if (this->usePad) {
                    TiledCopyPad(srcBase, dstBase);
                } else {
                    for (uint64_t tile = 0; tile < this->sliceLoopNum; ++tile) {
                        CopyData(srcBase + tile * this->ubSliceLen,
                                 dstBase + tile * this->ubSliceLen,
                                 this->ubSliceLen);
                    }
                    if (this->sliceTailLen > 0) {
                        CopyData(srcBase + this->sliceLoopNum * this->ubSliceLen,
                                 dstBase + this->sliceLoopNum * this->ubSliceLen,
                                 this->sliceTailLen);
                    }
                }
            }
        }
        PRINTF("[GatherCustom] coreDone\n");
    }

private:
    // ---------- Aligned: TQueBind double-buffer pipeline ----------
    __aicore__ inline void CopyData(uint64_t srcOffset, uint64_t dstOffset, uint64_t len)
    {
        CopyIn(srcOffset, len);
        CopyOut(dstOffset, len);
    }

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

    // ---------- Unaligned: TBuf + DataCopyPad + PipeBarrier ----------
    __aicore__ inline void TiledCopyPad(uint64_t srcBase, uint64_t dstBase)
    {
        for (uint64_t tile = 0; tile < this->sliceLoopNum; ++tile) {
            CopyPad(srcBase + tile * this->ubSliceLen,
                    dstBase + tile * this->ubSliceLen,
                    this->ubSliceLen);
        }
        if (this->sliceTailLen > 0) {
            CopyPad(srcBase + this->sliceLoopNum * this->ubSliceLen,
                    dstBase + this->sliceLoopNum * this->ubSliceLen,
                    this->sliceTailLen);
        }
    }

    __aicore__ inline void CopyPad(uint64_t srcOffset, uint64_t dstOffset, uint64_t len)
    {
        LocalTensor<T> local = padBuf.Get<T>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(len * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = static_cast<T>(0);
        DataCopyPad(local, xGm[srcOffset], copyParams, padParams);
        PipeBarrier<PIPE_MTE2>();
        DataCopyPad(yGm[dstOffset], local, copyParams);
        PipeBarrier<PIPE_MTE3>();
    }

private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    __gm__ int32_t* indicesGm;
    TPipe* pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> copyQueue;
    TBuf<QuePosition::VECCALC> padBuf;

    uint64_t numIndices;
    uint64_t sliceLength;
    uint64_t outerLength;
    uint64_t gatherDimSize;
    uint64_t sliceLoopNum;
    uint64_t ubSliceLen;
    uint64_t sliceTailLen;
    uint64_t myIndicesNum;
    uint64_t indicesStart;
    uint64_t indicesEnd;
    bool usePad;
};

extern "C" __global__ __aicore__ void gather_custom(
    GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    bool usePad = (tiling_data.sliceLength * sizeof(DTYPE_X) % ONE_BLK_BYTES != 0);
    TPipe pipe;
    KernelGatherCustom<DTYPE_X> op;
    op.Init(x, indices, y,
            tiling_data.numIndices, tiling_data.sliceLength,
            tiling_data.outerLength, tiling_data.gatherDimSize,
            tiling_data.sliceLoopNum, tiling_data.ubSliceLen, tiling_data.sliceTailLen,
            tiling_data.smallCoreIndicesNum, tiling_data.bigCoreIndicesNum, tiling_data.tailBlockNum,
            &pipe, usePad);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gather_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* indices, uint8_t* y,
    uint8_t* workspace, uint8_t* tiling)
{
    gather_custom<<<blockDim, l2ctrl, stream>>>(x, indices, y, workspace, tiling);
}
#endif
