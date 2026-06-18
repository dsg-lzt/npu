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
        TPipe* pipeIn)
    {
        pipe = pipeIn;
        this->numIndices = numIndices;
        this->sliceLength = sliceLength;
        this->outerLength = outerLength;
        this->gatherDimSize = gatherDimSize;
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
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)(indices));

        this->numPerBlock = ONE_BLK_BYTES / sizeof(T);
        this->sliceAligned = (sliceLength * sizeof(T) % ONE_BLK_BYTES == 0);

        // Batch indices: read ~8KB of int32 indices into UB at once
        uint64_t idxBufBytes = 8192;
        uint64_t idxPerBatch = idxBufBytes / sizeof(int32_t);
        if (idxPerBatch > this->myIndicesNum) idxPerBatch = this->myIndicesNum;
        this->indicesBatch = idxPerBatch;
        pipe->InitBuffer(indicesBuf, this->indicesBatch * sizeof(int32_t));

        if (this->sliceAligned) {
            pipe->InitBuffer(copyQueue, BUFFER_NUM, this->ubSliceLen * sizeof(T));
        } else {
            this->numPerBlock32 = this->numPerBlock;
            pipe->InitBuffer(rmwQueue, BUFFER_NUM, this->numPerBlock32 * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->sliceAligned) {
            ProcessAligned();
        } else {
            ProcessUnaligned();
        }
    }

private:
    // ============ Aligned: batch indices + TQueBind per slice ============
    __aicore__ inline void ProcessAligned()
    {
        uint64_t groupSize = this->gatherDimSize * this->sliceLength;
        uint64_t outGroupSize = this->numIndices * this->sliceLength;

        for (uint64_t outer = 0; outer < this->outerLength; ++outer) {
            uint64_t outerSrcBase = outer * groupSize;
            uint64_t outerDstBase = outer * outGroupSize;

            uint64_t remaining = this->myIndicesNum;
            uint64_t idxBase = this->indicesStart;

            while (remaining > 0) {
                uint64_t batch = remaining;
                if (batch > this->indicesBatch) batch = this->indicesBatch;
                remaining -= batch;

                LocalTensor<int32_t> idxUb = indicesBuf.Get<int32_t>();
                DataCopy(idxUb, indicesGm[idxBase], batch);
                PipeBarrier<PIPE_MTE2>();

                for (uint64_t i = 0; i < batch; ++i) {
                    int32_t gatherIdx = idxUb.GetValue(i);
                    uint64_t srcBase = outerSrcBase + gatherIdx * this->sliceLength;
                    uint64_t dstBase = outerDstBase + (idxBase + i) * this->sliceLength;

                    for (uint64_t tile = 0; tile < this->sliceLoopNum; ++tile) {
                        CopyIn(srcBase + tile * this->ubSliceLen, this->ubSliceLen);
                        CopyOut(dstBase + tile * this->ubSliceLen, this->ubSliceLen);
                    }
                    if (this->sliceTailLen > 0) {
                        uint64_t tailOff = this->sliceLoopNum * this->ubSliceLen;
                        CopyIn(srcBase + tailOff, this->sliceTailLen);
                        CopyOut(dstBase + tailOff, this->sliceTailLen);
                    }
                }
                idxBase += batch;
            }
        }
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

    // ============ Unaligned: batch indices + TQueBind RMW per 32B block ============
    __aicore__ inline void ProcessUnaligned()
    {
        uint64_t groupSize = this->gatherDimSize * this->sliceLength;
        uint64_t outGroupSize = this->numIndices * this->sliceLength;

        for (uint64_t outer = 0; outer < this->outerLength; ++outer) {
            uint64_t outerSrcBase = outer * groupSize;
            uint64_t outerDstBase = outer * outGroupSize;

            uint64_t remaining = this->myIndicesNum;
            uint64_t idxBase = this->indicesStart;

            while (remaining > 0) {
                uint64_t batch = remaining;
                if (batch > this->indicesBatch) batch = this->indicesBatch;
                remaining -= batch;

                LocalTensor<int32_t> idxUb = indicesBuf.Get<int32_t>();
                DataCopy(idxUb, indicesGm[idxBase], batch);
                PipeBarrier<PIPE_MTE2>();

                for (uint64_t i = 0; i < batch; ++i) {
                    int32_t gatherIdx = idxUb.GetValue(i);
                    uint64_t srcBase = outerSrcBase + gatherIdx * this->sliceLength;
                    uint64_t dstBase = outerDstBase + (idxBase + i) * this->sliceLength;
                    CopyAlignedRMW(srcBase, dstBase, this->sliceLength);
                }
                idxBase += batch;
            }
        }
    }

    __aicore__ inline void CopyAlignedRMW(uint64_t srcOffset, uint64_t dstOffset, uint64_t len)
    {
        while (len > 0) {
            uint64_t srcAligned = (srcOffset / this->numPerBlock32) * this->numPerBlock32;
            uint64_t dstAligned = (dstOffset / this->numPerBlock32) * this->numPerBlock32;
            uint64_t srcOffInBlock = srcOffset - srcAligned;
            uint64_t dstOffInBlock = dstOffset - dstAligned;
            uint64_t srcRemain = this->numPerBlock32 - srcOffInBlock;
            uint64_t dstRemain = this->numPerBlock32 - dstOffInBlock;
            uint64_t copyNow = len;
            if (srcRemain < copyNow) copyNow = srcRemain;
            if (dstRemain < copyNow) copyNow = dstRemain;

            LocalTensor<T> sBuf = rmwQueue.AllocTensor<T>();
            DataCopy(sBuf, xGm[srcAligned], this->numPerBlock32);
            rmwQueue.EnQue(sBuf);

            LocalTensor<T> dBuf = rmwQueue.AllocTensor<T>();
            DataCopy(dBuf, yGm[dstAligned], this->numPerBlock32);
            rmwQueue.EnQue(dBuf);

            LocalTensor<T> sData = rmwQueue.DeQue<T>();
            LocalTensor<T> dData = rmwQueue.DeQue<T>();

            for (uint64_t j = 0; j < copyNow; ++j) {
                dData.SetValue(dstOffInBlock + j, sData.GetValue(srcOffInBlock + j));
            }
            PipeBarrier<PIPE_MTE3>();

            DataCopy(yGm[dstAligned], dData, this->numPerBlock32);

            rmwQueue.FreeTensor(dData);
            rmwQueue.FreeTensor(sData);

            srcOffset += copyNow;
            dstOffset += copyNow;
            len -= copyNow;
        }
    }

private:
private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<int32_t> indicesGm;
    TPipe* pipe;

    // Aligned path
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> copyQueue;

    // Unaligned path (RMW)
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> rmwQueue;

    // Aligned path
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> copyQueue;

    // Unaligned path (RMW)
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> rmwQueue;
    uint64_t numPerBlock32;

    // Indices buffer
    uint64_t indicesBatch;
    TBuf<QuePosition::VECCALC> indicesBuf;

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
    uint64_t numPerBlock;
    bool sliceAligned;
};

extern "C" __global__ __aicore__ void gather_custom(
    GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelGatherCustom<DTYPE_X> op;
    op.Init(x, indices, y,
            tiling_data.numIndices, tiling_data.sliceLength,
            tiling_data.outerLength, tiling_data.gatherDimSize,
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
