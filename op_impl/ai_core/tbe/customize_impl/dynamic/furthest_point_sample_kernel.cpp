/**
 * FurthestPointSample - AscendC Kernel (op_kernel)
 *
 * Deploy target: Ascend 310P / CANN 8.3RC1
 *
 * Compiled by the CANN op build system.
 * Dispatched through TILING_KEY for dtype selection.
 */

#include "kernel_operator.h"

// ─── Kernel dispatch macro ──────────────────────────────────────────────
#define FPS_OP_IMPL(templateClass, ...)                                  \
    do {                                                                 \
        GET_TILING_DATA(fpsTiling, tiling);                              \
        templateClass<__VA_ARGS__> op;                                   \
        op.Init(points, sampled, workspace,                              \
                fpsTiling.ubPointsNum, fpsTiling.ubMinDistNum,           \
                fpsTiling.B, fpsTiling.N, fpsTiling.M,                   \
                fpsTiling.C, fpsTiling.batchPerCore,                     \
                fpsTiling.coreRemainder, fpsTiling.wsStride,             \
                fpsTiling.initVal);                                      \
        op.Process();                                                    \
    } while (0)

constexpr int32_t BUF_NUM = 2;
constexpr int32_t COORD_DIM = 3;

// ─── Kernel Template ────────────────────────────────────────────────────
template <typename T>
class KernelFPS {
public:
    __aicore__ inline KernelFPS() {}

    __aicore__ inline void Init(
        GM_ADDR pointsGm, GM_ADDR sampledGm, GM_ADDR wsGm,
        uint32_t ubPN, uint32_t ubMDN,
        uint32_t B, uint32_t N, uint32_t M,
        uint32_t C, uint32_t batchPerCore,
        uint32_t coreRemainder, uint32_t wsStride,
        float initVal)
    {
        B_      = static_cast<int32_t>(B);
        N_      = static_cast<int32_t>(N);
        M_      = static_cast<int32_t>(M);
        C_      = static_cast<int32_t>(C);
        tileN_  = static_cast<int32_t>(ubPN);
        initVal_ = static_cast<T>(initVal);
        wsStride_ = static_cast<int32_t>(wsStride);

        inputGm_  = reinterpret_cast<__gm__ T *>(pointsGm);
        outputGm_ = reinterpret_cast<__gm__ T *>(sampledGm);
        wsGm_     = reinterpret_cast<__gm__ T *>(wsGm);

        int32_t totalBlocks = AscendC::GetBlockNum();
        int32_t blockIdx    = AscendC::GetBlockIdx();
        if (static_cast<uint32_t>(blockIdx) < coreRemainder) {
            batchStart_ = blockIdx * static_cast<int32_t>(batchPerCore + 1);
            batchEnd_   = batchStart_ + static_cast<int32_t>(batchPerCore) + 1;
        } else {
            batchStart_ = static_cast<int32_t>(coreRemainder * (batchPerCore + 1) +
                          (blockIdx - static_cast<int32_t>(coreRemainder)) * batchPerCore);
            batchEnd_   = batchStart_ + static_cast<int32_t>(batchPerCore);
        }
        if (batchStart_ >= B_) batchEnd_ = batchStart_;
        if (batchEnd_   >  B_) batchEnd_ = B_;
    }

    __aicore__ inline void Process() {
        for (int32_t b = batchStart_; b < batchEnd_; ++b) {
            ProcessBatch(b);
        }
    }

private:
    __aicore__ inline void ProcessBatch(int32_t batchIdx) {
        __gm__ T *batchIn  = inputGm_  + batchIdx * N_ * C_;
        __gm__ T *batchOut = outputGm_ + batchIdx * M_ * C_;
        __gm__ T *batchWs  = wsGm_     + batchIdx * wsStride_;

        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN,  BUF_NUM> qX, qY, qZ;
        AscendC::TQue<AscendC::QuePosition::VECIN,  BUF_NUM> qMd;
        AscendC::TQue<AscendC::QuePosition::VECOUT, BUF_NUM> qMdOut;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> bufDist;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> bufTmp;

        pipe.InitBuffer(qX,     BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qY,     BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qZ,     BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qMd,    BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qMdOut, BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(bufDist, tileN_ * sizeof(T));
        pipe.InitBuffer(bufTmp,  tileN_ * sizeof(T));

        T selX = batchIn[0], selY = batchIn[1], selZ = batchIn[2];
        batchOut[0] = selX; batchOut[1] = selY; batchOut[2] = selZ;
        uint32_t selIdx = 0;
        int32_t numTiles = (N_ + tileN_ - 1) / tileN_;

        for (int32_t m = 1; m < M_; ++m) {
            T globalMax     = static_cast<T>(-65504.0);
            uint32_t globalIdx = 0;

            for (int32_t t = 0; t < numTiles; ++t) {
                int32_t tStart = t * tileN_;
                int32_t curN   = (tStart + tileN_ <= N_) ? tileN_ : (N_ - tStart);

                AscendC::LocalTensor<T> xLoc = qX.AllocTensor<T>();
                AscendC::LocalTensor<T> yLoc = qY.AllocTensor<T>();
                AscendC::LocalTensor<T> zLoc = qZ.AllocTensor<T>();
                AscendC::LocalTensor<T> mdIn = qMd.AllocTensor<T>();

                for (int32_t i = 0; i < curN; ++i) {
                    int32_t gi = tStart + i;
                    xLoc.SetValue(i, batchIn[gi * C_ + 0]);
                    yLoc.SetValue(i, batchIn[gi * C_ + 1]);
                    zLoc.SetValue(i, batchIn[gi * C_ + 2]);
                }
                if (m == 1) {
                    for (int32_t i = 0; i < curN; ++i)
                        mdIn.SetValue(i, initVal_);
                } else {
                    for (int32_t i = 0; i < curN; ++i)
                        mdIn.SetValue(i, batchWs[tStart + i]);
                }

                qX.EnQue(xLoc);
                qY.EnQue(yLoc);
                qZ.EnQue(zLoc);
                qMd.EnQue(mdIn);

                AscendC::LocalTensor<T> xTile = qX.DeQue<T>();
                AscendC::LocalTensor<T> yTile = qY.DeQue<T>();
                AscendC::LocalTensor<T> zTile = qZ.DeQue<T>();
                AscendC::LocalTensor<T> mdVal = qMd.DeQue<T>();
                AscendC::LocalTensor<T> dist  = bufDist.Get<T>();
                AscendC::LocalTensor<T> tmp   = bufTmp.Get<T>();

                AscendC::Subs(dist, xTile, selX, curN);
                AscendC::Mul(dist, dist, dist, curN);
                AscendC::Subs(tmp, yTile, selY, curN);
                AscendC::Mul(tmp, tmp, tmp, curN);
                AscendC::Add(dist, dist, tmp, curN);
                AscendC::Subs(tmp, zTile, selZ, curN);
                AscendC::Mul(tmp, tmp, tmp, curN);
                AscendC::Add(dist, dist, tmp, curN);
                AscendC::Min(mdVal, mdVal, dist, curN);

                AscendC::LocalTensor<T> mdOut = qMdOut.AllocTensor<T>();
                for (int32_t i = 0; i < curN; ++i)
                    mdOut.SetValue(i, mdVal.GetValue(i));
                qMdOut.EnQue<T>(mdOut);

                T localMax     = static_cast<T>(-65504.0);
                uint32_t localIdx = 0;
                for (int32_t i = 0; i < curN; ++i) {
                    T v = mdVal.GetValue(i);
                    if (v > localMax) { localMax = v; localIdx = i; }
                }
                if (localMax > globalMax) {
                    globalMax  = localMax;
                    globalIdx  = tStart + localIdx;
                }

                qX.FreeTensor(xTile);
                qY.FreeTensor(yTile);
                qZ.FreeTensor(zTile);
                qMd.FreeTensor(mdVal);

                AscendC::LocalTensor<T> mdFlush = qMdOut.DeQue<T>();
                for (int32_t i = 0; i < curN; ++i)
                    batchWs[tStart + i] = mdFlush.GetValue(i);
                qMdOut.FreeTensor(mdFlush);
            }

            selIdx = globalIdx;
            int32_t off = selIdx * C_;
            selX = batchIn[off + 0];
            selY = batchIn[off + 1];
            selZ = batchIn[off + 2];
            batchOut[m * C_ + 0] = selX;
            batchOut[m * C_ + 1] = selY;
            batchOut[m * C_ + 2] = selZ;
        }
    }

    int32_t   B_, N_, M_, C_, tileN_, wsStride_;
    T         initVal_;
    int32_t   batchStart_, batchEnd_;
    __gm__ T *inputGm_;
    __gm__ T *outputGm_;
    __gm__ T *wsGm_;
};

// ─── Kernel Entry Point ─────────────────────────────────────────────────
extern "C" __global__ __aicore__ void furthest_point_sample(
    GM_ADDR points, GM_ADDR sampled, GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(0)) {
        FPS_OP_IMPL(KernelFPS, float);
    } else if (TILING_KEY_IS(1)) {
        FPS_OP_IMPL(KernelFPS, half);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void furthest_point_sample_do(uint32_t blockDim, void *l2ctrl, void *stream,
                               uint8_t *points, uint8_t *sampled,
                               uint8_t *workspace, uint8_t *tiling)
{
    furthest_point_sample<<<blockDim, l2ctrl, stream>>>(
        points, sampled, workspace, tiling);
}
#endif
