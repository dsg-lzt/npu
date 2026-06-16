/**
 * FurthestPointSample - AscendC Kernel (op_kernel)
 *
 * Input:  points  (B, N, 3)  float32|float16
 * Output: sampled (B, M)     INT32 indices
 */

#include "kernel_operator.h"

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
        mdN_    = static_cast<int32_t>(ubMDN);
        initVal_ = static_cast<T>(initVal);
        wsStride_ = static_cast<int32_t>(wsStride);

        inputGm_  = reinterpret_cast<__gm__ T *>(pointsGm);
        outputGm_ = reinterpret_cast<__gm__ int32_t *>(sampledGm);
        mdWsGm_   = reinterpret_cast<__gm__ T *>(wsGm);

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
        __gm__ T       *batchIn   = inputGm_  + batchIdx * N_ * C_;
        __gm__ int32_t *batchOut  = outputGm_ + batchIdx * M_;
        __gm__ T       *batchMdWs = mdWsGm_   + batchIdx * wsStride_;

        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN,  BUF_NUM> qX, qY, qZ;
        AscendC::TQue<AscendC::QuePosition::VECIN,  BUF_NUM> qMdIn;
        AscendC::TQue<AscendC::QuePosition::VECOUT, BUF_NUM> qMdOut;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> mdBuf;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> bufDist;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> bufTmp;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> bufSca;

        pipe.InitBuffer(qX,     BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qY,     BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qZ,     BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qMdIn,  BUF_NUM, mdN_ * sizeof(T));
        pipe.InitBuffer(qMdOut, BUF_NUM, mdN_ * sizeof(T));
        pipe.InitBuffer(mdBuf,    mdN_ * sizeof(T));
        pipe.InitBuffer(bufDist,  tileN_ * sizeof(T));
        pipe.InitBuffer(bufTmp,   tileN_ * sizeof(T));
        pipe.InitBuffer(bufSca,   tileN_ * sizeof(T));

        AscendC::LocalTensor<T> mdAll = mdBuf.Get<T>();

        // Init minDist in GM workspace via DataCopy
        int32_t numTiles = (N_ + tileN_ - 1) / tileN_;
        {
            AscendC::LocalTensor<T> mdInit = qMdOut.AllocTensor<T>();
            AscendC::Duplicate(mdInit, initVal_, tileN_);
            qMdOut.EnQue<T>(mdInit);
            AscendC::LocalTensor<T> mdFlush = qMdOut.DeQue<T>();
            for (int32_t t = 0; t < numTiles; ++t) {
                int32_t tStart = t * tileN_;
                int32_t curN   = (tStart + tileN_ <= N_) ? tileN_ : (N_ - tStart);
                AscendC::GlobalTensor<T> wsGt;
                wsGt.SetGlobalBuffer(batchMdWs + tStart, curN);
                AscendC::DataCopy(wsGt, mdFlush, curN);
            }
            qMdOut.FreeTensor(mdFlush);
        }

        T selX = batchIn[0], selY = batchIn[1], selZ = batchIn[2];
        batchOut[0] = 0;
        uint32_t selIdx = 0;

        AscendC::PRINTF("[FPS] B=%d N=%d M=%d tileN=%d\n",
               static_cast<int32_t>(batchIdx), N_, M_, tileN_);

        for (int32_t m = 1; m < M_; ++m) {
            T globalMax     = static_cast<T>(-65504.0);
            uint32_t globalIdx = 0;

            for (int32_t t = 0; t < numTiles; ++t) {
                int32_t tStart = t * tileN_;
                int32_t curN   = (tStart + tileN_ <= N_) ? tileN_ : (N_ - tStart);

                AscendC::LocalTensor<T> xLoc = qX.AllocTensor<T>();
                AscendC::LocalTensor<T> yLoc = qY.AllocTensor<T>();
                AscendC::LocalTensor<T> zLoc = qZ.AllocTensor<T>();
                AscendC::LocalTensor<T> mdIn = qMdIn.AllocTensor<T>();

                for (int32_t i = 0; i < curN; ++i) {
                    int32_t gi = tStart + i;
                    xLoc.SetValue(i, batchIn[gi * C_ + 0]);
                    yLoc.SetValue(i, batchIn[gi * C_ + 1]);
                    zLoc.SetValue(i, batchIn[gi * C_ + 2]);
                }
                // Read minDist from GM workspace via DataCopy (MTE, ordered)
                {
                    AscendC::GlobalTensor<T> wsGt;
                    wsGt.SetGlobalBuffer(batchMdWs + tStart, curN);
                    AscendC::DataCopy(mdIn, wsGt, curN);
                }

                qX.EnQue(xLoc);
                qY.EnQue(yLoc);
                qZ.EnQue(zLoc);
                qMdIn.EnQue(mdIn);

                AscendC::LocalTensor<T> xTile = qX.DeQue<T>();
                AscendC::LocalTensor<T> yTile = qY.DeQue<T>();
                AscendC::LocalTensor<T> zTile = qZ.DeQue<T>();
                AscendC::LocalTensor<T> mdVal = qMdIn.DeQue<T>();
                AscendC::LocalTensor<T> dist  = bufDist.Get<T>();
                AscendC::LocalTensor<T> tmp   = bufTmp.Get<T>();
                AscendC::LocalTensor<T> sca   = bufSca.Get<T>();

                AscendC::Duplicate(sca, selX, curN);
                AscendC::Sub(dist, xTile, sca, curN);
                AscendC::Mul(dist, dist, dist, curN);
                AscendC::Duplicate(sca, selY, curN);
                AscendC::Sub(tmp, yTile, sca, curN);
                AscendC::Mul(tmp, tmp, tmp, curN);
                AscendC::Add(dist, dist, tmp, curN);
                AscendC::Duplicate(sca, selZ, curN);
                AscendC::Sub(tmp, zTile, sca, curN);
                AscendC::Mul(tmp, tmp, tmp, curN);
                AscendC::Add(dist, dist, tmp, curN);

                AscendC::Min(mdVal, mdVal, dist, curN);

                if (m <= 2 && t == 0) {
                    AscendC::PRINTF("[FPS] m=%d t=%d d0=%d d1=%d md0=%d md1=%d\n",
                           m, t,
                           static_cast<int32_t>(static_cast<float>(dist.GetValue(0)) * 1000.0f),
                           static_cast<int32_t>(static_cast<float>(dist.GetValue(1)) * 1000.0f),
                           static_cast<int32_t>(static_cast<float>(mdVal.GetValue(0)) * 1000.0f),
                           static_cast<int32_t>(static_cast<float>(mdVal.GetValue(1)) * 1000.0f));
                }

                AscendC::LocalTensor<T> mdOut = qMdOut.AllocTensor<T>();
                for (int32_t i = 0; i < curN; ++i) mdOut.SetValue(i, mdVal.GetValue(i));
                qMdOut.EnQue<T>(mdOut);

                float localMaxF = -65504.0f;
                T localMax = static_cast<T>(-65504.0);
                uint32_t localIdx = 0;
                for (int32_t i = 0; i < curN; ++i) {
                    T v = mdVal.GetValue(i);
                    float fv = static_cast<float>(v);
                    if (fv > localMaxF) { localMaxF = fv; localMax = v; localIdx = i; }
                }
                if (localMaxF > static_cast<float>(globalMax)) {
                    globalMax  = localMax;
                    globalIdx  = tStart + localIdx;
                }

                qX.FreeTensor(xTile);
                qY.FreeTensor(yTile);
                qZ.FreeTensor(zTile);
                qMdIn.FreeTensor(mdVal);

                AscendC::LocalTensor<T> mdFlush = qMdOut.DeQue<T>();
                // Write minDist to GM workspace via DataCopy (MTE, ordered)
                {
                    AscendC::GlobalTensor<T> wsGt;
                    wsGt.SetGlobalBuffer(batchMdWs + tStart, curN);
                    AscendC::DataCopy(wsGt, mdFlush, curN);
                }
                qMdOut.FreeTensor(mdFlush);
            }

            selIdx = globalIdx;
            int32_t off = selIdx * C_;
            selX = batchIn[off + 0];
            selY = batchIn[off + 1];
            selZ = batchIn[off + 2];
            batchOut[m] = static_cast<int32_t>(selIdx);

            AscendC::PRINTF("[FPS] m=%d sel=%d\n", m, static_cast<int32_t>(selIdx));
        }
    }

    int32_t       B_, N_, M_, C_, tileN_, mdN_, wsStride_;
    T             initVal_;
    int32_t       batchStart_, batchEnd_;
    __gm__ T       *inputGm_;
    __gm__ int32_t *outputGm_;
    __gm__ T       *mdWsGm_;
};

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
