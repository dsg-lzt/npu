/**
 * FurthestPointSample - AscendC Kernel (op_kernel)
 *
 * Input:  points [B,N,3] float32|float16,  temp [B,N] float32
 * Output: sampled [B,M] INT32 indices
 *
 * temp tensor used as minDist workspace via DataCopy (MTE, reliable).
 */

#include "kernel_operator.h"

#define FPS_OP_IMPL(templateClass, ...)                                  \
    do {                                                                 \
        GET_TILING_DATA(fpsTiling, tiling);                              \
        templateClass<__VA_ARGS__> op;                                   \
        op.Init(points, temp, sampled,                                   \
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
        GM_ADDR pointsGm, GM_ADDR tempGm, GM_ADDR sampledGm,
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
        inputGm_  = reinterpret_cast<__gm__ T *>(pointsGm);
        tempGm_   = reinterpret_cast<__gm__ float *>(tempGm);
        outputGm_ = reinterpret_cast<__gm__ int32_t *>(sampledGm);

        int32_t blockIdx = AscendC::GetBlockIdx();
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
        __gm__ float   *batchTemp = tempGm_   + batchIdx * N_;
        __gm__ int32_t *batchOut  = outputGm_ + batchIdx * M_;

        int32_t numTiles = (N_ + tileN_ - 1) / tileN_;

        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN,  BUF_NUM> qX, qY, qZ;
        AscendC::TQue<AscendC::QuePosition::VECIN,  BUF_NUM> qMdIn;
        AscendC::TQue<AscendC::QuePosition::VECOUT, BUF_NUM> qMdOut;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> bufDist;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> bufTmp;  // float, unused now

        pipe.InitBuffer(qX,      BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qY,      BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qZ,      BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qMdIn,   BUF_NUM, tileN_ * sizeof(float));
        pipe.InitBuffer(qMdOut,  BUF_NUM, tileN_ * sizeof(float));
        pipe.InitBuffer(bufDist, tileN_ * sizeof(float));

        // GlobalTensor for temp access (member-like, survives the loop)
        AscendC::GlobalTensor<float> tempGt;

        // Init temp to large values (reuse qMdOut for init write)
        {
            AscendC::LocalTensor<float> mdInit = qMdOut.AllocTensor<float>();
            AscendC::Duplicate(mdInit, static_cast<float>(initVal_), tileN_);
            qMdOut.EnQue<float>(mdInit);
            AscendC::LocalTensor<float> mdFlush = qMdOut.DeQue<float>();
            for (int32_t t = 0; t < numTiles; ++t) {
                int32_t tStart = t * tileN_;
                int32_t curN   = (tStart + tileN_ <= N_) ? tileN_ : (N_ - tStart);
                tempGt.SetGlobalBuffer(batchTemp + tStart, curN);
                AscendC::DataCopy(tempGt, mdFlush, curN);
            }
            qMdOut.FreeTensor(mdFlush);
        }

        T selX = batchIn[0], selY = batchIn[1], selZ = batchIn[2];
        batchOut[0] = 0;
        uint32_t selIdx = 0;

        AscendC::PRINTF("[FPS] B=%d N=%d M=%d tN=%d tiles=%d\n",
               static_cast<int32_t>(batchIdx), N_, M_, tileN_, numTiles);

        AscendC::GlobalTensor<float> tempRd, tempWr;

        for (int32_t m = 1; m < M_; ++m) {
            float globalMaxF = -1e30f;
            uint32_t globalIdx = 0;

            for (int32_t t = 0; t < numTiles; ++t) {
                int32_t tStart = t * tileN_;
                int32_t curN   = (tStart + tileN_ <= N_) ? tileN_ : (N_ - tStart);

                AscendC::LocalTensor<T>     xLoc = qX.AllocTensor<T>();
                AscendC::LocalTensor<T>     yLoc = qY.AllocTensor<T>();
                AscendC::LocalTensor<T>     zLoc = qZ.AllocTensor<T>();
                AscendC::LocalTensor<float> mdIn = qMdIn.AllocTensor<float>();

                // Load coords
                for (int32_t i = 0; i < curN; ++i) {
                    int32_t gi = tStart + i;
                    xLoc.SetValue(i, batchIn[gi * C_ + 0]);
                    yLoc.SetValue(i, batchIn[gi * C_ + 1]);
                    zLoc.SetValue(i, batchIn[gi * C_ + 2]);
                }
                // Load minDist from temp via DataCopy
                tempRd.SetGlobalBuffer(batchTemp + tStart, curN);
                AscendC::DataCopy(mdIn, tempRd, curN);

                qX.EnQue(xLoc);
                qY.EnQue(yLoc);
                qZ.EnQue(zLoc);
                qMdIn.EnQue(mdIn);

                AscendC::LocalTensor<T>     xTile = qX.DeQue<T>();
                AscendC::LocalTensor<T>     yTile = qY.DeQue<T>();
                AscendC::LocalTensor<T>     zTile = qZ.DeQue<T>();
                AscendC::LocalTensor<float> mdVal = qMdIn.DeQue<float>();
                AscendC::LocalTensor<float> dist  = bufDist.Get<float>();

                // dist[i] = ||pts[i] - selPt||²
                for (int32_t i = 0; i < curN; ++i) {
                    float dx = static_cast<float>(xTile.GetValue(i)) - static_cast<float>(selX);
                    float dy = static_cast<float>(yTile.GetValue(i)) - static_cast<float>(selY);
                    float dz = static_cast<float>(zTile.GetValue(i)) - static_cast<float>(selZ);
                    dist.SetValue(i, dx * dx + dy * dy + dz * dz);
                }

                // mdVal = min(mdVal, dist)
                for (int32_t i = 0; i < curN; ++i) {
                    float d = dist.GetValue(i);
                    if (d < mdVal.GetValue(i)) mdVal.SetValue(i, d);
                }

                // Save minDist back to temp via DataCopy
                AscendC::LocalTensor<float> mdOut = qMdOut.AllocTensor<float>();
                for (int32_t i = 0; i < curN; ++i) mdOut.SetValue(i, mdVal.GetValue(i));
                qMdOut.EnQue<float>(mdOut);

                // Find local argmax
                float localMaxF = -1e30f;
                uint32_t localIdx = 0;
                for (int32_t i = 0; i < curN; ++i) {
                    float v = mdVal.GetValue(i);
                    if (v > localMaxF) { localMaxF = v; localIdx = i; }
                }
                if (localMaxF > globalMaxF) {
                    globalMaxF = localMaxF;
                    globalIdx  = tStart + localIdx;
                }

                qX.FreeTensor(xTile);
                qY.FreeTensor(yTile);
                qZ.FreeTensor(zTile);
                qMdIn.FreeTensor(mdVal);

                AscendC::LocalTensor<float> mdFlush = qMdOut.DeQue<float>();
                tempWr.SetGlobalBuffer(batchTemp + tStart, curN);
                AscendC::DataCopy(tempWr, mdFlush, curN);
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

    int32_t       B_, N_, M_, C_, tileN_;
    T             initVal_;
    int32_t       batchStart_, batchEnd_;
    __gm__ T       *inputGm_;
    __gm__ float   *tempGm_;
    __gm__ int32_t *outputGm_;
};

extern "C" __global__ __aicore__ void furthest_point_sample(
    GM_ADDR points, GM_ADDR temp, GM_ADDR sampled, GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(0)) {
        FPS_OP_IMPL(KernelFPS, float);
    } else if (TILING_KEY_IS(1)) {
        FPS_OP_IMPL(KernelFPS, half);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void furthest_point_sample_do(uint32_t blockDim, void *l2ctrl, void *stream,
                               uint8_t *points, uint8_t *temp, uint8_t *sampled,
                               uint8_t *workspace, uint8_t *tiling)
{
    furthest_point_sample<<<blockDim, l2ctrl, stream>>>(
        points, temp, sampled, workspace, tiling);
}
#endif
