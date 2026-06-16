/**
 * FurthestPointSample - AscendC Kernel (compiled by Python build script)
 *
 * This file is compiled via furthest_point_sample.py
 * and dispatched through TILING_KEY for dtype selection.
 *
 * Tiling data is provided by op_host/fps_tiling.h via GET_TILING_DATA macro.
 */

#include "kernel_operator.h"
#include "fps_tiling_data.h"

constexpr int32_t BUF_NUM = 2;
constexpr int32_t COORD_DIM = 3;

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
                fpsTiling.initVal);                                       \
        op.Process();                                                    \
    } while (0)

// ─── Kernel Template ──────────────────────────────────────────────────────────
template <typename T>
class KernelFurthestPointSample {
public:
    __aicore__ inline KernelFurthestPointSample() {}

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
        wsGm_     = reinterpret_cast<uint8_t *>(wsGm);

        // Multi-core: each core handles a subset of batches
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
    // Compute squared L2 distance for a tile
    //   dst[i] = (x[i]-sx)^2 + (y[i]-sy)^2 + (z[i]-sz)^2
    __aicore__ inline void ComputeSqDistTile(
        LocalTensor<T> &dst, const LocalTensor<T> &x,
        const LocalTensor<T> &y, const LocalTensor<T> &z,
        T sx, T sy, T sz, int32_t count)
    {
        // dst = (x - sx)^2
        Subs(dst, x, sx, count);
        Mul(dst, dst, dst, count);
    }

    __aicore__ inline void ProcessBatch(int32_t batchIdx) {
        __gm__ T *batchIn  = inputGm_  + batchIdx * N_ * C_;
        __gm__ T *batchOut = outputGm_ + batchIdx * M_ * C_;
        __gm__ T *batchWs  = reinterpret_cast<__gm__ T *>(
            wsGm_ + batchIdx * wsStride_);

        // ── Setup data pipeline ──
        TPipe pipe;
        TQue<QuePosition::VECIN,  BUF_NUM> qX, qY, qZ;
        TQue<QuePosition::VECIN,  BUF_NUM> qMd;   // minDist load from GM
        TQue<QuePosition::VECOUT, BUF_NUM> qMdOut; // minDist store to GM
        TBuf<QuePosition::VECCALC> bufDist;
        TBuf<QuePosition::VECCALC> bufTmp;

        pipe.InitBuffer(qX,     BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qY,     BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qZ,     BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qMd,    BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qMdOut, BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(bufDist, tileN_ * sizeof(T));
        pipe.InitBuffer(bufTmp,  tileN_ * sizeof(T));

        // ── First point (index 0) ──
        T selPt[3] = { batchIn[0], batchIn[1], batchIn[2] };
        batchOut[0] = selPt[0];
        batchOut[1] = selPt[1];
        batchOut[2] = selPt[2];
        uint32_t selIdx = 0;

        // ── Main FPS loop ──
        int32_t numTiles = (N_ + tileN_ - 1) / tileN_;

        for (int32_t m = 1; m < M_; ++m) {
            T globalBest   = static_cast<T>(-65504.0);
            uint32_t globalIdx = 0;

            for (int32_t t = 0; t < numTiles; ++t) {
                int32_t tStart = t * tileN_;
                int32_t curN   = (tStart + tileN_ <= N_) ? tileN_ : (N_ - tStart);

                // ── Load data ──
                LocalTensor<T> xLoc = qX.AllocTensor<T>();
                LocalTensor<T> yLoc = qY.AllocTensor<T>();
                LocalTensor<T> zLoc = qZ.AllocTensor<T>();
                LocalTensor<T> mdIn = qMd.AllocTensor<T>();

                // Load point coordinates (element-wise for correctness)
                for (int32_t i = 0; i < curN; ++i) {
                    int32_t gi = tStart + i;
                    xLoc.SetValue(i, batchIn[gi * C_ + 0]);
                    yLoc.SetValue(i, batchIn[gi * C_ + 1]);
                    zLoc.SetValue(i, batchIn[gi * C_ + 2]);
                }

                // Load previous minDist from workspace (or init if m==1)
                if (m == 1) {
                    // First iteration: init with large values
                    for (int32_t i = 0; i < curN; ++i) {
                        mdIn.SetValue(i, static_cast<T>(65504.0));
                    }
                } else {
                    for (int32_t i = 0; i < curN; ++i) {
                        mdIn.SetValue(i, batchWs[tStart + i]);
                    }
                }

                qX.EnQue(xLoc);
                qY.EnQue(yLoc);
                qZ.EnQue(zLoc);
                qMd.EnQue(mdIn);

                // ── Compute ──
                LocalTensor<T> xTile = qX.DeQue<T>();
                LocalTensor<T> yTile = qY.DeQue<T>();
                LocalTensor<T> zTile = qZ.DeQue<T>();
                LocalTensor<T> mdVal = qMd.DeQue<T>();
                LocalTensor<T> dist  = bufDist.Get<T>();
                LocalTensor<T> tmp   = bufTmp.Get<T>();

                // Compute squared distances to selected point
                // dist[i] = (x_i - selPt[0])^2 + (y_i - selPt[1])^2 + (z_i - selPt[2])^2
                ComputeSqDistTile(dist, xTile, yTile, zTile,
                                   selPt[0], selPt[1], selPt[2], curN);
                // Actually need to do three accumulations - use tmp for Y and Z
                Subs(dist, xTile, selPt[0], curN);    // dist = x - sx
                Mul(dist, dist, dist, curN);            // dist = (x-sx)^2
                Subs(tmp, yTile, selPt[1], curN);      // tmp = y - sy
                Mul(tmp, tmp, tmp, curN);               // tmp = (y-sy)^2
                Add(dist, dist, tmp, curN);             // dist += (y-sy)^2
                Subs(tmp, zTile, selPt[2], curN);      // tmp = z - sz
                Mul(tmp, tmp, tmp, curN);               // tmp = (z-sz)^2
                Add(dist, dist, tmp, curN);             // dist += (z-sz)^2

                // Update minDist: mdVal[i] = min(mdVal[i], dist[i])
                Min(mdVal, mdVal, dist, curN);

                // Write updated minDist back to workspace and enqueue for output
                LocalTensor<T> mdOut = qMdOut.AllocTensor<T>();
                for (int32_t i = 0; i < curN; ++i) {
                    mdOut.SetValue(i, mdVal.GetValue(i));
                }
                qMdOut.EnQue<T>(mdOut);

                // Find local argmax
                T localBest    = static_cast<T>(-65504.0);
                uint32_t localIdx = 0;
                for (int32_t i = 0; i < curN; ++i) {
                    T v = mdVal.GetValue(i);
                    if (v > localBest) { localBest = v; localIdx = i; }
                }
                if (localBest > globalBest) {
                    globalBest = localBest;
                    globalIdx  = tStart + localIdx;
                }

                qX.FreeTensor(xTile);
                qY.FreeTensor(yTile);
                qZ.FreeTensor(zTile);
                qMd.FreeTensor(mdVal);

                // Flush minDist back to GM workspace
                LocalTensor<T> mdFlush = qMdOut.DeQue<T>();
                for (int32_t i = 0; i < curN; ++i) {
                    batchWs[tStart + i] = mdFlush.GetValue(i);
                }
                qMdOut.FreeTensor(mdFlush);
            }

            selIdx = globalIdx;
            int32_t off = selIdx * C_;
            selPt[0] = batchIn[off + 0];
            selPt[1] = batchIn[off + 1];
            selPt[2] = batchIn[off + 2];

            batchOut[m * C_ + 0] = selPt[0];
            batchOut[m * C_ + 1] = selPt[1];
            batchOut[m * C_ + 2] = selPt[2];
        }
    }

    int32_t   B_, N_, M_, C_;
    int32_t   tileN_, wsStride_;
    T         initVal_;
    int32_t   batchStart_, batchEnd_;
    __gm__ T *inputGm_;
    __gm__ T *outputGm_;
    uint8_t  *wsGm_;
};

// ─── Kernel Entry Point ───────────────────────────────────────────────────────
extern "C" __global__ __aicore__ void furthest_point_sample(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    uint64_t tilingKey = *reinterpret_cast<const uint64_t *>(
        reinterpret_cast<const uint8_t *>(tiling));

    if (tilingKey == 0) {
        // DTYPE_X = float, DTYPE_Y = float, DTYPE_Z = float
        FPS_OP_IMPL(KernelFurthestPointSample, float);
    } else if (tilingKey == 1) {
        // DTYPE_X = half, DTYPE_Y = half, DTYPE_Z = half
        FPS_OP_IMPL(KernelFurthestPointSample, half);
    }
}

#ifndef ASCENDC_CPU_DEBUG
// Host-side kernel launch function (called by tiling framework)
void furthest_point_sample_do(uint32_t blockDim, void *l2ctrl, void *stream,
                               uint8_t *input, uint8_t *output,
                               uint8_t *workspace, uint8_t *tiling)
{
    furthest_point_sample<<<blockDim, l2ctrl, stream>>>(input, output, workspace, tiling);
}
#endif
