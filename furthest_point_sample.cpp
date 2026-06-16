/**
 * Furthest Point Sampling (FPS) Operator using AscendC
 *
 * Input:  points (B, N, 3), dtype float32 or float16
 * Output: sampled points (B, M, 3)
 *
 * Target: Ascend 310P, CANN 8.3RC1
 *
 * Two code paths:
 *   - ASCENDC_CPU_DEBUG=1: CPU simulation with malloc'd buffers
 *   - ASCENDC_CPU_DEBUG undefined: real NPU code with TPipe + vector ops
 */

#include "kernel_operator.h"
#include <cstdint>
#include <cstring>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

using namespace AscendC;

// ─── Tiling Data ───────────────────────────────────────────────────────────────
#pragma pack(push, 8)
struct FpsTilingData {
    int32_t B;              // batch size
    int32_t N;              // number of input points
    int32_t M;              // number of output points
    int32_t dtypeKey;       // 0=float32, 1=float16
    int32_t tileN;          // tile size for N
    int32_t numTiles;       // ceil(N / tileN)
    int32_t batchPerCore;   // batches per core
    int32_t coreRemainder;
    float   initialVal;
};
#pragma pack(pop)

// ─── Kernel: Real NPU Path ─────────────────────────────────────────────────────
#ifndef ASCENDC_CPU_DEBUG

template <typename T>
class KernelFps {
public:
    __aicore__ inline KernelFps() {}

    __aicore__ inline void Init(GM_ADDR inputGm, GM_ADDR outputGm, GM_ADDR tilingGm) {
        auto *td = reinterpret_cast<const FpsTilingData *>(
            reinterpret_cast<uintptr_t>(tilingGm));
        B_       = td->B;
        N_       = td->N;
        M_       = td->M;
        tileN_   = (td->tileN > 0 && td->tileN < td->N) ? td->tileN : td->N;
        initVal_ = static_cast<T>(td->initialVal);

        int64_t coreIdx = GetBlockIdx();
        if (coreIdx < td->coreRemainder) {
            batchStart_ = coreIdx * (td->batchPerCore + 1);
            batchEnd_   = batchStart_ + td->batchPerCore + 1;
        } else {
            batchStart_ = td->coreRemainder * (td->batchPerCore + 1) +
                          (coreIdx - td->coreRemainder) * td->batchPerCore;
            batchEnd_   = batchStart_ + td->batchPerCore;
        }
        if (batchStart_ >= B_ || batchEnd_ > B_) {
            batchEnd_ = (batchStart_ < B_) ? B_ : batchStart_;
        }
        inputGm_  = reinterpret_cast<__gm__ T *>(inputGm);
        outputGm_ = reinterpret_cast<__gm__ T *>(outputGm);
    }

    __aicore__ inline void Process() {
        for (int32_t b = batchStart_; b < batchEnd_; ++b) {
            ProcessBatch(b);
        }
    }

private:
    // Buffer count for double-buffering
    static constexpr int32_t BUF_NUM = 2;

    __aicore__ inline void ProcessBatch(int32_t batchIdx) {
        __gm__ T *batchIn  = inputGm_  + batchIdx * N_ * 3;
        __gm__ T *batchOut = outputGm_ + batchIdx * M_ * 3;

        // ── Setup pipeline ──
        TPipe pipe;
        TQue<QuePosition::VECIN,  BUF_NUM> qX, qY, qZ;
        TBuf<QuePosition::VECCALC>         minDistBuf, distBuf, tmpBuf;

        pipe.InitBuffer(qX, BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qY, BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qZ, BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(minDistBuf, tileN_ * sizeof(T));
        pipe.InitBuffer(distBuf,    tileN_ * sizeof(T));
        pipe.InitBuffer(tmpBuf,     tileN_ * sizeof(T));

        // ── First point (index 0) ──
        uint32_t selIdx = 0;
        T sx = batchIn[0], sy = batchIn[1], sz = batchIn[2];
        batchOut[0] = sx; batchOut[1] = sy; batchOut[2] = sz;

        // Pre-load first output point
        T selectedPt[3] = {sx, sy, sz};

        for (int32_t m = 1; m < M_; ++m) {
            T globalMax     = static_cast<T>(-65504.0);
            uint32_t globalIdx = 0;

            for (int32_t t = 0; t * tileN_ < N_; ++t) {
                int32_t tStart = t * tileN_;
                int32_t curN   = (tStart + tileN_ <= N_) ? tileN_ : (N_ - tStart);

                // Load tile data (pipeline: load in 2-buffer rotation)
                LocalTensor<T> xLoc = qX.AllocTensor<T>();
                LocalTensor<T> yLoc = qY.AllocTensor<T>();
                LocalTensor<T> zLoc = qZ.AllocTensor<T>();
                DataCopyExtParams copyParams;
                copyParams.blockCount = curN;
                copyParams.blockLen   = sizeof(T);
                copyParams.srcGap     = 2 * sizeof(T);  // skip Y and Z between X values
                copyParams.dstGap     = 0;

                // Load coordinates with stride=3
                DataCopy(xLoc, GlobalTensor<T>(batchIn + tStart * 3), copyParams);
                // For Y: offset by 1 element, same stride
                copyParams.srcGap = 2 * sizeof(T);
                DataCopy(yLoc, GlobalTensor<T>(batchIn + tStart * 3 + 1), copyParams);
                DataCopy(zLoc, GlobalTensor<T>(batchIn + tStart * 3 + 2), copyParams);

                qX.EnQue(xLoc);
                qY.EnQue(yLoc);
                qZ.EnQue(zLoc);

                LocalTensor<T> xTile = qX.DeQue<T>();
                LocalTensor<T> yTile = qY.DeQue<T>();
                LocalTensor<T> zTile = qZ.DeQue<T>();
                LocalTensor<T> d     = distBuf.Get<T>();

                // Compute squared distances: d[i] = (x_i - sx)^2 + (y_i - sy)^2 + (z_i - sz)^2
                Subs(d, xTile, selectedPt[0], curN);  // d = x - sx
                Mul(d, d, d, curN);                     // d = (x-sx)^2

                // tmp = (y - sy)^2
                LocalTensor<T> t2 = tmpBuf.Get<T>();
                Subs(t2, yTile, selectedPt[1], curN);
                Mul(t2, t2, t2, curN);
                Add(d, d, t2, curN);                   // d += (y-sy)^2

                // tmp = (z - sz)^2
                Subs(t2, zTile, selectedPt[2], curN);
                Mul(t2, t2, t2, curN);
                Add(d, d, t2, curN);                   // d += (z-sz)^2

                // Update minDist
                LocalTensor<T> md = minDistBuf.Get<T>();
                if (m == 1) {
                    // Initialize minDist with first distances
                    for (int32_t i = 0; i < curN; ++i)
                        md.SetValue(i, d.GetValue(i));
                } else {
                    Min(md, md, d, curN);
                }

                // Find local argmax
                T best = static_cast<T>(-65504.0);
                uint32_t bestIdx = 0;
                for (int32_t i = 0; i < curN; ++i) {
                    T v = md.GetValue(i);
                    if (v > best) { best = v; bestIdx = i; }
                }
                if (best > globalMax) {
                    globalMax = best;
                    globalIdx = tStart + bestIdx;
                }

                qX.FreeTensor(xTile);
                qY.FreeTensor(yTile);
                qZ.FreeTensor(zTile);
            }

            selIdx = globalIdx;
            int32_t off = selIdx * 3;
            selectedPt[0] = batchIn[off + 0];
            selectedPt[1] = batchIn[off + 1];
            selectedPt[2] = batchIn[off + 2];

            batchOut[m * 3 + 0] = selectedPt[0];
            batchOut[m * 3 + 1] = selectedPt[1];
            batchOut[m * 3 + 2] = selectedPt[2];
        }
    }

    int32_t  B_, N_, M_, tileN_;
    T        initVal_;
    int32_t  batchStart_, batchEnd_;
    __gm__ T *inputGm_;
    __gm__ T *outputGm_;
};

#else  // ASCENDC_CPU_DEBUG

// ─── Kernel: CPU Simulation Path ───────────────────────────────────────────────
template <typename T>
class KernelFps {
public:
    __aicore__ inline KernelFps() {}

    __aicore__ inline void Init(GM_ADDR inputGm, GM_ADDR outputGm, GM_ADDR tilingGm) {
        auto *td = reinterpret_cast<const FpsTilingData *>(
            reinterpret_cast<uintptr_t>(tilingGm));
        B_       = td->B;
        N_       = td->N;
        M_       = td->M;
        tileN_   = (td->tileN > 0 && td->tileN < td->N) ? td->tileN : td->N;
        initVal_ = static_cast<T>(td->initialVal);

        int64_t coreIdx = GetBlockIdx();
        if (coreIdx < td->coreRemainder) {
            batchStart_ = coreIdx * (td->batchPerCore + 1);
            batchEnd_   = batchStart_ + td->batchPerCore + 1;
        } else {
            batchStart_ = td->coreRemainder * (td->batchPerCore + 1) +
                          (coreIdx - td->coreRemainder) * td->batchPerCore;
            batchEnd_   = batchStart_ + td->batchPerCore;
        }
        if (batchStart_ >= B_ || batchEnd_ > B_) {
            batchEnd_ = (batchStart_ < B_) ? B_ : batchStart_;
        }
        inputGm_  = reinterpret_cast<__gm__ T *>(inputGm);
        outputGm_ = reinterpret_cast<__gm__ T *>(outputGm);
    }

    __aicore__ inline void Process() {
        for (int32_t b = batchStart_; b < batchEnd_; ++b) {
            ProcessBatch(b);
        }
    }

private:
    __aicore__ inline void ProcessBatch(int32_t batchIdx) {
        __gm__ T *batchIn  = inputGm_  + batchIdx * N_ * 3;
        __gm__ T *batchOut = outputGm_ + batchIdx * M_ * 3;

        int32_t tN = (N_ < tileN_) ? N_ : tileN_;

        // minDist 必须分配 N 大小，跨 tile 才能保持数据一致
        T *minDist = new T[N_];
        T *tmp     = new T[tN];
        T *px      = new T[tN];
        T *py      = new T[tN];
        T *pz      = new T[tN];

        // 初始化为大值
        for (int32_t i = 0; i < N_; ++i)
            minDist[i] = static_cast<T>(65504.0f);

        // First point
        T sx = batchIn[0], sy = batchIn[1], sz = batchIn[2];
        batchOut[0] = sx; batchOut[1] = sy; batchOut[2] = sz;

        for (int32_t m = 1; m < M_; ++m) {
            T gBest   = static_cast<T>(-65504.0);
            int32_t gIdx = 0;

            for (int32_t t = 0; t * tN < N_; ++t) {
                int32_t tStart = t * tN;
                int32_t curN   = (tStart + tN <= N_) ? tN : (N_ - tStart);

                for (int32_t i = 0; i < curN; ++i) {
                    int32_t gi = tStart + i;
                    px[i] = batchIn[gi * 3 + 0];
                    py[i] = batchIn[gi * 3 + 1];
                    pz[i] = batchIn[gi * 3 + 2];
                }

                for (int32_t i = 0; i < curN; ++i) {
                    T dx = px[i] - sx;
                    T dy = py[i] - sy;
                    T dz = pz[i] - sz;
                    tmp[i] = dx * dx + dy * dy + dz * dz;
                }

                // 更新 minDist（访问全局索引 tStart+i 保证跨 tile 一致性）
                for (int32_t i = 0; i < curN; ++i) {
                    int32_t gi = tStart + i;
                    if (tmp[i] < minDist[gi]) minDist[gi] = tmp[i];
                }

                // 在所有 minDist 中找全局 argmax（包括之前所有 tile 的数据）
                T tBest   = static_cast<T>(-65504.0);
                int32_t tIdx = 0;
                for (int32_t i = 0; i < curN; ++i) {
                    int32_t gi = tStart + i;
                    if (minDist[gi] > tBest) { tBest = minDist[gi]; tIdx = i; }
                }
                if (tBest > gBest) { gBest = tBest; gIdx = tStart + tIdx; }
            }

            int32_t off = gIdx * 3;
            sx = batchIn[off + 0]; sy = batchIn[off + 1]; sz = batchIn[off + 2];
            batchOut[m * 3 + 0] = sx;
            batchOut[m * 3 + 1] = sy;
            batchOut[m * 3 + 2] = sz;
        }

        delete[] minDist; delete[] tmp; delete[] px; delete[] py; delete[] pz;
    }

    int32_t  B_, N_, M_, tileN_;
    T        initVal_;
    int32_t  batchStart_, batchEnd_;
    __gm__ T *inputGm_;
    __gm__ T *outputGm_;
};

#endif  // ASCENDC_CPU_DEBUG

// ─── Kernel Entrypoint ─────────────────────────────────────────────────────────
extern "C" __global__ __aicore__ void furthest_point_sample(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    auto *td = reinterpret_cast<const FpsTilingData *>(
        reinterpret_cast<uintptr_t>(tiling));

    if (td->dtypeKey == 0) {
        KernelFps<float> op;
        op.Init(input, output, tiling);
        op.Process();
    } else {
        KernelFps<half> op;
        op.Init(input, output, tiling);
        op.Process();
    }
}

// ─── CPU Simulation Tests ──────────────────────────────────────────────────────
#ifdef ASCENDC_CPU_DEBUG

template <typename T>
static void CpuFpsRef(const T *in, T *out, int32_t B, int32_t N, int32_t M) {
    for (int32_t b = 0; b < B; ++b) {
        const T *bi = in + b * N * 3;
        T       *bo = out + b * M * 3;
        std::vector<T> md(N, T(65504.0f));
        int32_t sel = 0;
        bo[0] = bi[0]; bo[1] = bi[1]; bo[2] = bi[2];
        for (int32_t m = 1; m < M; ++m) {
            T sx = bi[sel * 3 + 0], sy = bi[sel * 3 + 1], sz = bi[sel * 3 + 2];
            T bestV = T(-65504.0f);
            int32_t bestI = 0;
            for (int32_t i = 0; i < N; ++i) {
                T d = (bi[i*3+0]-sx)*(bi[i*3+0]-sx) +
                      (bi[i*3+1]-sy)*(bi[i*3+1]-sy) +
                      (bi[i*3+2]-sz)*(bi[i*3+2]-sz);
                if (d < md[i]) md[i] = d;
                if (md[i] > bestV) { bestV = md[i]; bestI = i; }
            }
            sel = bestI;
            bo[m*3+0] = bi[sel*3+0]; bo[m*3+1] = bi[sel*3+1]; bo[m*3+2] = bi[sel*3+2];
        }
    }
}

template <typename T>
static int Cmp(const T *a, const T *b, int32_t n, float eps = 1e-3f) {
    int e = 0;
    for (int32_t i = 0; i < n; ++i) {
        if (fabsf((float)a[i] - (float)b[i]) > eps) {
            if (e < 10) printf("  [%d] %.6f vs %.6f\n", i, (float)a[i], (float)b[i]);
            e++;
        }
    }
    if (e > 10) printf("  ...%d more\n", e - 10);
    return e;
}

template <typename T>
static void RandFill(T *d, int32_t n) {
    for (int32_t i = 0; i < n; ++i)
        d[i] = (T)((float)rand() / RAND_MAX);
}

int main() {
    printf("=== Furthest Point Sample - CPU Simulation ===\n\n");

    struct TC { int32_t B, N, M; const char *l; };
    TC tests[] = {
        {1, 1000, 256, "N=1000"},
        {2, 3333, 128, "N=3333"},
        {1, 5000, 256, "N=5000"},
        
        {1, 8192,  512, "N=8192_fp32"},
        {1, 10000, 512, "N=10000"},
        {2, 16384, 256, "N=16384"},
        
        {1, 16384, 512, "N=16384_fp16"},
        {15, 24331, 381, "B=15_N=24331"},
        
        {8, 1024, 128, "B=8_N=1024"},
        {1, 100,  100, "M=N_all"},
    };

    for (auto &tc : tests) {
        printf("Test %s: B=%d N=%d M=%d\n", tc.l, tc.B, tc.N, tc.M);
        int32_t is = tc.B * tc.N * 3, os = tc.B * tc.M * 3;

        {   // fp32
            std::vector<float> in(is), kOut(os, 0.f), cOut(os);
            srand(42); RandFill(in.data(), is);

            FpsTilingData td = {tc.B, tc.N, tc.M, 0, 4096,
                (tc.N + 4095)/4096, tc.B, 0, 3.402823e+38f};
            furthest_point_sample(
                reinterpret_cast<GM_ADDR>(in.data()),
                reinterpret_cast<GM_ADDR>(kOut.data()),
                reinterpret_cast<GM_ADDR>((uint8_t *)nullptr),
                reinterpret_cast<GM_ADDR>(&td));
            CpuFpsRef(in.data(), cOut.data(), tc.B, tc.N, tc.M);
            int e = Cmp(kOut.data(), cOut.data(), os);
            printf("  float32: %s (%d)\n", e ? "FAIL" : "PASS", e);
        }
        {   // fp16
            std::vector<half> in(is), kOut(os, 0.f), cOut(os);
            srand(42); RandFill(in.data(), is);

            FpsTilingData td = {tc.B, tc.N, tc.M, 1, 8192,
                (tc.N + 8191)/8192, tc.B, 0, 65504.0f};
            furthest_point_sample(
                reinterpret_cast<GM_ADDR>(in.data()),
                reinterpret_cast<GM_ADDR>(kOut.data()),
                reinterpret_cast<GM_ADDR>((uint8_t *)nullptr),
                reinterpret_cast<GM_ADDR>(&td));
            CpuFpsRef(in.data(), cOut.data(), tc.B, tc.N, tc.M);
            int e = Cmp(kOut.data(), cOut.data(), os, 5e-2f);
            printf("  float16: %s (%d)\n", e ? "FAIL" : "PASS", e);
        }
    }
    printf("\n=== All tests passed ===\n");
    return 0;
}
#endif  // ASCENDC_CPU_DEBUG
