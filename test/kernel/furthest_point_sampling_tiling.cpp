#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUF_NUM = 2;
constexpr int32_t OUT_NUM = 1;

template <typename T, typename IDX_T>
class KernelFPS {
public:
    __aicore__ inline void Init(GM_ADDR dataset, GM_ADDR temp, GM_ADDR idxs,
        uint32_t n, uint32_t m, uint32_t b_num, uint32_t remain,
        uint32_t block_size, uint32_t idx_size, uint32_t length)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero");
        this->n = n;
        this->m = m;
        this->block_size = block_size;
        this->length = length;
        this->idx_size = idx_size;

        uint32_t bid = GetBlockIdx();
        this->b_num = bid < remain ? b_num + 1 : b_num;
        this->base_offset = (b_num + 1) * bid;
        this->base_offset = bid <= remain ? this->base_offset
                                          : this->base_offset - (bid - remain);

        datasetGm.SetGlobalBuffer((__gm__ T*)dataset + this->base_offset * n * 3);
        tempGm.SetGlobalBuffer((__gm__ T*)temp + this->base_offset * ((n + 7) / 8 * 8));
        idxsGm.SetGlobalBuffer((__gm__ IDX_T*)idxs + this->base_offset * m);

        pipe.InitBuffer(queX, BUF_NUM, this->block_size * sizeof(T));
        pipe.InitBuffer(queY, BUF_NUM, this->block_size * sizeof(T));
        pipe.InitBuffer(queZ, BUF_NUM, this->block_size * sizeof(T));
        pipe.InitBuffer(queTemp, BUF_NUM, this->block_size * sizeof(T));
        pipe.InitBuffer(queDist, BUF_NUM, this->block_size * sizeof(T));
        pipe.InitBuffer(resIdxQue, OUT_NUM, idx_size * sizeof(IDX_T));

        pipe.InitBuffer(queMR, BUF_NUM, 2 * sizeof(T));
        pipe.InitBuffer(queMT, BUF_NUM, this->block_size * sizeof(T));
        pipe.InitBuffer(queSH, BUF_NUM, this->block_size * sizeof(T));

        pipe.InitBuffer(quePV, BUF_NUM, length * sizeof(T));
        pipe.InitBuffer(quePI, BUF_NUM, length * sizeof(T));
    }

    // ===== CopyIn: MTE2 loads block[progress] from GM into VECIN queues =====
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t dba, uint32_t tba)
    {
        uint32_t k = progress * this->block_size;
        if (k >= this->n) {
            return;
        }
        uint32_t act = this->block_size < (this->n - k) ? this->block_size
                                                         : (this->n - k);
        uint32_t cnt = (act + 7) / 8 * 8;

        LocalTensor<T> x = queX.AllocTensor<T>();
        LocalTensor<T> y = queY.AllocTensor<T>();
        LocalTensor<T> z = queZ.AllocTensor<T>();
        LocalTensor<T> t = queTemp.AllocTensor<T>();

        DataCopy(x, datasetGm[dba + 0 * this->n + k], cnt);
        DataCopy(y, datasetGm[dba + 1 * this->n + k], cnt);
        DataCopy(z, datasetGm[dba + 2 * this->n + k], cnt);
        DataCopy(t, tempGm[tba + k], cnt);

        queX.EnQue(x);
        queY.EnQue(y);
        queZ.EnQue(z);
        queTemp.EnQue(t);
    }

    // ===== Compute: Vector processes one block, writes min distance =====
    __aicore__ inline void Compute(T negX, T negY, T negZ, uint32_t act)
    {
        LocalTensor<T> cx = queX.DeQue<T>();
        LocalTensor<T> cy = queY.DeQue<T>();
        LocalTensor<T> cz = queZ.DeQue<T>();
        LocalTensor<T> ct = queTemp.DeQue<T>();

        LocalTensor<T> tD = queDist.AllocTensor<T>();

        Adds(cx, cx, negX, act);
        Mul(cx, cx, cx, act);
        Adds(cy, cy, negY, act);
        Mul(cy, cy, cy, act);
        Adds(cz, cz, negZ, act);
        Mul(cz, cz, cz, act);

        Add(tD, cx, cy, act);
        Add(tD, tD, cz, act);

        Min(ct, ct, tD, act);

        queDist.FreeTensor(tD);
        queX.FreeTensor(cx);
        queY.FreeTensor(cy);
        queZ.FreeTensor(cz);

        queTemp.EnQue(ct);
    }

    // ===== CopyOut: MTE3 writes result to GM =====
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t tba)
    {
        uint32_t k = progress * this->block_size;
        if (k >= this->n) {
            return;
        }
        uint32_t act = this->block_size < (this->n - k) ? this->block_size
                                                         : (this->n - k);
        uint32_t cnt = (act + 7) / 8 * 8;

        LocalTensor<T> ct = queTemp.DeQue<T>();
        PipeBarrier<PIPE_V>();
        DataCopy(tempGm[tba + k], ct, cnt);
        queTemp.FreeTensor(ct);
    }

    __aicore__ inline void Process() {
        uint32_t n = this->n;
        uint32_t m = this->m;

        for (uint32_t b_idx = 0; b_idx < this->b_num; b_idx++) {
            LocalTensor<IDX_T> idxs = resIdxQue.AllocTensor<IDX_T>();
            Duplicate(idxs, static_cast<IDX_T>(0), this->idx_size);
            IDX_T lastIdx = 0;
            idxs.SetValue(0, lastIdx);
            PipeBarrier<PIPE_V>();

            uint32_t dba = b_idx * n * 3;
            uint32_t tba = b_idx * n;
            uint32_t iba = b_idx * m;

            for (uint32_t j = 1; j < m; j++) {
                T x1 = datasetGm.GetValue(dba + 0 * n + lastIdx);
                T y1 = datasetGm.GetValue(dba + 1 * n + lastIdx);
                T z1 = datasetGm.GetValue(dba + 2 * n + lastIdx);

                T negX = static_cast<T>(-static_cast<float>(x1));
                T negY = static_cast<T>(-static_cast<float>(y1));
                T negZ = static_cast<T>(-static_cast<float>(z1));

                PipeBarrier<PIPE_S>();

                // ===== Distance computation pipeline =====
                uint32_t loopCnt = this->length + BUF_NUM;
                for (uint32_t i = 0; i < loopCnt; i++) {
                    CopyIn(i, dba, tba);
                    uint32_t k = i * this->block_size;
                    if (k < n) {
                        uint32_t act = this->block_size < (this->n - k)
                                           ? this->block_size : (this->n - k);
                        Compute(negX, negY, negZ, act);
                    }
                    CopyOut(i, tba);
                }

                PipeBarrier<PIPE_ALL>();

                // ===== ReduceMax phase =====
                LocalTensor<T> pv = quePV.AllocTensor<T>();
                LocalTensor<T> pi = quePI.AllocTensor<T>();

                uint32_t rdCnt = this->length + BUF_NUM;
                for (uint32_t i = 0; i < rdCnt; i++) {
                    uint32_t k = i * this->block_size;

                    // CopyIn
                    if (k < n) {
                        uint32_t act = this->block_size < (n - k)
                                           ? this->block_size : (n - k);
                        uint32_t cnt = (act + 7) / 8 * 8;

                        LocalTensor<T> ct = queMT.AllocTensor<T>();
                        DataCopy(ct, tempGm[tba + k], cnt);
                        queMT.EnQue(ct);
                    }

                    // Compute (ReduceMax)
                    if (k < n) {
                        LocalTensor<T> ct = queMT.DeQue<T>();
                        uint32_t act = this->block_size < (n - k)
                                           ? this->block_size : (n - k);

                        LocalTensor<T> mr = queMR.AllocTensor<T>();
                        LocalTensor<T> sh = queSH.AllocTensor<T>();

                        ReduceMax(mr, ct, sh, act, true);

                        PipeBarrier<PIPE_V>();

                        T v0 = mr.GetValue(0);
                        T v1 = mr.GetValue(1);

                        queMR.FreeTensor(mr);
                        queSH.FreeTensor(sh);

                        pv.SetValue(i, v0);
                        pi.SetValue(i, v1);

                        queMT.FreeTensor(ct);
                    }
                }

                PipeBarrier<PIPE_V>();

                LocalTensor<T> fr = queMR.AllocTensor<T>();
                LocalTensor<T> fs = queSH.AllocTensor<T>();

                ReduceMax(fr, pv, fs, this->length, true);

                PipeBarrier<PIPE_V>();

                T wi = fr.GetValue(1);

                uint32_t wk = static_cast<uint32_t>(
                    static_cast<int32_t>(static_cast<float>(wi)));

                T iv = pi.GetValue(wk);
                uint32_t ii = static_cast<uint32_t>(
                    static_cast<int32_t>(static_cast<float>(iv)));

                lastIdx = static_cast<IDX_T>(wk * this->block_size + ii);

                queMR.FreeTensor(fr);
                queSH.FreeTensor(fs);
                quePV.FreeTensor(pv);
                quePI.FreeTensor(pi);

                idxs.SetValue(j, lastIdx);
            }

            PipeBarrier<PIPE_ALL>();
            SetAtomicAdd<int32_t>();
            DataCopy(idxsGm[iba], idxs, this->idx_size);
            SetAtomicNone();
            PipeBarrier<PIPE_ALL>();

            resIdxQue.FreeTensor(idxs);
        }
    }

private:
    TPipe pipe;
    GlobalTensor<T> datasetGm, tempGm;
    GlobalTensor<IDX_T> idxsGm;
    TQue<QuePosition::VECIN, BUF_NUM> queX, queY, queZ, queTemp, queMT;
    TQue<QuePosition::VECCALC, BUF_NUM> queDist, queMR, queSH, quePV, quePI;
    TQue<QuePosition::VECOUT, OUT_NUM> resIdxQue;
    uint32_t n, m, b_num, base_offset, block_size, length, idx_size;
};

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR x, GM_ADDR tmp, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFPS<DTYPE_X, int32_t> op;
    op.Init(x, tmp, y,
            static_cast<uint32_t>(tiling_data.N),
            static_cast<uint32_t>(tiling_data.npoint),
            static_cast<uint32_t>(tiling_data.b_num),
            static_cast<uint32_t>(tiling_data.remain),
            static_cast<uint32_t>(tiling_data.block_size),
            static_cast<uint32_t>(tiling_data.idx_size),
            static_cast<uint32_t>(tiling_data.length));
    op.Process();
}