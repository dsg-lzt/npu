#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FurthestPointSamplingTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, N);
    TILING_DATA_FIELD_DEF(uint64_t, npoint);
    TILING_DATA_FIELD_DEF(uint64_t, b_num);
    TILING_DATA_FIELD_DEF(uint64_t, remain);
    TILING_DATA_FIELD_DEF(uint64_t, block_size);
    TILING_DATA_FIELD_DEF(uint64_t, idx_size);
    TILING_DATA_FIELD_DEF(uint64_t, length);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FurthestPointSampling, FurthestPointSamplingTilingData);
}
