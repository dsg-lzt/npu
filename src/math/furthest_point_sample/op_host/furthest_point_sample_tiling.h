/**
 * FurthestPointSample - Tiling Data Definition (Host-side)
 */
#ifndef FPS_TILING_H
#define FPS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FpsTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, C);
  TILING_DATA_FIELD_DEF(uint32_t, dataTypeLength);
  TILING_DATA_FIELD_DEF(uint32_t, ubPointsNum);
  TILING_DATA_FIELD_DEF(uint32_t, ubMinDistNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileLoopNum);
  TILING_DATA_FIELD_DEF(uint32_t, batchPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, coreRemainder);
  TILING_DATA_FIELD_DEF(uint32_t, wsOffset);
  TILING_DATA_FIELD_DEF(uint32_t, wsStride);
  TILING_DATA_FIELD_DEF(float,    initVal);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FurthestPointSample, FpsTilingData)

}  // namespace optiling

#endif  // FPS_TILING_H
