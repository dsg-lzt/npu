/**
 * FurthestPointSample - Raw Tiling Data Layout
 *
 * This struct MUST match the layout produced by BEGIN_TILING_DATA_DEF /
 * TILING_DATA_FIELD_DEF macros in op_host/fps_tiling.h.
 *
 * The GET_TILING_DATA macro in the kernel reads this struct directly
 * from the tiling buffer.
 *
 * WARNING: Fields MUST exactly match fps_tiling.h order and types.
 */
#ifndef FPS_TILING_DATA_H_
#define FPS_TILING_DATA_H_

#include <cstdint>

#pragma pack(push, 8)
struct FpsTilingRaw {
    uint32_t B;
    uint32_t N;
    uint32_t M;
    uint32_t C;
    uint32_t dataTypeLength;
    uint32_t ubPointsNum;
    uint32_t ubMinDistNum;
    uint32_t tileLoopNum;
    uint32_t batchPerCore;
    uint32_t coreRemainder;
    uint32_t wsOffset;
    uint32_t wsStride;
    float    initVal;
};
#pragma pack(pop)

#endif  // FPS_TILING_DATA_H_
