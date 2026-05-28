#ifndef CELU_TILING_H
#define CELU_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(CeluTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);
    TILING_DATA_FIELD_DEF(uint32_t, lastBlockLength);
    TILING_DATA_FIELD_DEF(float, alpha);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Celu, CeluTilingData)

}  // namespace optiling

#endif  // CELU_TILING_H

