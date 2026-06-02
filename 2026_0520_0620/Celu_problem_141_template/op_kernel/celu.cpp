/*!
 * \file celu.cpp
 * \brief Celu 算子 kernel 入口
 */

#include "celu.h"

enum class CeluTilingKey : uint32_t
{
    TILING_KEY_CELU_MODE_0 = 0,
    TILING_KEY_CELU_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void celu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(CeluTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_0)) {
        NsCelu::Celu<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(CeluTilingKey::TILING_KEY_CELU_MODE_1)) {
        NsCelu::Celu<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
