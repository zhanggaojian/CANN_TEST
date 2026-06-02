/*!
 * \file celu_tiling.cpp
 * \brief Celu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/celu_tiling_data.h"
#include "../op_kernel/celu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CeluTilingFunc(gert::TilingContext* context)
{
    // TODO: 实现 Tiling 逻辑
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    CeluTilingData* tiling = context->GetTilingData<CeluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::Shape *inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    tiling->totalNum = inputShape->GetShapeSize();
    if (tiling->totalNum <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t usedCoreNum = std::max<int64_t>(1, std::min<int64_t>(coreNum, tiling->totalNum));
    tiling->blockFactor = tiling->totalNum / usedCoreNum; //每个core上处理的元素个数
    tiling->ubFactor = std::min<uint64_t>(ubSize, tiling->blockFactor); //每个core上ub的处理的元素个数
    tiling->lastBlockFactor = tiling->totalNum - (usedCoreNum - 1) * tiling->blockFactor;
    float alpha = 1.0f;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float *alphaAttr = attrs->GetAttrPointer<float>(0);
        if (alphaAttr != nullptr) {
            alpha = *alphaAttr;
        }
    }
    if (alpha <= 0.0f) {
        return ge::GRAPH_FAILED;
    }
    tiling->alpha = alpha;
    context->SetBlockDim(usedCoreNum);

    // 使用 GET_TPL_TILING_KEY 获取 tilingKey
    uint64_t tilingKey = GET_TPL_TILING_KEY(CELU_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct CeluCompileInfo {};

IMPL_OP_OPTILING(Celu).Tiling(CeluTilingFunc).TilingParse<CeluCompileInfo>(TilingParseForCelu);

} // namespace optiling
