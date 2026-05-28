#include "celu_tiling.h"

#include <algorithm>
#include <cstdint>

#include "graph/operator_reg.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kDefaultTileLength = 8192;
constexpr uint32_t kMinBlockLength = 1;

uint64_t GetShapeElementCount(const gert::Shape *shape) {
    if (shape == nullptr) {
        return 1;
    }
    const int64_t dimNum = shape->GetDimNum();
    if (dimNum == 0) {
        return 1;
    }

    uint64_t total = 1;
    for (int64_t i = 0; i < dimNum; ++i) {
        const int64_t dim = shape->GetDim(i);
        if (dim <= 0) {
            return 0;
        }
        total *= static_cast<uint64_t>(dim);
    }
    return total;
}
}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *inputShape = context->GetInputShape(0);
    const uint64_t total64 = GetShapeElementCount(inputShape);
    if (total64 == 0 || total64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    float alpha = 1.0f;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float *alphaAttr = attrs->GetAttrPointer<float>(0);
        if (alphaAttr != nullptr) {
            alpha = *alphaAttr;
        }
    }
    if (!(alpha > 0.0f)) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }

    const uint32_t totalLength = static_cast<uint32_t>(total64);
    const uint32_t usedCoreNum = std::max<uint32_t>(
        1, std::min<uint32_t>(coreNum, totalLength));
    const uint32_t blockLength = (totalLength + usedCoreNum - 1) / usedCoreNum;
    const uint32_t lastBlockLength =
        totalLength - blockLength * (usedCoreNum - 1);
    const uint32_t tileLength =
        std::max<uint32_t>(kMinBlockLength, std::min<uint32_t>(kDefaultTileLength, blockLength));

    CeluTilingData tiling;
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(tileLength);
    tiling.set_blockLength(blockLength);
    tiling.set_lastBlockLength(lastBlockLength);
    tiling.set_alpha(alpha);

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Celu : public OpDef {
public:
    explicit Celu(const char *name) : OpDef(name) {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("alpha").AttrType(OPTIONAL).Float(1.0f);

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310p")
            .AddConfig("ascend910")
            .AddConfig("ascend910b");
    }
};

OP_ADD(Celu);

}  // namespace ops

