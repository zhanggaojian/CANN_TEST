#include "celu_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr int32_t kBufferNum = 2;
}

class KernelCelu {
public:
    __aicore__ inline KernelCelu() = default;

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, GM_ADDR tiling) {
        GET_TILING_DATA(tilingData, tiling);
        totalLength_ = tilingData.get_totalLength();
        tileLength_ = tilingData.get_tileLength();
        blockLength_ = tilingData.get_blockLength();
        lastBlockLength_ = tilingData.get_lastBlockLength();
        alpha_ = tilingData.get_alpha();
        invAlpha_ = 1.0f / alpha_;

        const uint32_t coreId = GetBlockIdx();
        const uint32_t blockOffset = coreId * blockLength_;
        const uint32_t currentBlockLength =
            (coreId == GetBlockNum() - 1) ? lastBlockLength_ : blockLength_;

        startOffset_ = blockOffset;
        currentLength_ = currentBlockLength;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(self) + startOffset_,
                                currentLength_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out) + startOffset_,
                               currentLength_);

        pipe_.InitBuffer(inQueue_, kBufferNum, tileLength_ * sizeof(float));
        pipe_.InitBuffer(outQueue_, kBufferNum, tileLength_ * sizeof(float));
        pipe_.InitBuffer(posBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(negBuf_, tileLength_ * sizeof(float));
    }

    __aicore__ inline void Process() {
        uint32_t offset = 0;
        while (offset < currentLength_) {
            const uint32_t calcLength =
                (currentLength_ - offset > tileLength_) ? tileLength_ : (currentLength_ - offset);
            CopyIn(offset, calcLength);
            Compute(calcLength);
            CopyOut(offset, calcLength);
            offset += calcLength;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        LocalTensor<float> input = inQueue_.AllocTensor<float>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(length * sizeof(float)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(input, selfGm_[offset], copyParams, padParams);
        inQueue_.EnQue(input);
    }

    __aicore__ inline void Compute(uint32_t length) {
        LocalTensor<float> input = inQueue_.DeQue<float>();
        LocalTensor<float> output = outQueue_.AllocTensor<float>();
        LocalTensor<float> pos = posBuf_.Get<float>();
        LocalTensor<float> neg = negBuf_.Get<float>();

        Maxs(pos, input, 0.0f, length);
        Muls(neg, input, invAlpha_, length);
        Exp(neg, neg, length);
        Adds(neg, neg, -1.0f, length);
        Muls(neg, neg, alpha_, length);
        Mins(neg, neg, 0.0f, length);
        Add(output, pos, neg, length);

        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        LocalTensor<float> output = outQueue_.DeQue<float>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(length * sizeof(float)), 0, 0};
        DataCopyPad(outGm_[offset], output, copyParams);
        outQueue_.FreeTensor(output);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> inQueue_;
    TQue<QuePosition::VECOUT, kBufferNum> outQueue_;
    TBuf<QuePosition::VECCALC> posBuf_;
    TBuf<QuePosition::VECCALC> negBuf_;
    GlobalTensor<float> selfGm_;
    GlobalTensor<float> outGm_;

    uint32_t totalLength_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t blockLength_ = 0;
    uint32_t lastBlockLength_ = 0;
    uint32_t startOffset_ = 0;
    uint32_t currentLength_ = 0;
    float alpha_ = 1.0f;
    float invAlpha_ = 1.0f;
};

extern "C" __global__ __aicore__ void celu(GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                           GM_ADDR tiling) {
    (void)workspace;
    KernelCelu op;
    op.Init(self, out, tiling);
    op.Process();
}

