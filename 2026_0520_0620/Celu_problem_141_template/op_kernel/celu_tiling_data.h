/*!
 * \file celu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _CELU_TILING_DATA_H_
#define _CELU_TILING_DATA_H_

struct CeluTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 0;  // 每个核处理的元素数量
    int64_t lastBlockFactor = 0; // 最后一个核处理的元素数量
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
    float alpha = 1.0f;      // alpha 参数
};
#endif
