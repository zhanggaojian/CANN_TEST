/*!
 * \file celu_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __CELU_TILING_KEY_H__
#define __CELU_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define CELU_TPL_SCH_MODE_0 0
#define CELU_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    Celu,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, CELU_TPL_SCH_MODE_0, CELU_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, CELU_TPL_SCH_MODE_0, CELU_TPL_SCH_MODE_1)));

#endif
