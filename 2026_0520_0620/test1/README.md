# Celu Ascend C Custom Operator

This directory contains a CANN Ascend C custom operator implementation for
`torch.nn.functional.celu` with `float32` ND tensors.

Formula:

```text
celu(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
```

Supported behavior:

- Input: `self`, `float32`, ND, 0-8 dimensions.
- Attribute: `alpha`, positive `float`, default `1.0`.
- Output: `out`, `float32`, same shape as input.
- Handles scalar input, non-32-aligned element counts, `NaN`, `+Inf`, and `-Inf`.

## Files

- `op_host/celu.cpp`: operator definition, shape/type inference, tiling.
- `op_host/celu_tiling.h`: tiling data shared by host and kernel.
- `op_kernel/celu.cpp`: Ascend C AI Core kernel.
- `tests/test_reference.py`: PyTorch/NumPy reference tests for the required math.
- `CMakeLists.txt`: minimal custom operator build entry.

## Build

Set the CANN package path first, for example:

```bash
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
mkdir -p build
cd build
cmake .. -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
cmake --build . -j
```

The exact packaging/install command can vary by CANN version and local contest
template. The source files are kept in the common `op_host` + `op_kernel`
layout so they can also be copied into an `msopgen` generated project.

## Reference Test

```bash
python tests/test_reference.py
```

