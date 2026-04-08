# Phase 3 Engine — NanoCache Core

C++ CUDA 实现，与 llama.cpp 深度集成。

## 文件说明

| 文件 | 说明 |
|------|------|
| `nanocache_offload.h` | 头文件：Slot 定义、Manager 声明、tensor 分类辅助 |
| `nanocache_offload.cpp` | 核心实现：INT8 量化/反量化、LRU offload 管理器 |
| `nanocache_llama_adapter.cpp` | llama.cpp 集成入口：注入 `cb_eval` 回调 |
| `nanocache_cuda.cu` | CUDA kernel：稀疏注意力（L1 层）|
| `nanocache_attention.h/cpp` | attention 接口（备用）|
| `llama_integration/` | 详细集成文档 |

## 编译

```bash
# 方式一：在 llama.cpp 内编译（推荐）
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON
make llama-eval-callback -j$(nproc)

# 方式二：独立编译 phase3_engine
nvcc -O3 -arch=sm_90 nanocache_cuda.cu nanocache_offload.cpp \
    -o nanocache_engine \
    -lcuda -lcublas -lstdc++
```

## llama.cpp 集成

```cpp
#include "nanocache_offload.h"

// 全局 manager
NanoCacheOffloadManager g_nc;

// 在 llama_init_params 中注入
params.cb_eval = nanocache_cb_eval;
params.cb_eval_user_data = &g_nc;
params.warmup = false;  // NanoCache 自己管理预热

// main() 开头初始化 CUDA
g_nc.init();
```

## INT8 量化规格

- 格式：per-head max-abs INT8
- scale = 127.0 / max(|fp16|)
- 存储：`(fp16 * scale).round() + 128 → uint8`
- 反量化：`(uint8 - 128) / scale → fp16`
- RAM buffer：`cudaMallocHost` 2GB pinned memory

## Smart LRU

`STALE_THRESHOLD = 8`：连续 8 个 decode step 未访问的 slot 才会被 offload。正常推理（< 50 步）时所有 slot 均为热 slot，LRU 不干预。
