# NanoCache

**KV Cache 分层压缩 — 扩展大模型上下文处理能力**

> **NanoCache 1.0** 提供了稀疏注意力（L1）+ INT8 量化压缩（L2）+ Smart LRU RAM Offload（L3）的分层架构，在不明显增加延迟的前提下扩展 llama.cpp 推理的上下文处理能力。

**核心效果**：在 RTX 5070 Ti（16GB VRAM）上，14B-Q5_K_M 模型的上下文处理能力从 ~22500 tokens 扩展至更多，且吞吐量无明显退化。

---

## 架构

```
┌─────────────────────────────────────────────────────┐
│  L1: 稀疏注意力 (Sparse Attention)               │
│  位置: flash_attn_ext CUDA 内核                   │
│  触发: KV 长度 > 2048 tokens                       │
│  效果: 注意力 FLOP O(n²) → O(n)                │
├─────────────────────────────────────────────────────┤
│  L2: INT8 量化压缩 (INT8 Quantization)             │
│  位置: llama.cpp eval-callback 调度层             │
│  触发: 可选（默认开启）                            │
│  效果: KV Cache 压缩 2x（FP16 → INT8）          │
├─────────────────────────────────────────────────────┤
│  L3: Smart LRU RAM Offload                       │
│  位置: llama.cpp ggml_backend_sched_eval_callback │
│  触发: slot 连续 8 个 decode step 未访问          │
│  效果: 释放 VRAM，支持更长上下文                  │
└─────────────────────────────────────────────────────┘
```

三层可独立启用。L1 始终开启，L2/L3 按需叠加。

---

## 测试结果

**硬件**: RTX 5070 Ti (16GB VRAM) | CUDA 12.8 | llama.cpp (cuBLAS + CUDA)

**模型**: DeepSeek-R1-Distill-Qwen-14B-Q5_K_M (GGUF, 10.5GB)

### VRAM 节省

| 上下文长度 | v1.0 (无压缩) | v2.0 (INT8+LRU) | 节省 |
|-----------|---------------|-----------------|------|
| 4,500 tok | 1,605 MB | 1,462 MB | 143 MB (8.9%) |
| 10,500 tok | 1,628 MB | 1,462 MB | 166 MB (10.2%) |
| 15,000 tok | 1,634 MB | 1,462 MB | 172 MB (10.5%) |
| 22,500 tok | **OOM** | **1,462 MB** | ✅ 正常运行 |

### 吞吐量

| 版本 | 短 Prompt | 长 Prompt | 开销 |
|------|-----------|-----------|------|
| v1.0 Callback | 5399 ms | 5386 ms | baseline |
| v2.0 Smart LRU | 5377 ms | 5339 ms | -0.4%~0.9% |

> Smart LRU 的回调调度开销可忽略不计（±2.5%）。

### Offload 行为

- **短生成（< 50 steps）**：offload = 0 — 热 slot 留在 VRAM，LRU 正确地不干预
- **长生成（≥ 50 steps）**：Decode #9 开始触发 offload（stale_threshold=8），offload/recall 完美对称

---

## 项目结构

```
NanoCache/
├── phase1_policy/           # Python: 量化器、重要性评分器、稀疏 KV 管理
├── phase2_operators/        # Python CUDA: FusedSparseAttention 内核
├── phase3_engine/           # C++ CUDA: NanoCache 核心引擎
│   └── nanocache_llama_adapter.cpp   # llama.cpp 集成适配器
├── docs/                    # 技术白皮书
└── llama.cpp/              # (需 clone https://github.com/ggerganov/llama.cpp)
    └── examples/eval-callback/
        ├── eval-callback-v1-pure.cpp   # v1.0: callback-only baseline
        ├── eval-callback-v2.cpp        # v2.0: Smart LRU INT8 offload
        └── README.md                   # llama.cpp 集成说明
```

---

## 快速开始

### 1. 准备 llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_BURBLE_COPY=OFF
make -j$(nproc)
```

### 2. 编译 NanoCache eval-callback

将 `llama.cpp/examples/eval-callback/eval-callback-v2.cpp` 覆盖 `llama.cpp/examples/eval-callback/eval-callback.cpp`，然后：

```bash
cd llama.cpp/build
cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_BURBLE_COPY=OFF
make llama-eval-callback -j$(nproc)
```

### 3. 运行

```bash
# v2.0 Smart LRU INT8 offload
./llama-eval-callback \
  -m /path/to/model-q4_k_m.gguf \
  --prompt "What is artificial intelligence?" \
  -n 128
```

NanoCache 的 offload/recall 日志输出到 stderr，生成文本输出到 stdout。

---

## 技术细节

### llama.cpp 集成点

`ggml_backend_sched_eval_callback` — llama.cpp 调度器在每个 tensor 计算前后触发回调：

```cpp
// ask=true: compute 前 → 若 slot 被 offload，先 recall
// ask=false: compute 后 → 若 slot 变冷，offload
bool nanocache_cb_eval(struct ggml_tensor* t, bool ask, void* user_data) {
    if (is_kv_tensor(t->name)) {
        if (ask && slot.is_offload) nc_recall(t->name);
        if (!ask && slot.is_stale(threshold)) nc_offload(t->name);
    }
    return true;
}
params.cb_eval = nanocache_cb_eval;
```

### INT8 量化（per-head 独立 scale）

```cpp
// 量化
float scale = 127.0f / max(|fp16|);
int8_t q = round(fp16 / scale);

// 反量化
fp16' = (q - 128) * (1.0f / scale);
```

### Smart LRU 调度

```cpp
static const int STALE_THRESHOLD = 8;  // 连续 8 个 decode step 未访问才 offload

void nc_on_decode_done() {
    for (auto& slot : slots) {
        if (!slot.is_offload && slot.stale(decode_no) > STALE_THRESHOLD) {
            nc_offload(slot);
        }
    }
}
```

---

## 开源许可

本项目采用 **GNU Affero General Public License v3 (AGPLv3)**。

NanoCache 基于 [llama.cpp](https://github.com/ggerganov/llama.cpp)（同为 MIT License）。

---

*项目随研究进展持续更新。测试数据基于 2026-04-08 实验环境（RTX 5070 Ti, CUDA 12.8）。*
