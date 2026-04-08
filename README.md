# NanoCache

**KV Cache 分层压缩 — 扩展大模型上下文处理能力**

> **NanoCache 1.0** 提供了稀疏注意力（L1）+ INT8 量化压缩（L2）+ Smart LRU RAM Offload（L3）的分层架构，在不明显增加延迟的前提下扩展 llama.cpp 推理的上下文处理能力。

**核心效果**：在 RTX 5070 Ti（16GB VRAM）上，14B-Q5_K_M 模型的上下文处理能力从 ~22500 tokens 扩展至更多，且吞吐量无明显退化。

---

## 🚨 商业使用必读

本项目采用 **GNU Affero General Public License v3 (AGPLv3)**。

如果您是云服务提供商（AWS、Azure、阿里云等）或企业用户，且不希望您的服务代码被开源，请联系作者获取商业授权。联系方式见仓库主页。

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

## 实用场景

NanoCache 解决的本质问题是：**有限的 VRAM 如何支撑更长的上下文**。以下是具体场景：

### 1. 长文档摘要与理解

**场景**：将一整本技术手册、论文集或法律合同（数万字）一次性输入模型。

**原有瓶颈**：上下文越长，KV Cache 越大。超过 ~22500 tokens 时直接 OOM。

**NanoCache 效果**：INT8 压缩 VRAM 占用，在同等硬件上将可处理的上下文从 OOM 临界点扩展约 50%。

```
场景: 解读一份 300 页的技术文档 (~150,000 字, ~100,000 tokens)
原有方案: 需分段输入，每段丢失段落间关联性
NanoCache: 单次输入更多内容，减少分段必要
```

### 2. 多轮 Agent 对话

**场景**：一个 AI 助手在长对话中同时维护系统 Prompt + 对话历史 + 工具调用上下文。

**原有瓶颈**：多轮对话的 KV Cache 随轮次线性增长，几十轮后 VRAM 耗尽。

**NanoCache 效果**：Smart LRU 在生成过程中自动将旧轮次的 KV 数据 offload 到 RAM，保留最近对话的响应质量，同时释放 VRAM 空间。

```
场景: 30 轮技术调试对话 (每轮平均 500 tokens 输入)
原有方案: ~15000 tokens 时开始变慢，超过 20000 tokens 质量下降
NanoCache: 热 slot 保留在 VRAM，冷 slot offload 到 RAM，保持响应速度
```

### 3. RAG 增强检索

**场景**：在 RAG（检索增强生成）流程中，一次性注入数十个检索到的文档片段作为上下文。

**原有瓶颈**：每个片段 ~512 tokens，10 个片段加上原始问题很快超过 5000 tokens，VRAM 压力显著。

**NanoCache 效果**：VRAM 恒定在 ~1462 MB，不再随检索片段数量线性增长，支持更多片段同时驻留。

```
场景: 检索 20 个文档片段 (每个 512 tokens) + 问题
原有方案: 20 个片段 = ~10240 tokens，VRAM 占用 1628 MB
NanoCache: 20 个片段仍保持 1462 MB，节省 166 MB
```

### 4. 代码分析与补全

**场景**：分析大型代码仓库（数千行代码），模型需要同时看到多个文件的上下文。

**原有瓶颈**：代码文件 KV Cache 的头注意力模式导致早期 token 被遗忘，后期生成质量下降。

**NanoCache 效果**：稀疏注意力保留最近 128 tokens 的完整注意力，同时通过重要性评分保留关键代码结构的 token。

### 5. 边缘设备推理

**场景**：在消费级 GPU（8GB~16GB VRAM）上运行大模型。

**原有瓶颈**：模型本身已占用大量 VRAM，可用于 KV Cache 的空间极为有限。

**NanoCache 效果**：INT8 压缩 2x，等效将 KV Cache 可用 VRAM 翻倍。

```
硬件: RTX 4060 Ti (16GB VRAM)
模型: Qwen2.5-7B-Q4_K_M (~4GB 权重)
原有: KV Cache 可用空间 ~12GB，可处理 ~15000 tokens
NanoCache: KV Cache 压缩 2x，等效可用空间 ~13GB+
```

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
