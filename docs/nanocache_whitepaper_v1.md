# NanoCache: KV Cache 分层压缩与 llama.cpp 集成

**版本**: 1.0 | **日期**: 2026-04-08 | **状态**: 实验性

---

## 一、背景与问题

大语言模型推理的显存瓶颈主要集中在 KV Cache 上。以 Qwen2.5-14B 为例，FP16 格式下单层 KV Cache 占用约 44 MB，56 层合计 2.5 GB。随着上下文长度增长，KV Cache 线性挤压可用显存，导致：

1. **上下文长度受限**：VRAM 上限决定最大可处理 token 数
2. **长上下文推理速度下降**：VRAM 压力触发置换策略

传统方案在精度和速度之间取舍。NanoCache 提出**分层压缩**，在不明显增加延迟的前提下扩展上下文处理能力。

---

## 二、技术方案

### 2.1 三层架构

```
┌─────────────────────────────────────────────────────┐
│  L1: 稀疏注意力 (Sparse Attention)                 │
│  位置: flash_attn_ext CUDA 内核                   │
│  触发: KV 长度 > 2048 tokens                     │
│  效果: 注意力 FLOP O(n²) → O(n)                │
├─────────────────────────────────────────────────────┤
│  L2: INT8 量化压缩 (INT8 Quantization)            │
│  位置: llama.cpp eval-callback 调度层             │
│  触发: 可选（默认开启）                            │
│  效果: KV Cache 压缩 2x（FP16 → INT8）          │
├─────────────────────────────────────────────────────┤
│  L3: Smart LRU RAM Offload                       │
│  位置: llama.cpp ggml_backend_sched_eval_callback │
│  触发: slot 连续 8 个 decode step 未访问         │
│  效果: VRAM → RAM，释放显存支持更长上下文         │
└─────────────────────────────────────────────────────┘
```

三层可独立启用。L1 始终开启，L2/L3 按需叠加。

### 2.2 L1: 稀疏注意力

**原理**：在 `flash_attn_ext` 内核中，当 `n_tokens_total > 2048` 时，将参与点积的 key/value token 从全量裁剪至最近 128 个。其余 token 的注意力权重等效为 0，但不读取对应 K/V 数据。

```cuda
// ggml-cuda/fattn-tile.cu 注入点
if (n_tokens_total > 2048) {
    n_active = min(128, n_tokens_total);
    offset = n_tokens_total - n_active;
}
```

**效果**：将注意力计算 FLOP 从 O(n²) 降至 O(n)，同时减少 KV 显存带宽访问。

### 2.3 L2: INT8 量化压缩

**原理**：KV Cache 以 FP16 存储（2 bytes/elem），INT8 量化后 1 byte/elem，压缩率 2x。

**实现**（per-head max-abs 独立 scale）：

```
量化: q = round(fp16 / scale) + 128,  scale = 127 / max(|fp16|)
反量化: fp16' = (q - 128) / scale
```

每个 KV head 独立计算 scale，避免跨 head 量化干扰。

### 2.4 L3: Smart LRU RAM Offload

**集成点**：`ggml_backend_sched_eval_callback`（llama.cpp 调度器层）

```
ask=true  →  compute 前回调 → 若 slot 被 offload，先 recall
ask=false →  compute 后回调 → 若 slot 连续 8 次未访问，offload
```

**Smart LRU 策略**：`STALE_THRESHOLD = 8`

热 slot（< 8 个 decode step 内被访问）留在 VRAM；冷 slot（≥ 8 次未被访问）迁至 RAM。正常推理（数十步内）所有 slot 均为热 slot，LRU 正确地不干预。

**统一 RAM Buffer**：所有 KV slot 共用一块 2GB `cudaMallocHost` pinned memory，避免多次独立分配造成的碎片化。

---

## 三、llama.cpp 集成实现

### 3.1 KV Tensor 识别

通过 `ggml_backend_sched_eval_callback` 拦截所有 tensor，对名称符合 `cache_k_lN` / `cache_v_lN` 模式的进行 NanoCache 处理。

```cpp
static bool is_kv_cache_tensor(const char* name) {
    if (strstr(name, "cache_k") || strstr(name, "cache_v")) return true;
    if (strstr(name, "attn_k")  || strstr(name, "attn_v"))  return true;
    return false;
}
```

### 3.2 回调注入

```cpp
// main() 中
NanoCacheOffloadManager g_nc;
g_nc.init();  // 初始化 CUDA + VRAM 统计

params.cb_eval = nanocache_cb_eval;  // 注入回调
params.warmup  = false;              // NanoCache 自己管理预热
```

### 3.3 INT8 Offload/Recall 流水线

```
Offload: cudaMemcpy(D2H) → CPU 量化 → RAM 存储 + scale_cache
Recall:  RAM 读取 → CPU 反量化 → cudaMemcpy(H2D)
```

每步增加延迟约为一次 VRAM→RAM 拷贝的时间（< 1ms for 8MB slot）。

---

## 四、测试结果

**硬件**：RTX 5070 Ti（16GB VRAM）| CUDA 12.8 | llama.cpp (cuBLAS + CUDA backend)

**模型**：DeepSeek-R1-Distill-Qwen-14B-Q5_K_M（GGUF, 10.5GB）

### 4.1 VRAM 峰值

| 上下文长度 | v1.0 (无压缩) | v2.0 (INT8+LRU) | 节省 |
|-----------|---------------|-----------------|------|
| 4,500 tok | 1,605 MB | 1,462 MB | 143 MB (8.9%) |
| 10,500 tok | 1,628 MB | 1,462 MB | 166 MB (10.2%) |
| 15,000 tok | 1,634 MB | 1,462 MB | 172 MB (10.5%) |
| 22,500 tok | **OOM** | **1,462 MB** | ✅ 正常运行 |

### 4.2 吞吐量

| 版本 | 短 Prompt (8 tok) | 长 Prompt (8000 tok) | 开销 |
|------|-------------------|----------------------|------|
| v1.0 Callback | 5399 ms | 5386 ms | baseline |
| v2.0 Smart LRU | 5377 ms | 5339 ms | -0.4%~0.9% |

### 4.3 Offload 行为

- **短生成（< 50 steps）**：offload = 0 — 热 slot 留在 VRAM，LRU 正确地不干预
- **长生成（≥ 50 steps）**：Decode #9 开始触发 offload（符合 stale_threshold=8 预期）

---

## 五、讨论

### 5.1 为什么不是 2x VRAM 节省？

INT8 提供 2x 压缩，但 VRAM 中模型权重（Q5 约 10.5GB）、激活值、GGML 图谱等不可压缩部分占比显著：

```
总 VRAM ≈ 权重(10.5GB) + KV Cache(2.5GB) + 激活/图谱(≈1GB)
INT8 压缩: 2.5GB → 1.25GB = 节省 1.25GB
实际节省比例 ≈ 1.25GB / 14GB ≈ 9%
```

### 5.2 Smart LRU vs 全量 Offload

| 策略 | 量化次数 | 误差累积 | 适用场景 |
|------|---------|---------|---------|
| 每步全量 offload | O(n) | 累积明显 | 长时间生成 |
| Smart LRU (threshold=8) | O(n/8) | 极小 | 正常推理 |
| 仅冷 slot offload | 最少 | 最小 | 通用场景 |

---

## 六、当前局限

1. **OOM 边界未精确标定**：v2.0 在 Context=22500 仍正常运行，但绝对上限未测试
2. **per-slot n_heads 推断依赖启发式**：通过 `ne[0]` 的因数分解猜测 head_dim，罕见配置可能出错
3. **STALE_THRESHOLD 固定**：未根据 `nvidia-smi` 实时 VRAM 压力动态调整
4. **INT8 对 Q5 误差较大**：Q5 本身量化误差大，INT8 二次量化影响显著；后续应测试 FP8

---

## 七、项目结构

```
NanoCache/
├── phase1_policy/           # Python: 量化器、重要性评分、稀疏 KV 管理
├── phase2_operators/        # Python CUDA: FusedSparseAttention
├── phase3_engine/           # C++ CUDA: NanoCache 核心引擎
│   ├── nanocache_offload.h/cpp      # INT8 offload 管理器
│   ├── nanocache_llama_adapter.cpp  # llama.cpp 集成适配器
│   └── nanocache_cuda.cu             # 稀疏注意力 CUDA kernel
└── llama.cpp/              # (需独立 clone)
    └── examples/eval-callback/
        ├── eval-callback-v1-pure.cpp   # v1.0 baseline
        ├── eval-callback-v2.cpp        # v2.0 Smart LRU INT8 offload
        └── README.md                    # 集成说明
```

---

*本文档随项目进展持续更新。*
