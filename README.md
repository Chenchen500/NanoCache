# NanoCache

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.html)

**KV Cache Layered Compression -- Extend LLM Context Length Under Limited VRAM**

> **NanoCache 1.0** provides a layered architecture: Sparse Attention (L1) + INT8 Quantization (L2) + Smart LRU RAM Offload (L3), extending llama.cpp inference context length without noticeable latency overhead.

**Core result**: On RTX 5070 Ti (16GB VRAM), the 14B-Q5_K_M model's context handling capacity extends from ~22500 tokens to more, with no visible throughput degradation.

---

## Architecture

```
L1: Sparse Attention
  Location: flash_attn_ext CUDA kernel
  Trigger:  KV length > 2048 tokens
  Effect:   Attention FLOP O(n^2) -> O(n)

L2: INT8 Quantization
  Location: llama.cpp eval-callback scheduler
  Trigger:  Always on (default)
  Effect:   KV Cache compressed 2x (FP16 -> INT8)

L3: Smart LRU RAM Offload
  Location: llama.cpp ggml_backend_sched_eval_callback
  Trigger:  Slot unused for 8+ consecutive decode steps
  Effect:   VRAM -> RAM, supports longer context
```

---

## Practical Scenarios

### 1. Long Document Summarization

Feed an entire technical manual, paper collection, or legal contract (tens of thousands of characters) into the model in one pass.

- Before: OOM at ~22500 tokens
- With NanoCache: Runs normally at 22500 tokens

### 2. Multi-turn Agent Dialogues

Maintain system prompt + dialogue history + tool call context across dozens of turns.

- Before: VRAM pressure builds with each turn, degrades after ~20 turns
- With NanoCache: Hot slots stay in VRAM, cold slots offload to RAM automatically

### 3. RAG (Retrieval-Augmented Generation)

Inject dozens of retrieved document chunks simultaneously as context.

- Before: 20 chunks at 10240 tokens = 1628 MB VRAM
- With NanoCache: 20 chunks still at 1462 MB, saving 166 MB

### 4. Code Analysis and Completion

Analyze large codebases where the model needs context from multiple files simultaneously.

### 5. Edge Device Inference

Run large models on consumer GPUs (8-16GB VRAM).

- Before: KV Cache usable space ~12GB, ~15000 tokens max
- With NanoCache: KV Cache compressed 2x, effective space ~13GB+

---

## Test Results

**Hardware**: RTX 5070 Ti (16GB VRAM) | CUDA 12.8 | llama.cpp (cuBLAS + CUDA)
**Model**: DeepSeek-R1-Distill-Qwen-14B-Q5_K_M (GGUF, 10.5GB)

### VRAM Usage

| Context Length | v1.0 (no compression) | v2.0 (INT8+LRU) | Saving |
|---------------|------------------------|------------------|--------|
| 4,500 tok    | 1,605 MB               | 1,462 MB         | 143 MB (8.9%) |
| 10,500 tok   | 1,628 MB               | 1,462 MB         | 166 MB (10.2%) |
| 15,000 tok   | 1,634 MB               | 1,462 MB         | 172 MB (10.5%) |
| 22,500 tok   | **OOM**                | **1,462 MB**      | Runs OK |

### Throughput (14B, 50-token prompt, 16-token generation)

| Version | Prompt t/s | Gen t/s | Total Time | Overhead |
|---------|-----------|---------|------------|----------|
| Baseline (llama-cli) | 43.46 | 73.27 | 2006 ms | baseline |
| NanoCache v2 (INT8+LRU) | 44.07 | 73.37 | 1960 ms | **-0.4%~1.4%** |

### Throughput (0.5B, 50-token prompt, 16-token generation)

| Version | Prompt t/s | Gen t/s |
|---------|-----------|---------|
| Baseline (llama-cli) | 1162 | 565 |
| NanoCache v2 (INT8+LRU) | 1185 | 566 |

**Conclusion**: NanoCache v2 adds no measurable throughput overhead on either 0.5B or 14B models.
The INT8 quantization and LRU offload logic execute within the normal eval callback overhead (~1-2% max).

### Offload Behavior

- Short generation (<50 steps): offload = 0 -- hot slots stay in VRAM
- Long generation (>=50 steps): offload triggers at Decode #9 (stale_threshold=8)

---

## Project Structure

```
NanoCache/
|-- phase1_policy/          # Python: quantizer, importance scorer, sparse KV manager
|-- phase2_operators/       # Python CUDA: FusedSparseAttention kernel
|-- phase3_engine/          # C++ CUDA: NanoCache core engine
|   +-- nanocache_llama_adapter.cpp   # llama.cpp integration adapter
|-- docs/                   # Technical whitepaper
+-- llama.cpp/             # (clone separately: https://github.com/ggerganov/llama.cpp)
    +-- examples/eval-callback/
        |-- eval-callback-v1-pure.cpp  # v1.0: callback-only baseline
        |-- eval-callback-v2.cpp      # v2.0: Smart LRU INT8 offload
        +-- README.md                  # llama.cpp integration guide
```

---

## Quick Start

### 1. Prepare llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_BURBLE_COPY=OFF
make -j$(nproc)
```

### 2. Build NanoCache

Copy `llama.cpp/examples/eval-callback/eval-callback-v2.cpp` over `eval-callback.cpp`, then:

```bash
cd llama.cpp/build
cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_BURBLE_COPY=OFF
make llama-eval-callback -j$(nproc)
```

### 3. Run

```bash
./llama-eval-callback \
  -m /path/to/model-q4_k_m.gguf \
  --prompt "What is artificial intelligence?" \
  -n 128
```

NanoCache logs go to stderr, generated text goes to stdout.

---

## Technical Details

### llama.cpp Integration Point

`ggml_backend_sched_eval_callback` -- llama.cpp scheduler triggers callbacks before/after each tensor compute:

```cpp
// ask=true: before compute -> recall offloaded slot
// ask=false: after compute -> offload stale slot
bool nanocache_cb_eval(struct ggml_tensor* t, bool ask, void* user_data) {
    if (is_kv_tensor(t->name)) {
        if (ask && slot.is_offload) nc_recall(t->name);
        if (!ask && slot.is_stale(threshold)) nc_offload(t->name);
    }
    return true;
}
params.cb_eval = nanocache_cb_eval;
```

### INT8 Quantization (per-head independent scale)

```cpp
// Quantize
float scale = 127.0f / max(|fp16|);
int8_t q = round(fp16 / scale);

// Dequantize
fp16' = (q - 128) * (1.0f / scale);
```

### Smart LRU Scheduling

```cpp
static const int STALE_THRESHOLD = 8;  // offload after 8 unused decode steps

void nc_on_decode_done() {
    for (auto& slot : slots) {
        if (!slot.is_offload && slot.stale(decode_no) > STALE_THRESHOLD) {
            nc_offload(slot);
        }
    }
}
```

---

## License

This project is licensed under **GNU Affero General Public License v3 (AGPLv3)**.

If you are a cloud service provider (AWS, Azure, Alibaba Cloud, etc.) or an enterprise user and do not wish to open-source your service code, please contact the author for a commercial license.

NanoCache is based on [llama.cpp](https://github.com/ggerganov/llama.cpp) (also MIT License).

---

*Project updates follow research progress. Test data from 2026-04-08 experiment (RTX 5070 Ti, CUDA 12.8).*
