/**
 * NanoCache 2.0 llama.cpp 集成
 *
 * 集成方式:
 *   params.cb_eval = nanocache_cb_eval
 *
 * 每次 llama_decode 完成后（ask=false 时机），
 * 对所有 K/V cache tensor 执行 INT8 量化 + offload → RAM
 * 下次 llama_decode 开始前，recall → VRAM
 */

#pragma once

#include "llama.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <map>
#include <set>

// ============================================================
// CUDA Helpers
// ============================================================

#define CUDA_CHECK(call)                                                      \
    do {                                                                     \
        cudaError_t e = (call);                                              \
        if (e != cudaSuccess) {                                              \
            fprintf(stderr, "[NanoCache CUDA] %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e));                                   \
        }                                                                    \
    } while (0)

// ============================================================
// KV Cache Slot
// ============================================================

struct KVCacheSlot {
    void*   vram_ptr    = nullptr;   // VRAM 地址 (tensor->data)
    uint8_t* ram_int8  = nullptr;   // RAM INT8 副本
    float*  scales     = nullptr;   // per-head scale factors

    size_t  n_elements = 0;         // 元素总数
    size_t  n_tokens   = 0;         // token 数
    size_t  size_fp16  = 0;        // FP16 字节数
    size_t  size_int8  = 0;        // INT8 字节数

    int     layer      = -1;
    bool    is_k       = false;
    bool    is_offloaded = false;  // 当前是否在 RAM
};

// ============================================================
// NanoCache Offload Manager
// ============================================================

struct NanoCacheOffloadManager {
    int     n_heads    = 128;   // Qwen2.5-14B: 128 Q heads, 8 KV heads (GQA)
    int     n_kv_heads = 8;
    int     head_dim   = 128;
    int     decode_count = 0;

    std::map<std::string, KVCacheSlot> slots;

    uint64_t offload_count = 0;
    uint64_t recall_count  = 0;

    bool     initialized = false;
    bool     cuda_ok    = false;
    size_t   vram_total = 0;

    bool init();
    bool register_tensor(const char* name, void* ptr, size_t n_elements,
                         int layer, bool is_k);
    bool offload_slot(const std::string& key);
    bool recall_slot(const std::string& key);
    void on_seen_tensor(const char* name, void* ptr, size_t n_elements);
    void on_decode_done();
    void on_decode_start();
    void print_stats() const;
};

// ============================================================
// Global instance
// ============================================================

extern NanoCacheOffloadManager g_nc;

// ============================================================
// Tensor classification helpers
// ============================================================

static inline int extract_layer(const char* name) {
    const char* patterns[] = {"blk.", "layer.", ".layer_"};
    for (auto p : patterns) {
        const char* pos = strstr(name, p);
        if (pos) {
            int v = atoi(pos + strlen(p));
            if (v >= 0) return v;
        }
    }
    return -1;
}

static inline bool is_k_from_name(const char* name) {
    if (strstr(name, "cache_k")) return true;
    if (strstr(name, "attn_k") && !strstr(name, "attn_kv")) return true;
    if (strstr(name, ".k") && strstr(name, "attn")) return true;
    if (strstr(name, "_k_")) return true;
    return false;
}

static inline bool is_v_from_name(const char* name) {
    if (strstr(name, "cache_v")) return true;
    if (strstr(name, "attn_v") && !strstr(name, "attn_kv")) return true;
    if (strstr(name, ".v") && strstr(name, "attn")) return true;
    if (strstr(name, "_v_")) return true;
    return false;
}

static inline bool is_kv_cache_tensor(const char* name) {
    if (!name) return false;
    if (strstr(name, "cache_k") || strstr(name, "cache_v")) return true;
    if (strstr(name, "attn_k")  || strstr(name, "attn_v"))  return true;
    if (strstr(name, ".kv") || strstr(name, "_kv_"))       return true;
    return false;
}

// ============================================================
// eval callback
// ============================================================

bool nanocache_cb_eval(struct ggml_tensor* t, bool ask, void* user_data);
