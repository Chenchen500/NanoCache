/**
 * NanoCache 2.0 llama.cpp 集成 — 核心实现
 *
 * 集成方式:
 *   params.cb_eval = nanocache_cb_eval
 *
 * 每次 llama_decode 完成后（ask=false 时机），
 * 对所有 K/V cache tensor 执行 INT8 量化 + offload → RAM
 * 下次 llama_decode 开始前，recall → VRAM
 */

#include "nanocache_offload.h"
#include "llama.h"

#include <cuda_runtime.h>
#include <clocale>
#include <cstring>
#include <cmath>
#include <stdio.h>

// ============================================================
// NanoCache Offload Manager — 实现
// ============================================================

NanoCacheOffloadManager g_nc;

bool NanoCacheOffloadManager::init() {
    cudaError_t e = cudaGetDeviceCount(reinterpret_cast<int*>(&n_heads));
    if (e != cudaSuccess) {
        fprintf(stderr, "[NanoCache] CUDA init failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    vram_total = prop.totalGlobalMem;
    cuda_ok = true;
    initialized = true;
    fprintf(stderr, "[NanoCache] GPU: %s, VRAM: %.1f GB\n",
            prop.name, (float)vram_total / (1024 * 1024 * 1024));
    return true;
}

bool NanoCacheOffloadManager::register_tensor(const char* tensor_name, void* vram_ptr,
                                              size_t n_elements, int layer, bool is_k) {
    if (!cuda_ok || !vram_ptr) return false;
    std::string key = tensor_name;
    if (slots.find(key) != slots.end()) return false;

    KVCacheSlot s = {};
    s.vram_ptr   = vram_ptr;
    s.n_elements = n_elements;
    s.layer      = layer;
    s.is_k       = is_k;
    s.n_tokens   = n_elements / (n_kv_heads * head_dim);
    s.size_fp16  = n_elements * 2;   // FP16 = 2 bytes
    s.size_int8  = n_elements;       // INT8 = 1 byte

    // 分配 RAM INT8 缓冲区
    CUDA_CHECK(cudaMallocHost(&s.ram_int8, s.size_int8));
    CUDA_CHECK(cudaMallocHost(&s.scales, n_kv_heads * s.n_tokens * sizeof(float)));

    slots[key] = s;
    fprintf(stderr,
            "[NanoCache] Registered: %s (L%d, %s, %zu tok, %.1f KB FP16)\n",
            tensor_name, layer, is_k ? "K" : "V",
            s.n_tokens, (float)s.size_fp16 / 1024);
    return true;
}

bool NanoCacheOffloadManager::offload_slot(const std::string& key) {
    auto it = slots.find(key);
    if (it == slots.end() || it->second.is_offloaded || !it->second.vram_ptr)
        return false;

    KVCacheSlot& s = it->second;

    // 分配临时 host buffer
    float* h_fp16 = (float*)malloc(s.size_fp16);
    CUDA_CHECK(cudaMemcpy(h_fp16, s.vram_ptr, s.size_fp16, cudaMemcpyDeviceToHost));

    // Per-head max-abs INT8 量化
    for (int h = 0; h < n_kv_heads; h++) {
        float mx = 0.0f;
        for (size_t t = 0; t < s.n_tokens; t++) {
            size_t base = (t * n_kv_heads + h) * head_dim;
            for (int d = 0; d < head_dim; d++) {
                float v = fabsf(h_fp16[base + d]);
                if (v > mx) mx = v;
            }
        }
        float sc = (mx > 1e-5f) ? (127.0f / mx) : 1.0f;
        s.scales[h] = sc;
        for (size_t t = 0; t < s.n_tokens; t++) {
            size_t base = (t * n_kv_heads + h) * head_dim;
            for (int d = 0; d < head_dim; d++) {
                float v = h_fp16[base + d] * sc;
                if (v > 127.0f) v = 127.0f;
                else if (v < -127.0f) v = -127.0f;
                s.ram_int8[base + d] = (uint8_t)(v + 128.0f);
            }
        }
    }

    free(h_fp16);
    s.is_offloaded = true;
    offload_count++;
    fprintf(stderr,
            "[NanoCache] OFFLOAD: %s (FP16 %.1f KB → INT8 %.1f KB)\n",
            key.c_str(), (float)s.size_fp16 / 1024, (float)s.size_int8 / 1024);
    return true;
}

bool NanoCacheOffloadManager::recall_slot(const std::string& key) {
    auto it = slots.find(key);
    if (it == slots.end() || !it->second.is_offloaded || !it->second.vram_ptr)
        return false;

    KVCacheSlot& s = it->second;
    float* h_fp16 = (float*)malloc(s.size_fp16);

    for (int h = 0; h < n_kv_heads; h++) {
        float inv = 1.0f / s.scales[h];
        for (size_t t = 0; t < s.n_tokens; t++) {
            size_t base = (t * n_kv_heads + h) * head_dim;
            for (int d = 0; d < head_dim; d++) {
                int8_t q = (int8_t)(s.ram_int8[base + d] - 128);
                h_fp16[base + d] = (float)q * inv;
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(s.vram_ptr, h_fp16, s.size_fp16, cudaMemcpyHostToDevice));
    free(h_fp16);

    s.is_offloaded = false;
    recall_count++;
    fprintf(stderr, "[NanoCache] RECALL:  %s (INT8 → FP16 restored)\n", key.c_str());
    return true;
}

void NanoCacheOffloadManager::on_seen_tensor(const char* name, void* ptr,
                                              size_t n_elements) {
    if (!initialized || !cuda_ok) return;
    int layer = extract_layer(name);
    if (layer < 0) return;
    bool is_k = is_k_from_name(name);
    bool is_v = is_v_from_name(name);
    if (!is_k && !is_v) return;

    std::string key = name;
    if (slots.find(key) == slots.end()) {
        register_tensor(name, ptr, n_elements, layer, is_k);
    } else {
        KVCacheSlot& s = slots[key];
        if (s.vram_ptr != ptr) {
            fprintf(stderr, "[NanoCache] KV tensor %s relocated\n", name);
            s.vram_ptr = ptr;
        }
    }
}

void NanoCacheOffloadManager::on_decode_done() {
    if (!initialized) return;
    decode_count++;

    fprintf(stderr,
            "\n[NanoCache] Decode #%d done: triggering offload for %zu KV slots...\n",
            decode_count, slots.size());

    int off = 0;
    for (auto& kv : slots) {
        if (!kv.second.is_offloaded) {
            if (offload_slot(kv.first)) off++;
        }
    }
    fprintf(stderr,
            "[NanoCache] Offload done: %d slots offloaded "
            "(total: %llu off / %llu recall)\n\n",
            off,
            (unsigned long long)offload_count,
            (unsigned long long)recall_count);
}

void NanoCacheOffloadManager::on_decode_start() {
    if (!initialized) return;
    fprintf(stderr,
            "\n[NanoCache] Decode #%d starting: recalling %zu slots...\n",
            decode_count + 1, slots.size());

    int recalled = 0;
    for (auto& kv : slots) {
        if (kv.second.is_offloaded) {
            if (recall_slot(kv.first)) recalled++;
        }
    }
    fprintf(stderr,
            "[NanoCache] Recall done: %d slots restored\n\n",
            recalled);
}

void NanoCacheOffloadManager::print_stats() const {
    fprintf(stderr, "\n[NanoCache] =========== Final Stats ===========\n");
    fprintf(stderr, "  Offload count: %llu\n",  (unsigned long long)offload_count);
    fprintf(stderr, "  Recall count:  %llu\n",  (unsigned long long)recall_count);
    fprintf(stderr, "  Slots tracked: %zu\n",   slots.size());
    fprintf(stderr, "===========================================\n\n");
}

// ============================================================
// NanoCache eval callback
// ============================================================

bool nanocache_cb_eval(struct ggml_tensor* t, bool ask, void* /* user_data */) {
    if (!g_nc.initialized) return true;

    if (is_kv_cache_tensor(t->name)) {
        if (ask) {
            // Decode 前：recall 被 offload 的 slot
            std::string key = t->name;
            auto it = g_nc.slots.find(key);
            if (it != g_nc.slots.end() && it->second.is_offloaded) {
                g_nc.recall_slot(key);
            }
            // 注册新发现的 tensor
            g_nc.on_seen_tensor(t->name, t->data, t->ne[0] * t->ne[1]);
            return true;
        } else {
            // Decode 后：不做 individual offload，由 llama_decode 结束后统一处理
            return true;
        }
    }
    return true;
}
