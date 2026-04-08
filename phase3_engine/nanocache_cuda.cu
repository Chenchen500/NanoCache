/**
 * NanoCache Sparse Attention CUDA Kernel
 *
 * L1 层核心：在 flash_attn_ext 内核中，当 KV 长度 > 2048 时，
 * 将参与点积的 key/value token 从全量裁剪至最近 128 个。
 *
 * 数学上等价于将其他 token 的注意力权重置零，但避免了读取
 * 那些 token 的 K/V 数据，节省显存带宽。
 *
 * 集成方式（ggml-cuda/fattn-tile.cu 补丁）:
 *   if (n_tokens_total > 2048) {
 *       n_active = min(128, n_tokens_total);
 *       offset = n_tokens_total - n_active;
 *   }
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_KERNEL __global__
#define CUDA_CHECK(call)                                                       \
    do {                                                                      \
        cudaError_t e = (call);                                               \
        if (e != cudaSuccess)                                                  \
            fprintf(stderr, "[NanoCache CUDA] %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e));                                    \
    } while (0)

namespace nanocache {

// ============================================================
// Config
// ============================================================

static const int THREADS_PER_BLOCK = 256;
static const int WARP_SIZE = 32;
static const int MAX_SPARSE_TOKENS = 128;   // 稀疏注意力窗口大小
static const int SPARSE_THRESHOLD = 2048;  // 超过此长度触发稀疏

// ============================================================
// Sparse Attention Kernel
//
// 对每个 query token，计算与最近 MAX_SPARSE_TOKENS 个 key 的点积
// (而非全量 n_keys 个)，大幅减少显存读取和计算量。
// ============================================================

template <typename T>
__global__
void sparse_attn_fwd_kernel(
    const T* __restrict__ Q,    // [n_queries, n_heads, head_dim]
    const T* __restrict__ K,    // [n_keys_total, n_heads, head_dim]
    const T* __restrict__ V,    // [n_keys_total, n_heads, head_dim]
    float* __restrict__ O,     // [n_queries, n_heads, head_dim]
    float* __restrict__ L,     // [n_queries, n_heads] (logsumexp for backward)
    int n_queries,
    int n_keys_total,
    int n_heads,
    int head_dim,
    int n_active_keys        // = min(MAX_SPARSE_TOKENS, n_keys_total)
) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= n_queries * n_heads) return;

    int q = q_idx / n_heads;
    int h = q_idx % n_heads;

    const T* q_ptr = Q + (q * n_heads + h) * head_dim;
    const T* k_base = K + h * head_dim;           // 每个 head 一个 kv_head
    const T* v_base = V + h * head_dim;

    // 起始位置：从最近 n_active_keys 个 key 开始
    int k_start = n_keys_total - n_active_keys;

    // 计算 Q · K 点积（仅最近 n_active_keys 个）
    float qk_max = -1e9f;
    float qk_sums[4] = {0.f, 0.f, 0.f, 0.f};
    int vec_len = (head_dim + 3) / 4;

    for (int kk = k_start; kk < n_keys_total; kk += 4) {
        int remaining = n_keys_total - kk;
        int vec_elems = min(4, remaining);

        float qk_local[4] = {0.f, 0.f, 0.f, 0.f};
        for (int d = 0; d < head_dim; d++) {
            float qd = (float)q_ptr[d];
            for (int l = 0; l < vec_elems; l++) {
                float kd = (float)k_base[kk + l * n_heads * head_dim + d];
                qk_local[l] += qd * kd;
            }
        }
        for (int l = 0; l < 4; l++) {
            qk_local[l] = expf(qk_local[l] - qk_max);
            qk_sums[l] += qk_local[l];
        }
    }

    float qk_sum_total = qk_sums[0] + qk_sums[1] + qk_sums[2] + qk_sums[3];
    float qk_max_new = qk_max;
    // logsumexp
    float lse = logf(qk_sum_total) + qk_max_new;

    // 计算加权 V 求和
    float* o_ptr = O + (q * n_heads + h) * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float val = 0.f;
        for (int kk = k_start; kk < n_keys_total; kk++) {
            float qk = 0.f;
            for (int dd = 0; dd < head_dim; dd++) {
                qk += (float)q_ptr[dd] * (float)k_base[kk * n_heads * head_dim + dd];
            }
            qk = expf(qk - lse);
            val += qk * (float)v_base[kk * n_heads * head_dim + d];
        }
        o_ptr[d] = val;
    }

    if (L) L[q * n_heads + h] = lse;
}

// ============================================================
// 辅助: 确定是否触发稀疏
// ============================================================

inline bool should_use_sparse(int n_keys_total) {
    return n_keys_total > SPARSE_THRESHOLD;
}

inline int get_active_keys(int n_keys_total) {
    if (n_keys_total <= SPARSE_THRESHOLD) return n_keys_total;
    return MAX_SPARSE_TOKENS;
}

// ============================================================
// Host wrapper
// ============================================================

template <typename T>
bool sparse_attn_fwd(
    const T* Q,
    const T* K,
    const T* V,
    float* O,
    float* L,
    int n_queries,
    int n_keys_total,
    int n_heads,
    int head_dim,
    cudaStream_t stream
) {
    int n_active = get_active_keys(n_keys_total);

    if (!should_use_sparse(n_keys_total)) {
        // 标准 attention（不裁剪），调用方走常规 flash_attn
        return false;
    }

    int n_blocks = (n_queries * n_heads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sparse_attn_fwd_kernel<T><<<n_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        Q, K, V, O, L,
        n_queries, n_keys_total, n_heads, head_dim,
        n_active
    );

    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ============================================================
// VRAM 统计
// ============================================================

struct VRAMStats {
    size_t total;
    size_t free;
    size_t used_nc;     // NanoCache 管理的
};

bool get_vram_stats(VRAMStats* stats) {
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    stats->total = total;
    stats->free  = free;
    stats->used_nc = total - free;
    return true;
}

}  // namespace nanocache

// ============================================================
// Explicit instantiation
// ============================================================

template bool nanocache::sparse_attn_fwd<float>(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    int n_queries, int n_keys_total, int n_heads, int head_dim,
    cudaStream_t stream
);

template bool nanocache::sparse_attn_fwd<__half>(
    const __half* Q, const __half* K, const __half* V,
    float* O, float* L,
    int n_queries, int n_keys_total, int n_heads, int head_dim,
    cudaStream_t stream
);
