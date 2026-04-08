"""
NanoCache Phase 2 — Fused Sparse Attention CUDA Kernel

Python wrapper around the raw CUDA kernel (nanocache_cuda.cu)。
提供高层接口：给定 Q/K/V，计算稀疏注意力输出。

稀疏策略（L1）:
    当 n_keys > SPARSE_THRESHOLD (2048) 时，
    K/V 只取最近 MAX_ACTIVE_TOKENS (128) 个参与计算。

这不是独立运行的文件——由 phase3_engine/nanocache_cuda.cu
通过 pybind11 或 ctypes 调用。
"""

from __future__ import annotations
import numpy as np
from typing import Optional

# 常量（与 nanocache_cuda.cu 保持一致）
SPARSE_THRESHOLD   = 2048
MAX_ACTIVE_TOKENS  = 128


def compute_sparse_mask(n_tokens: int) -> tuple[int, int]:
    """
    确定给定 n_tokens 是否触发稀疏，以及活跃 token 范围。

    Returns:
        (offset, n_active):
            offset: 从 n_tokens - n_active 开始
            n_active: 实际参与计算的 token 数
    """
    if n_tokens <= SPARSE_THRESHOLD:
        return 0, n_tokens
    return n_tokens - MAX_ACTIVE_TOKENS, MAX_ACTIVE_TOKENS


def sparse_attention_mask(n_keys: int, n_queries: int) -> np.ndarray:
    """
    构建稀疏注意力 mask 矩阵。

    Args:
        n_keys: KV cache 中总 token 数
        n_queries: 查询 token 数

    Returns:
        mask: shape (n_queries, n_keys), bool
               True = 参与注意力计算
    """
    offset, n_active = compute_sparse_mask(n_keys)
    mask = np.zeros((n_queries, n_keys), dtype=bool)
    mask[:, offset:offset + n_active] = True
    return mask


def estimate_flops_saved(n_keys: int, n_queries: int, head_dim: int) -> float:
    """
    估算稀疏注意力节省的 FLOP。

    标准 attention:  n_queries × n_keys × head_dim × 2
    稀疏 attention:   n_queries × n_active × head_dim × 2

    Returns:
        节省比例 (0.0 ~ 1.0)
    """
    offset, n_active = compute_sparse_mask(n_keys)
    full_flops    = n_queries * n_keys        * head_dim * 2
    sparse_flops  = n_queries * n_active     * head_dim * 2
    return 1.0 - (sparse_flops / full_flops)


class FusedSparseAttention:
    """
    Python 前端：用 NumPy 模拟 fused sparse attention kernel 的行为。
    实际 CUDA kernel 在 nanocache_cuda.cu，通过 ctypes 调用。

    这个类的存在是为了在没有 CUDA 的环境下也能验证算法逻辑。
    """

    def __init__(self, n_heads: int, head_dim: int, dtype=np.float16):
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self.dtype    = dtype

    def forward(
        self,
        Q: np.ndarray,   # (n_queries, n_heads, head_dim)
        K: np.ndarray,   # (n_keys_total, n_kv_heads, head_dim)
        V: np.ndarray,   # (n_keys_total, n_kv_heads, head_dim)
    ) -> np.ndarray:
        """
        计算稀疏注意力输出。

        当 n_keys_total > 2048 时，只取 K/V 最近 128 个 token。
        """
        n_queries  = Q.shape[0]
        n_keys_tot  = K.shape[0]
        n_kv_heads  = K.shape[1]

        offset, n_active = compute_sparse_mask(n_keys_tot)

        # 裁剪 K/V 到活跃窗口
        K_sparse = K[offset:offset + n_active]   # (n_active, n_kv_heads, head_dim)
        V_sparse = V[offset:offset + n_active]

        # 广播 Q 到各 KV head（GQA 场景）
        # Q: (n_queries, n_heads, head_dim)
        # K_sparse: (n_active, n_kv_heads, head_dim)
        # 输出: (n_queries, n_heads, head_dim)

        # 简化：每 group 有 n_heads/n_kv_heads 个 query head 共享一个 KV head
        group_size = self.n_heads // n_kv_heads
        O = np.zeros((n_queries, self.n_heads, self.head_dim), dtype=self.dtype)

        for h in range(n_kv_heads):
            for g in range(group_size):
                qh = h * group_size + g
                Qh = Q[:, qh, :]  # (n_queries, head_dim)

                # Q · K^T
                qk = np.einsum("qd,kd->qk", Qh, K_sparse[:, h, :])  # (n_queries, n_active)
                qk = qk - qk.max(axis=-1, keepdims=True)  # 数值稳定 softmax
                a  = np.exp(qk)
                a_sum = a.sum(axis=-1, keepdims=True) + 1e-8
                a_norm = a / a_sum

                # 加权 V
                ov = np.einsum("qk,kd->qd", a_norm, V_sparse[:, h, :])  # (n_queries, head_dim)
                O[:, qh, :] = ov

        return O
