"""
NanoCache Phase 1 — INT8 Per-Head Quantizer

将 KV Cache 从 FP16 量化为 INT8，压缩率 2x。

量化公式（per-head max-abs 量化）:
    scale = 127.0 / max(|fp16|)
    q     = round(fp16 * scale) + 128   → [0, 255]

反量化:
    fp16' = (q - 128) / scale
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


class Int8Quantizer:
    """Per-head INT8 量化器"""

    def __init__(self, n_kv_heads: int, head_dim: int, n_tokens: int):
        self.n_kv_heads = n_kv_heads
        self.head_dim  = head_dim
        self.n_tokens  = n_tokens
        self.n_elements = n_kv_heads * head_dim * n_tokens

    def quantize(self, kv_fp16: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 KV cache FP16 转为 INT8

        Args:
            kv_fp16: shape (n_kv_heads, n_tokens, head_dim), dtype float16

        Returns:
            kv_int8: shape 同上, dtype uint8, 范围 [0, 255]
            scales:  shape (n_kv_heads, n_tokens), dtype float32
        """
        assert kv_fp16.shape == (self.n_kv_heads, self.n_tokens, self.head_dim), \
            f"expected {(self.n_kv_heads, self.n_tokens, self.head_dim)}, got {kv_fp16.shape}"

        kv_int8 = np.empty_like(kv_fp16).astype(np.uint8)
        scales  = np.empty((self.n_kv_heads, self.n_tokens), dtype=np.float32)

        for h in range(self.n_kv_heads):
            head_data = kv_fp16[h]  # (n_tokens, head_dim)
            mx = np.abs(head_data).max()
            scale = 127.0 / mx if mx > 1e-5 else 1.0
            scales[h] = scale

            q = np.round(head_data * scale).astype(np.int32) + 128
            np.clip(q, 0, 255, out=q)
            kv_int8[h] = q.astype(np.uint8)

        return kv_int8, scales

    def dequantize(self, kv_int8: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        将 INT8 反量化为 FP16

        Args:
            kv_int8: shape (n_kv_heads, n_tokens, head_dim), dtype uint8
            scales:  shape (n_kv_heads, n_tokens), dtype float32

        Returns:
            kv_fp16: shape 同上, dtype float16
        """
        assert kv_int8.shape == (self.n_kv_heads, self.n_tokens, self.head_dim)
        assert scales.shape  == (self.n_kv_heads, self.n_tokens)

        kv_fp16 = np.empty_like(kv_int8).astype(np.float32)
        for h in range(self.n_kv_heads):
            inv = 1.0 / scales[h]  # (n_tokens,)
            head_int8 = kv_int8[h].astype(np.int32) - 128  # (n_tokens, head_dim)
            kv_fp16[h] = head_int8.astype(np.float32) * inv[:, np.newaxis]

        return kv_fp16.astype(np.float16)

    def vrams_savings(self, n_layers: int) -> Tuple[float, float]:
        """
        计算 VRAM 节省量

        Returns:
            fp16_bytes, int8_bytes
        """
        fp16 = self.n_elements * 2 * n_layers
        i8   = self.n_elements * 1 * n_layers
        return fp16, i8
