"""
NanoCache Phase 1 — 配置

集中管理所有可调参数。
"""

from __future__ import annotations


# ─── L1: 稀疏注意力 ─────────────────────────────────────

SPARSE_THRESHOLD: int = 2048
"""KV 长度超过此值时触发稀疏注意力"""

MAX_ACTIVE_TOKENS: int = 128
"""稀疏注意力窗口大小（只保留最近 N 个 token 参与计算）"""


# ─── L2: INT8 量化 ──────────────────────────────────────

INT8_ENABLED: bool = True
"""是否启用 INT8 量化压缩"""

DEFAULT_HEAD_DIM: int = 128
"""默认 attention head 维度（GQA 场景需正确设置 n_kv_heads）"""


# ─── L3: Smart LRU Offload ──────────────────────────────

STALE_THRESHOLD: int = 8
"""超过 N 个 decode step 未被访问才触发 offload"""

RAM_BUFFER_SIZE_MB: int = 2048
"""统一 RAM buffer 大小（MB），用于存储 offload 的 INT8 数据"""


# ─── 模型配置 ──────────────────────────────────────────

MODEL_CONFIGS = {
    "Qwen2.5-14B": {
        "n_heads":   128,    # Q heads
        "n_kv_heads": 8,     # GQA: 8 KV heads
        "head_dim":  128,
        "n_layers":  56,
    },
    "Qwen2.5-0.5B": {
        "n_heads":   32,
        "n_kv_heads": 32,
        "head_dim":  64,
        "n_layers":  24,
    },
    "DeepSeek-R1-14B": {
        "n_heads":   128,
        "n_kv_heads": 8,
        "head_dim":  128,
        "n_layers":  56,
    },
}


def get_model_config(name: str) -> dict:
    return MODEL_CONFIGS.get(name, {
        "n_heads": 32, "n_kv_heads": 32, "head_dim": 128, "n_layers": 32
    })
