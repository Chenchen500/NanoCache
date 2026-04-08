"""
NanoCache Phase 1 — 重要性评分器

决定在稀疏注意力窗口外，哪些 token 应该被优先保留。
评分策略：
  1. 近期性（recency）：越近期的 token 越重要
  2. 注意力权重（attention）：被频繁引用的 token 更重要

NanoCache 采用近期性策略（实现简单、无需反向传播），
但记录注意力分数供实验对比。
"""

from __future__ import annotations
import numpy as np
from typing import List


class ImportanceScorer:
    """
    评估每个历史 token 的重要性，供稀疏 KV 保留决策使用。

    当前策略: 指数衰减近期性
        score[t] = exp(-alpha * (n_tokens - 1 - t))

    阈值触发后，保留分数最高的 MAX_KEEP 个 token。
    """

    def __init__(self, alpha: float = 0.05, max_keep: int = 128):
        """
        Args:
            alpha: 衰减系数，越大则早期 token 衰减越快
            max_keep: 稀疏窗口大小（保留的重要 token 数）
        """
        self.alpha    = alpha
        self.max_keep = max_keep
        self._n_tokens = 0

    def score_all(self, n_tokens: int) -> np.ndarray:
        """
        对 [0, n_tokens) 每个位置计算重要性分数。

        分数随 token 变老指数衰减。

        Returns:
            scores: shape (n_tokens,), dtype float32
        """
        t = np.arange(n_tokens, dtype=np.float32)
        scores = np.exp(-self.alpha * (n_tokens - 1 - t))
        self._n_tokens = n_tokens
        return scores

    def topk_mask(self, scores: np.ndarray, k: int = None) -> np.ndarray:
        """
        返回保留 mask（分数最高的 k 个位置为 True）。

        Returns:
            mask: shape (n_tokens,), dtype bool
        """
        k = k or self.max_keep
        mask = np.zeros_like(scores, dtype=bool)
        idx = np.argsort(scores)[-k:]  # 分数最高的 k 个
        mask[idx] = True
        return mask

    def merge_attention_scores(
        self,
        attn_weights: np.ndarray,
        decay: float = 0.9
    ) -> np.ndarray:
        """
        可选：将注意力权重合并到分数中。

        attn_weights: shape (seq_len,), 每个历史位置在最后一次被注意的权重
        """
        base = self.score_all(len(attn_weights))
        # 注意力权重加成（加权几何平均）
        combined = np.power(base, decay) * np.power(attn_weights + 1e-8, 1 - decay)
        return combined

    def decide_retention(
        self,
        n_tokens: int,
        threshold: int = 2048
    ) -> np.ndarray:
        """
        决策：给定当前 token 总数，决定哪些位置应保留在 KV cache。

        Args:
            n_tokens: 当前 KV cache 中的 token 总数
            threshold: 超过此长度才开始稀疏（2048）

        Returns:
            keep_mask: shape (n_tokens,), bool
        """
        if n_tokens <= threshold:
            # 全保留
            return np.ones(n_tokens, dtype=bool)

        scores = self.score_all(n_tokens)
        return self.topk_mask(scores)
