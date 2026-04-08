"""
NanoCache Phase 1 — 稀疏 KV Cache 管理器

在 KV Cache 长度超过 SPARSE_THRESHOLD 时，
将活跃 token 数裁剪至 MAX_ACTIVE，保持稀疏注意力窗口。

策略:
  L1: flash_attn_ext CUDA 内核级稀疏（实际计算层面）
  L1+: Python 层管理 slot 元数据，跟踪哪些 KV slot 正在被使用

NanoCache 的 L1 实现是内核级的（在 llama.cpp 的 fattn-tile.cu 中），
这个管理器记录每层的 slot 状态，用于 offload 决策。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import time


@dataclass
class KVSlot:
    """单个 KV slot 的元数据"""
    layer:     int
    is_k:      bool       # True=K, False=V
    slot_id:   int        # 在该层的位置（token index）
    n_tokens:  int        # 该 slot 覆盖的 token 数
    is_active: bool = True # 当前是否在计算路径中
    last_used: float = field(default_factory=time.time)
    is_offloaded: bool = False  # 是否已 offload 到 RAM


class SparseKVManager:
    """
    管理所有层的 KV slot，支持：

    1. 注册新 slot（forward 发现新的 KV tensor）
    2. 标记活跃 slot（forward 期间被引用）
    3. 触发稀疏裁剪（超过阈值时）
    4. LRU offload 决策（冷 slot offload 到 RAM）
    """

    # KV cache 超过此长度开始稀疏注意力
    SPARSE_THRESHOLD = 2048

    # 稀疏注意力窗口大小（内核级裁剪到 128）
    MAX_ACTIVE_TOKENS = 128

    # 超过此数量 decode step 未被访问，触发 offload
    STALE_THRESHOLD = 8

    def __init__(self):
        # (layer, is_k) -> list of KVSlot
        self._slots: Dict[tuple, List[KVSlot]] = {}
        self._decode_count = 0
        self._seen_this_decode: Set[tuple] = set()

    # ─── 注册 ───────────────────────────────────────────────

    def register(self, layer: int, is_k: bool, slot_id: int, n_tokens: int):
        key = (layer, is_k)
        if key not in self._slots:
            self._slots[key] = []
        self._slots[key].append(KVSlot(
            layer=layer, is_k=is_k,
            slot_id=slot_id, n_tokens=n_tokens
        ))

    # ─── Forward 跟踪 ───────────────────────────────────────

    def on_seen(self, layer: int, is_k: bool):
        """Forward 期间看到某个 slot，标记为活跃"""
        key = (layer, is_k)
        self._seen_this_decode.add(key)

    def on_decode_done(self) -> List[tuple]:
        """
        Decode 结束后，返回需要 offload 的 slot 列表。

        策略：
          - 所有本轮未被 on_seen 标记的 slot → 候选 offload
          - 如果当前总 slot 数 > SPARSE_THRESHOLD，连同最早的 slot 一起 offload

        Returns:
            [(layer, is_k), ...] 待 offload 的 slot
        """
        self._decode_count += 1
        to_offload = []

        # 候选：STALE_THRESHOLD 个 decode 未被访问
        for key, slots in self._slots.items():
            for s in slots:
                if not s.is_active:
                    if self._decode_count - s.last_used_decode >= self.STALE_THRESHOLD:
                        to_offload.append(key)

        # 强制稀疏：如果总 KV 长度超标，offload 最老的
        total = self.total_slots()
        if total > self.SPARSE_THRESHOLD:
            oldest = self._oldest_slots(total - self.SPARSE_THRESHOLD + self.MAX_ACTIVE_TOKENS)
            to_offload.extend(oldest)

        # 重置活跃标记
        for slots in self._slots.values():
            for s in slots:
                s.is_active = (key in self._seen_this_decode for key in [slots])
        self._seen_this_decode.clear()

        return list(set(to_offload))

    # ─── 查询 ───────────────────────────────────────────────

    def total_slots(self) -> int:
        return sum(len(v) for v in self._slots.values())

    def active_count(self) -> int:
        return sum(1 for slots in self._slots.values() for s in slots if s.is_active)

    def stale_count(self) -> int:
        return sum(
            1 for slots in self._slots.values()
            for s in slots
            if not s.is_active and self._decode_count - s.last_used_decode >= self.STALE_THRESHOLD
        )

    def stats(self) -> dict:
        return {
            "total_slots":   self.total_slots(),
            "active_count":  self.active_count(),
            "stale_count":   self.stale_count(),
            "decode_count":  self._decode_count,
            "sparse_triggered": self.total_slots() > self.SPARSE_THRESHOLD,
        }

    def _oldest_slots(self, n: int) -> List[tuple]:
        """返回最早的 n 个 slot"""
        all_slots = []
        for key, slots in self._slots.items():
            for s in slots:
                all_slots.append((key, s.last_used_decode))
        all_slots.sort(key=lambda x: x[1])
        return [k for k, _ in all_slots[:n]]

    def should_sparse(self) -> bool:
        """KV 长度是否已触发稀疏注意力"""
        return self.total_slots() > self.SPARSE_THRESHOLD

    def active_mask(self) -> List[bool]:
        """
        返回当前各 slot 是否在稀疏窗口内。
        用于 flash_attn_ext 内核决定使用全量还是稀疏计算。
        """
        total = self.total_slots()
        if total <= self.SPARSE_THRESHOLD:
            return [True] * total

        # 稀疏：只保留最近 MAX_ACTIVE_TOKENS 个
        mask = [False] * total
        mask[-self.MAX_ACTIVE_TOKENS:] = [True] * self.MAX_ACTIVE_TOKENS
        return mask
