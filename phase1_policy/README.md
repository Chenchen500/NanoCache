# Phase 1 Policy — Python KV Cache Strategy

Python 策略层：不依赖 CUDA，纯 CPU 计算。

## 文件说明

| 文件 | 说明 |
|------|------|
| `quantizer.py` | INT8 per-head 量化器（encode/decode）|
| `importance_scorer.py` | 重要性评分器（指数衰减 / 注意力权重）|
| `sparse_kv_manager.py` | KV slot 元数据管理 + LRU 决策 |

## 快速开始

```python
from quantizer import Int8Quantizer
from importance_scorer import ImportanceScorer
from sparse_kv_manager import SparseKVManager

# 量化
q = Int8Quantizer(n_kv_heads=8, head_dim=128, n_tokens=512)
kv_int8, scales = q.quantize(kv_fp16)
kv_restore = q.dequantize(kv_int8, scales)

# 评分
scorer = ImportanceScorer(alpha=0.05, max_keep=128)
mask = scorer.decide_retention(n_tokens=4096, threshold=2048)

# Slot 管理
mgr = SparseKVManager()
mgr.register(layer=0, is_k=True, slot_id=0, n_tokens=128)
mgr.on_seen(layer=0, is_k=True)
to_offload = mgr.on_decode_done()
```
