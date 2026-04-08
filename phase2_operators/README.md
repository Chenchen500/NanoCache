# Phase 2 Operators — Fused Sparse Attention

Python + CUDA 实现。

## 文件说明

| 文件 | 说明 |
|------|------|
| `fused_sparse_attention.py` | 稀疏注意力 mask 生成 + NumPy 模拟实现 |

## 稀疏策略（L1 层）

当 `n_keys > 2048` 时，K/V 只取最近 128 个 token 参与点积计算：

```
offset = n_keys - 128
n_active = 128
```

其余 token 的贡献在数学上等效于零（注意力权重置零），但不读取对应 K/V 数据，节省显存带宽。

## NumPy 模拟（无 CUDA 环境验证逻辑）

```python
from fused_sparse_attention import FusedSparseAttention, compute_sparse_mask

attn = FusedSparseAttention(n_heads=128, head_dim=128)
O = attn.forward(Q, K, V)

# 查看是否触发稀疏
offset, n_active = compute_sparse_mask(4096)
# offset=3968, n_active=128
```

## CUDA Kernel

实际 kernel 在 `phase3_engine/nanocache_cuda.cu`，通过 `pybind11` 或 `ctypes` 调用。
