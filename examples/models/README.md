# 最新大模型 PyPTO 实现

本目录包含使用 PyPTO 实现的最新大模型推理内核，针对 Ascend NPU 优化。

## 支持的模型

### 1. Kimi K2 (月之暗面)
- **位置**: `kimi/kimi_k2_decode.py`
- **架构特点**:
  - MoE (Mixture of Experts): 8 个专家 + 1 个共享专家
  - 混合注意力：滑动窗口 + 全局注意力
  - 支持超长上下文 (128K+)
  - GQA (Grouped Query Attention)
- **配置**:
  - Hidden: 4096
  - Heads: 32 (Q) / 8 (KV)
  - Experts: 8 + 1 shared
  - Context: 4096 (sliding window)

### 2. Xiaomi MiLM (小米大模型)
- **位置**: `milm/milm_decode.py`
- **架构特点**:
  - Llama-style Transformer
  - SwiGLU 激活函数
  - RoPE 位置编码
  - GQA 分组查询注意力
  - 针对移动/边缘部署优化
- **配置**:
  - Hidden: 4096
  - Heads: 32 (Q) / 8 (KV)
  - Intermediate: 11008 (SwiGLU)
  - Context: 4096

### 3. Qwen3 (阿里巴巴)
- **位置**: `qwen3/` (已有实现)
- **参考**: Qwen3-32B 架构

### 4. DeepSeek V3.2
- **位置**: `deepseek_v3_2/` (已有实现)
- **参考**: DeepSeek V3 MoE 架构

## 使用示例

### Kimi K2 Decode
```python
from models.kimi.kimi_k2_decode import build_kimi_k2_decode_program

# 构建程序
program = build_kimi_k2_decode_program(
    batch=16,
    max_seq_len=4096,
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    num_experts=8,
    num_active_experts=4,
)

# 获取解码函数
decode_fn = program.kimi_k2_decode_layer
```

### Xiaomi MiLM Decode
```python
from models.milm.milm_decode import build_milm_decode_program

# 构建程序
program = build_milm_decode_program(
    batch=16,
    max_seq_len=4096,
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    intermediate_size=11008,
)

# 获取解码函数
decode_fn = program.milm_decode_layer
```

## 架构对比

| 模型 | Hidden | Heads(Q/KV) | Experts | Context | 特点 |
|------|--------|-------------|---------|---------|------|
| Kimi K2 | 4096 | 32/8 | 8+1 | 128K+ | MoE, 混合注意力 |
| MiLM | 4096 | 32/8 | - | 4K | SwiGLU, 边缘优化 |
| Qwen3 | 5120 | 64/8 | - | 4K | 高带宽注意力 |
| DeepSeek V3 | - | - | MoE | - | 稀疏 MoE |

## PyPTO 关键操作

这些实现使用了以下 PyPTO 核心操作：

- `pl.matmul`: 矩阵乘法
- `pl.slice`: 张量切片
- `pl.assemble`: 张量组装
- `pl.row_sum`, `pl.row_max`: 行归约
- `pl.row_expand_mul`, `pl.col_expand_mul`: 广播乘法
- `pl.exp`, `pl.rsqrt`, `pl.sigmoid`: 数学函数
- `pl.parallel`: 并行循环
- `pl.auto_incore()`: InCore 内存管理
- `pl.fillpad`: 填充操作

## 性能优化技巧

1. **分块计算**: 使用 `K_CHUNK`, `Q_OUT_CHUNK` 等分块减少 InCore 压力
2. **并行循环**: 使用 `pl.parallel` 进行循环级并行
3. **批处理**: 使用 `BATCH_TILE` 进行批处理优化
4. **在线 Softmax**: Flash Decoding 使用数值稳定的在线 softmax
5. **融合操作**: 将多个操作融合减少内存访问

## 参考资源

- PyPTO 文档：`/data/z00885570/pypto3.0/pypto-lib/docs/`
- PTO IR 手册：`/data/z00885570/pypto3.0/PTOAS/docs/PTO_IR_manual.md`
- Qwen3 实现：`/data/z00885570/pypto3.0/pypto-lib/examples/models/qwen3/`
- DeepSeek 实现：`/data/z00885570/pypto3.0/pypto-lib/examples/models/deepseek_v3_2/`

## 下一步

- [ ] 添加 prefill 实现
- [ ] 添加多卡并行支持
- [ ] 性能基准测试
- [ ] 添加更多模型（GLM-4, Yi 等）
