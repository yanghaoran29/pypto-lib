# 分页KV缓存与block_table间接寻址
本文解释`block_table`在分页KV缓存核函数中的使用方式，以及它为何会引入**编译期无法解析**的运行时动态依赖。
涉及文件：
- `models/qwen3/14b/qwen3_14b_decode.py` — 单层解码
- `models/qwen3/14b/qwen3_14b_decode_full.py` — 多层全量解码

## 1. 什么是block_table
在分页KV缓存中，每个请求的KV项存储在**固定大小的物理块**中，这些物理块在内存中可以是不连续的。
`block_table`是一个扁平化的INT32类型张量，用于将**逻辑块索引**映射到**物理块ID**：

```
逻辑视角：block_table[b, sb]  →  物理块ID (pbid)
核函数布局：block_table[b * max_blocks_per_seq + sb] （扁平化一维张量）
```

**核函数签名**（`qwen3_14b_decode.py`，第130行）：
```python
block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32]
```

`BLOCK_TABLE_FLAT_DYN`是`pl.dynamic()`动态变量，因此一份编译后的程序可处理任意批次大小。主机端会按照`[batch * max_blocks_per_seq]`的形状分配该张量。

## 2. KV缓存布局
```
k_cache / v_cache: [num_total_blocks * num_kv_heads * BLOCK_SIZE, head_dim]
```
给定物理块ID `pbid` 和KV头索引`ki`，缓存中对应块的起始行计算方式为：
```python
cache_row = (pbid * num_kv_heads + ki) * BLOCK_SIZE
```

## 3. 通过block_table实际读取的数据

`block_table`本身只存储物理块ID（INT32整数），但通过它间接寻址后，真正被加载的数据是：

| 数据 | 张量 | 形状（每次读取） | 用途 |
|------|------|----------------|------|
| K 历史分块 | `k_cache` | `[BLOCK_SIZE, head_dim]` | QK 矩阵乘法的 K 操作数 |
| V 历史分块 | `v_cache` | `[BLOCK_SIZE, head_dim]` | SV 矩阵乘法的 V 操作数 |

即：`block_table` → `pbid` → `cache_row` → 从 `k_cache` / `v_cache` 加载历史 KV tile。
`block_table` 本身在计算中不直接参与矩阵运算，仅作为**地址查找表**使用。

## 4. 核函数中的访问模式
### 4.1 缓存写入（slot_mapping 路径）
将当前token的K/V写入缓存时使用`slot_mapping`（第307-309行），该变量直接提供物理行地址——**无需查询block_table**：
```python
slot      = pl.tensor.read(slot_mapping, [b])
slot_block  = slot // BLOCK_SIZE
slot_offset = slot - slot_block * BLOCK_SIZE
cache_row   = (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
```

### 4.2 缓存读取（block_table 路径）
注意力机制读取历史K/V数据时，需要通过`block_table`遍历所有上下文块：
```python
block_table_base = b * max_blocks_per_seq          # 单请求行基址
for sb in pl.range(ctx_blocks):
    block_table_idx = block_table_base + sb
    pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
    cache_row = (pbid * num_kv_heads + ki) * BLOCK_SIZE
    k_tile = k_cache[cache_row : cache_row + BLOCK_SIZE, :]
```
每个`sb`迭代中，`block_table`会被读取两次：一次在`qk_matmul`阶段加载 K tile，一次在`sv_matmul`阶段加载 V tile。

## 5. 为何会产生运行时动态依赖
pypto会将核函数编译为静态任务图，其中DMA调度、分块地址、操作执行顺序均在**编译期**确定。
而`block_table`通过**间接寻址**打破了这一特性：

```
block_table_idx （静态：循环归纳变量）
       ↓
tensor.read(block_table, [block_table_idx])   ← 仅运行时可知值
       ↓
pbid                                          ← 运行时值
       ↓
cache_row = f(pbid)                           ← 运行时地址
       ↓
k_cache[cache_row : cache_row + BLOCK_SIZE]   ← DMA源地址编译期未知
       ↓
matmul(q_tile, k_tile)                        ← 依赖上述所有结果
```

`tensor.read`之后的每一步都依赖前一步的结果。编译器无法跨迭代预取或流水线化K/V分块，因为**下一个块的物理地址**必须等到运行时解析出`pbid`后才能确定。

### 依赖关系表
| 数值 | 编译期可知？ |
|-------|----------------------|
| `block_table_idx` (`b * max_blocks_per_seq + sb`) | 是 — 循环归纳变量 |
| `pbid` (从`block_table`读取) | **否 — 运行时** |
| `cache_row` (`f(pbid)`) | **否 — 运行时** |
| K/V分块地址 | **否 — 运行时** |
| QK / SV 矩阵乘法输入 | **否 — 运行时** |

## 6. 结果：使用pl.range而非pl.pipeline
遍历上下文块的`sb`循环使用`pl.range`（串行执行），而非`pl.pipeline`（软件流水线）：
```python
for sb in pl.range(ctx_blocks):   # 串行 — 无法流水线化
    pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
    ...
```
`pl.pipeline`要求编译器提前知晓所有分块地址，从而实现跨阶段的DMA与计算重叠执行。由于每次迭代的K/V地址都依赖运行时的`pbid`，编译器无法进行静态地址分析，因此无法应用流水线优化。

### 6.1 注意力计算的实际依赖图（非简单顺序链）

在每个请求的注意力计算中（`qwen3_14b_decode.py` Scope 2），四个 `pl.at` kernel 的数据依赖关系**不是简单的一一对应顺序链**，而是包含两处跳级依赖的 DAG：

```
rope_kv_cache (写入 all_q_padded + v_cache[slot])
    │ (all_q_padded)                    ╲ (v_cache，block_table 间接读取)
    ▼                                     ▼
  qk_matmul ─────────▶ softmax ─────────▶ sv_matmul ─────▶ online_softmax
  (all_raw_scores)     │ (all_exp_padded)  │ (all_oi_tmp)
                       │                    │
                       ╰─── all_cur_mi/li ──╯ (跨一级依赖)
```

**依赖表：**

| pl.at | 依赖前驱 | 数据来源 | 是否仅依赖前一个？ |
|-------|---------|---------|-----------------|
| `qk_matmul` | `rope_kv_cache` | `all_q_padded` + `k_cache[pbid]` | ✅ 是（单一前驱） |
| `softmax` | `qk_matmul` | `all_raw_scores0/1` | ✅ 是（单一前驱） |
| `sv_matmul` | `softmax` + **`rope_kv_cache`** | `all_exp_padded0/1` + `v_cache[pbid]` | ❌ **两个前驱**（跨级） |
| `online_softmax` | `sv_matmul` + **`softmax`** | `all_oi_tmp0/1` + `all_cur_mi/li` | ❌ **两个前驱**（跨级） |

**两处跳级依赖的原因：**

1. **`sv_matmul → rope_kv_cache`（经由 v_cache）**  
   `rope_kv_cache` 将当前 token 的 V 写入 `v_cache[slot]`（通过 `slot_mapping` 得到物理地址）。`sv_matmul` 读取历史 V tile 时，最后一个 `sb` 对应的物理块 `pbid` 正是当前 token 所在块。但编译器无法静态确认这个依赖：写入地址来自 `slot_mapping`（静态），读取地址来自 `block_table[sb]`（运行时），二者的物理对应关系只有运行时才能确定。

2. **`online_softmax → softmax`（经由 all_cur_mi/li）**  
   `online_softmax` 需要两类数据：
   - `all_oi_tmp`（来自 `sv_matmul`，直接前驱）
   - `all_cur_mi/li`（来自 `softmax`，跨一级）

   `sv_matmul` 只负责 `exp_tile × V` 矩阵乘，不产生 `mi/li`，因此 `online_softmax` 必须直接依赖 `softmax`。

**block_table 的影响：**

若使用连续 KV cache，编译器**可能**通过静态地址区间分析发现 `v_cache` 的 RAW hazard，在任务图中显式建立 `sv_matmul → rope_kv_cache` 依赖边。引入 `block_table` 后，`sv_matmul` 读取的 `v_cache` 地址是 `f(pbid)` 的函数，编译器无法静态判断是否与 `rope_kv_cache` 的写入重叠，该依赖**只能靠程序顺序（program order）隐式保证**，对编译器的依赖分析不透明。

## 7. 对比：连续KV缓存
在连续KV缓存（无分页）中，块`sb`和头`ki`对应的缓存行计算方式为：
```python
cache_row = (b * max_blocks_per_seq * num_kv_heads + ki * max_blocks_per_seq + sb) * BLOCK_SIZE
```
这是静态循环变量的**纯仿射函数**，因此：
- 编译器可在编译期计算所有地址
- 执行块`sb`的计算时，可同时发起块`sb+1`的DMA预取
- `sb`循环可使用`pl.pipeline`实现

## 8. decode_full.py 中的block_table使用差异

`qwen3_14b_decode_full.py` 是多层全量解码版本，`block_table` 的**逻辑含义**与单层版本完全相同，但在以下几个维度有所不同。

### 8.1 KV cache 布局增加层维度

单层版本的缓存行计算：
```python
cache_row = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
```

多层版本新增 `layer_cache_base` 偏移（第438、485行）：
```python
layer_cache_base = layer_idx * layer_cache_rows
cache_row = layer_cache_base + (pbid * num_kv_heads + kvh) * BLOCK_SIZE
```

其中 `layer_cache_rows = batch * max_blocks_per_seq * num_kv_heads * BLOCK_SIZE`。
即：所有层的 K/V cache 合并为一个大张量 `[num_layers * layer_cache_rows, head_dim]`，通过 `layer_cache_base` 分隔各层。**`block_table` 本身仍是一份，所有层共享同一套物理块映射。**

### 8.2 注意力阶段结构重组（stage-major 融合）

单层版本：对每个 `(b, gi)`，在 `for sb in pl.range(ctx_blocks)` 内串行完成 QK + softmax + SV，再换下一个 `gi`：
```
对每个 b:
  对每个 gi（两两配对，pl.parallel）:
    for sb in pl.range(ctx_blocks):   ← block_table 读取在此
      QK matmul
    for sb in pl.range(ctx_blocks):
      softmax
    for sb in pl.range(ctx_blocks):   ← block_table 读取在此
      SV matmul
```

多层版本：将四个阶段**提到 gi 循环外**，变为 stage-major 结构（第427-542行）：
```
对每个 b:
  Stage 2.2（QK）:   for gi: for sb in pl.parallel(ctx_blocks): block_table 读取 → k_cache
  Stage 2.3（softmax）: for gi: for sb in pl.parallel(ctx_blocks): 纯计算，不访问 block_table
  Stage 2.4（SV）:   for gi: for sb in pl.parallel(ctx_blocks): block_table 读取 → v_cache
  Stage 2.5（online softmax 归约）: for gi: for sb in pl.range(...): 纯计算
```

这带来两个变化：
1. **`sb` 循环从 `pl.range` 改为 `pl.parallel`**（配合 `chunked_loop_optimizer`）。在单层版本中 `sb` 循环是串行的（`pl.range`），多层版本将其改为并行（`pl.parallel(ctx_blocks, chunk=SB_BATCH)`），由多核分担不同 `sb` 的计算。但间接寻址的本质未变：每个 `sb` 任务仍需在运行时读 `block_table` 才能确定 K/V 物理地址。
2. **`block_table` 读取次数不变**：每个 `(gi, sb)` 对仍然在 QK 阶段和 SV 阶段各读一次，两次共享同一个 `pbid` 的计算结果（通过 `all_raw_scores` / `all_exp_padded` 中间张量串联两阶段）。

### 8.3 对比总结

| 维度 | `decode.py`（单层） | `decode_full.py`（多层） |
|------|--------------------|-----------------------|
| KV cache 形状 | `[num_blocks * num_kv_heads * BLOCK_SIZE, head_dim]` | `[num_layers * ... , head_dim]`，按层偏移 |
| block_table 共享 | 单层独用 | 所有层共享同一份 |
| sb 循环类型 | `pl.range`（串行） | `pl.parallel`（多核并行） |
| 注意力阶段结构 | gi-major：gi 内嵌套 sb | stage-major：先全部 QK，再全部 softmax，再全部 SV |
| block_table 读取次数（每请求每层） | `ctx_blocks × 2`（QK + SV 各一次） | 同上 |
| 间接寻址的本质 | 运行时读 pbid，再算 cache_row | 完全相同，仅增加 `layer_cache_base` 加法 |

## 9. slot_mapping 的构建逻辑

`slot_mapping` 与 `block_table` 形成互补：前者用于**写入**当前 token 的 KV，后者用于**读取**历史 KV。

### 9.1 init_slot_mapping() 实现

`qwen3_14b_decode.py` 第 712-720 行（`decode_full.py` 第 753-761 行相同）：

```python
def init_slot_mapping():
    slots = torch.empty(batch, dtype=torch.int32)
    for b in range(batch):
        pos = int(seq_lens_seed[b].item()) - 1        # 当前 token 的序列位置（0-based）
        logical_block = pos // BLOCK_SIZE              # 逻辑块号
        page_offset = pos % BLOCK_SIZE                 # 块内偏移
        phys_block = b * max_blocks_per_seq + logical_block  # 物理块 ID（测试用恒等映射）
        slots[b] = phys_block * BLOCK_SIZE + page_offset    # 物理行地址
    return slots
```

**核心计算链**：

```
seq_lens[b] → pos = seq_lens[b] - 1 → logical_block = pos // BLOCK_SIZE
                                    ↘ page_offset = pos % BLOCK_SIZE
                                                    ↓
            phys_block = block_table[b, logical_block]   (实际推理中从 block_table 查询)
                            ↓
            slot = phys_block * BLOCK_SIZE + page_offset   (最终物理行地址)
```

**测试简化**：上述代码中 `phys_block = b * max_blocks_per_seq + logical_block` 直接用算术公式替代了查表，因为 `init_block_table()` 返回恒等映射 `torch.arange(num_blocks)`，即 `block_table[i] == i`。

### 9.2 在 kernel 中的使用

`qwen3_14b_decode.py` 第 307-309 行（`rope_kv_cache` 阶段）：

```python
slot = pl.tensor.read(slot_mapping, [b])          # 运行时读取
slot_block = slot // BLOCK_SIZE                    # 反解物理块 ID
slot_offset = slot - slot_block * BLOCK_SIZE       # 反解块内偏移
cache_row = (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
k_cache = pl.assemble(k_cache, ..., [cache_row, 0])
v_cache = pl.assemble(v_cache, ..., [cache_row, 0])
```

**关键点**：
- `slot_mapping` 已经是**扁平化的物理行地址**，kernel 直接用它定位缓存写入位置
- 虽然 `slot` 是运行时读取的动态值，但写入地址只与**单个请求**相关，不影响跨请求的并行性
- 与 `block_table` 的区别：`slot_mapping` 是**标量索引**（一个请求对应一个 slot），`block_table` 是**向量索引**（一个请求对应 `ctx_blocks` 个 pbid）

### 9.3 slot_mapping 与 block_table 的对应关系

在正确的推理流程中，当前 token 写入后，下一步 decode 时：

```
当前 decode 步（token t）:
  slot_mapping[b] = phys_block_t * BLOCK_SIZE + offset_t  ← 写入 KV
      ↓ (KV cache 更新)
下一次 decode 步（token t+1）:
  block_table[b, sb] 包含 phys_block_t               ← 读取历史 KV
```

即：`slot_mapping[b]` 写入的物理地址，会在后续 decode 步骤中通过 `block_table` 被读取。但二者的**构造时机不同**：
- `slot_mapping`：每次 decode 前由调度器根据当前序列长度动态生成
- `block_table`：由 KV cache 管理器维护，记录每个请求已分配的所有物理块

测试代码中二者都是静态预先初始化的，但实际推理中都是运行时动态值。

## 10. 总结
| 维度 | 连续KV缓存 | 分页KV缓存（block_table） |
|--------|--------------------|-----------------------------|
| K/V分块地址 | 静态仿射地址 | 运行时间接地址 |
| 循环方式 | `pl.pipeline`（DMA与计算重叠） | `pl.range` / `pl.parallel`（无法跨块预取） |
| 编译期可调度性 | 完全支持 | 被`tensor.read`阻塞 |
| 内存灵活性 | 要求连续分配 | 支持任意物理布局 |
| 写入路径 | 直接计算物理地址 | 通过 `slot_mapping` 查表（标量） |
| 读取路径 | 直接计算物理地址 | 通过 `block_table` 查表（向量，长度 = `ctx_blocks`） |

通过`block_table`实现的间接寻址，是分页KV缓存的核心代价：**以牺牲编译期调度能力和预取优化机会为代价**，换取灵活的非连续物理内存分配能力。

---
