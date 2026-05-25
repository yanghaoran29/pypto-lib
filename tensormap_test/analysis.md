# TensorMap 必要性分析与实测

本目录两个样例 (`paged_consumer_block_table.py` / `dyn_sequence_reduce.py`) 对应"编译期无法确定依赖关系、必须依赖运行期 TensorMap 兜底"的两类典型场景。本文档结合 simpler 源码 (`pypto/runtime/src/a5/runtime/.../`) 与 swimlane profiling 结果, 分析 TensorMap 在这两类场景下的实际行为。

---

## 1. TensorMap 的必要性分析

PyPTO 的依赖关系跟踪有两层:

1. **编译期 (compile-time)**: 当 `pl.slice` / `pl.assemble` 的偏移、shape 全部是静态常量、循环界静态可知时, codegen 可以把任务间的 RAW/WAR/WAW 边写死到 IR / 派发顺序里, 运行时不再需要查表。
2. **运行期 TensorMap (runtime fallback)**: 当编译期看到的偏移 / shape / 循环界含有符号变量 (来自 `pl.tensor.read`、`pl.tensor.dim`、`pl.dynamic` 等), 编译器没法静态枚举所有 producer-consumer 对, 就只能把依赖关系延迟到运行期, 由 AICPU 上的 [`PTO2TensorMap`](../../pypto/runtime/src/a5/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h) 在每个任务 submit 时用 buffer 指针做 hash 查表 + byte-range overlap 判断。

需要 TensorMap 兜底的两个典型场景:

### 场景一: `block_table` 在多个 `pl.at` 中访问

paged-KV-cache 模式下, Stage 1 写 `paged_y` 的位置或 Stage 2 读 `paged_y` 的位置由 `block_table[ob]` 动态查表决定。编译期看到的 slice 偏移是 ``src_row = page_id * PAGE_M`` 这样的符号表达式 (其中 `page_id` 来自 `pl.tensor.read(block_table, [ob])`), 编译器没法证明 "Stage 2 task `ob` 只依赖 Stage 1 task `block_table[ob]`", 只能把依赖交给运行期 TensorMap。

样例: [`paged_consumer_block_table.py`](paged_consumer_block_table.py) —— Stage 1 顺序写 `paged_y`, Stage 2 用 `pl.tensor.read(block_table, [ob])` 查表得到 `page_id`, 再从 `paged_y[page_id*PAGE_M : ...]` gather 出来做 RMSNorm。

### 场景二: 变长 `seq_len` (循环计数是运行期值)

prefill / 多 batch 推理中, 每个 batch 的真实序列长度从 `seq_lens[b]` 运行期读出, 用作 `pl.parallel` / `pl.range` 的循环界。编译期看到的是 ``seq_blocks_b = pl.tensor.read(seq_lens, [b]) // SEQ_TILE`` 这样的符号 scalar, 没法枚举具体迭代数, 编译器也就没法把 producer-consumer 对写死。

样例: [`dyn_sequence_reduce.py`](dyn_sequence_reduce.py) —— 外层 `for b in pl.parallel(BATCH)`, 每个 batch 用 `pl.tensor.read(seq_lens, [b])` 拿到本 batch 的真实长度后, 用 `for sb in pl.parallel(seq_blocks_b)` 跑 Stage 1, 再用 `for sb in pl.range(seq_blocks_b)` 在 Stage 2 内部累加。

---

## 2. 测试样例实测结果

实测办法: 每个样例都加 `--enable-l2-swimlane` 跑一次, 拿到合并 swimlane JSON, 统计 `name == "dependency"` 的 flow event 在生产者 tid → 消费者 tid 上的连接关系 (`ph='s'` 端是 producer, `ph='f'` 端是 consumer)。

### 2.1 `block_table` 场景: 依赖被识别为**全连接**

| 指标 | 数值 |
| --- | --- |
| dep flow 总数 | 512 |
| 唯一 producer tid | 16 (= NUM_PAGES) |
| 唯一 consumer tid | 16 (= NUM_PAGES) |
| 每个 producer 的 fan-out | **全部 16** (min=max=median=16) |
| 每个 consumer 的 fan-in | **全部 16** (min=max=median=16) |

即 Stage 1 的 16 个 pl.at 实例 × Stage 2 的 16 个 pl.at 实例形成 **16×16 全连接的二部图**, 每个 Stage 2 task 误依赖了所有 16 个 Stage 1 task。

**在当前simpler实现中，不能确定该方案一定需要TensorMap。**

#### 从 simpler 中找证据: 为什么会这么识别

**a) TensorMap 的查表是 buffer-粒度 hash + byte-range overlap** (参考 [`pto_tensormap.h:471-524`](../../pypto/runtime/src/a5/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h)):

```cpp
template <typename Fn>
void lookup(const Tensor &tensor, Fn &&on_match) {
    uint32_t bucket_index = hash(tensor.buffer.addr);   // ← 只按 base_ptr hash
    PTO2TensorMapEntry *cur_entry = buckets[bucket_index];
    ...
    while (cur_entry != nullptr) {
        ...
        if (tensor.buffer.addr == cur_entry->buffer_addr) {
            auto overlap_status = cur_entry->check_overlap(tensor);
            if (overlap_status != OverlapStatus::NO_OVERLAP) {
                ... on_match(*cur_entry, overlap_status) ...
            }
        }
        cur_entry = next_entry;
    }
}
```

所有以 `paged_y` 为 base_ptr 的 entry 都落到同一个 bucket; 之后由 `check_overlap` 来 refine。

**b) `check_overlap` 三级 cascade** (参考 [`pto_tensormap.h:203-321`](../../pypto/runtime/src/a5/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h)):

- **L1**: O(1) byte-range 求交。若 `[in_begin, in_end) ∩ [ent_begin, ent_end) = ∅` 返回 `NO_OVERLAP`。
- **L2**: O(ndims) 超矩形精确判断, 仅在 dtype/ndims/strides 完全一致时启用。
- **L3**: 其它情况一律保守返回 `OTHER` (按"可能 overlap"对待)。

**c) producer entry 在 `register_task_outputs` 时插入** (参考 [`pto_dep_compute.h:140-154`](../../pypto/runtime/src/a5/runtime/tensormap_and_ringbuffer/runtime/pto_dep_compute.h)):

```cpp
inline void register_task_outputs(
        const DepInputs &inputs, PTO2TaskId task_id,
        PTO2TensorMap &tensor_map, bool in_manual_scope) {
    ...
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            const Tensor *tensor = inputs.tensors[i].ptr;
            if (!tensor->manual_dep) {
                tensor_map.insert(*tensor, task_id);
            }
        }
    }
}
```

注意这里 `tensor_map.insert(*tensor, ...)` 插入的是 **task 的 INOUT/OUTPUT_EXISTING tensor**, 而不是 `pl.assemble` 内部那块小 sub-slice。也就是说每个 Stage 1 task 把 `paged_y` 这个**整个 buffer** 作为它的产物注册进 TensorMap, 16 个 Stage 1 task 各自插入一条覆盖整个 `paged_y` 的 entry, 16 条 entry 全在同一个 bucket 里。

**d) Stage 2 lookup 时, L1 byte-range 全部命中**:

Stage 2 task 用 `pl.slice(paged_y, [PAGE_M, N1], [src_row, 0])` 读 paged_y 的一小块。但 lookup 拿到的 `tensor` 是这个 task 的 input tensor, 它的 buffer 仍是整块 `paged_y` (`tensor.buffer.addr == paged_y`), `start_offset` / `extent_elem` 描述整个 buffer 的 byte 范围。L1 byte-range intersection 看到 reader 的范围与 16 条 producer entry 的范围**完全重合** → 全部返回 `OTHER` (非 NO_OVERLAP) → `lookup()` 把全部 16 条 entry 当作 producer 回调出去 → 触发 16×16 全连接。

#### 直观结论

block_table 的动态指针并不是 "因为 `page_id` 是符号值, L1 byte-range 算不出来" —— 而是 **L1 byte-range 拿到的 byte 范围本身就是整块 buffer**, 16 个 producer 在 byte 维度上完全重合。L2 hyper-rectangle 也救不了, 因为 `pl.assemble` 注册到 TensorMap 的是 producer-side 的"整张 paged_y", strides 一致, byte-range 完全 cover, 永远走 `OTHER` 这一路。

要让 simpler 能 narrow 这种依赖, 需要在 `register_task_outputs` 阶段就把 `pl.assemble(paged_y, ..., [m0, 0])` 翻成一个**只覆盖 sub-slice 的 producer entry** (start_offset/shape 写入实际写入的子片), 然后 reader 那侧的 slice 也带准确 start_offset。这部分是 simpler 当前实现的一个保守点。

### 2.2 变长 `seq_len` 场景: 依赖被**正确按 batch 隔离**

| 指标 | 数值 |
| --- | --- |
| dep flow 总数 | 152 |
| 唯一 producer tid | 76 (= sum of seq_blocks_b over BATCH=16) |
| 唯一 consumer tid | 16 (= BATCH, 每个 batch 一个 Stage 2 task) |
| producer 的 fan-out 分布 | min=1, max=3, median=1 (histogram: 1→48, 2→18, 3→10) |
| consumer 的 fan-in 分布 | min=2, max=12, median=8 (跟当 batch 的 seq_blocks_b 同阶) |

**每个 Stage 2 task 看到的前序 Stage 1 task 数 = 当 batch 的 `seq_blocks_b`** (2、4、5、6、8、9、10、11、12 等), 不同 batch 互不串扰, 跨 batch 的误依赖完全消失。fan-out histogram 也显示绝大多数 Stage 1 task 只挂 1 个下游 consumer (= 它所在 batch 的那个 Stage 2)。

#### 为什么能正确识别

[`dyn_sequence_reduce.py`](dyn_sequence_reduce.py) 把中间缓冲 `stage12_y_b = pl.create_tensor([HEAD_DIM, MAX_SEQ], dtype=pl.FP32)` **挪到 `for b in pl.parallel(BATCH)` 内部**:

```python
for b in pl.parallel(BATCH):
    seq_len_b    = pl.tensor.read(seq_lens, [b])
    seq_blocks_b = seq_len_b // SEQ_TILE

    # 每次 pl.parallel(BATCH) 迭代各自分配一份 stage12_y_b,
    # 不同 batch 拿到不同的 GM 指针。
    stage12_y_b = pl.create_tensor([HEAD_DIM, MAX_SEQ], dtype=pl.FP32)

    for sb in pl.parallel(seq_blocks_b):
        with pl.at(name_hint="stage1_silu_rmsnorm"):
            ... stage12_y_b = pl.assemble(stage12_y_b, normed, [0, s0])

    with pl.at(name_hint="stage2_reduce"):
        acc = pl.full([HEAD_DIM, SEQ_TILE], dtype=pl.FP32, value=0.0)
        for sb in pl.range(seq_blocks_b):
            blk = pl.slice(stage12_y_b, [HEAD_DIM, SEQ_TILE], [0, s0])
            acc = pl.add(acc, blk)
        ...
```

由于每个 batch 各自 `pl.create_tensor` 一份, 不同 batch 的 `stage12_y_b` 拿到的 **`buffer.addr` 不同**, TensorMap 按 base_ptr hash (`pto_tensormap.h:473`) 天然把不同 batch 的 producer/consumer 分到不同的 bucket, 跨 batch 的误重叠根本进不到 `check_overlap` 这一步。

同一 batch 内部, Stage 1 的 `seq_blocks_b` 个 task 仍然全部以 stage12_y_b 为 output 注册进 TensorMap (同一个 base_ptr), Stage 2 task lookup 时 L1 byte-range 又会把这些 entry 全部带回。因此 consumer 的 fan-in 大约等于 `seq_blocks_b`, 这正是观测到的结果 —— Stage 2 task 看到的 fan-in 就是 **当前 batch 的 Stage 1 task 数量**, 跨 batch 互不影响。

> 注: producer fan-out 偶有 2 或 3, 是因为同一 Stage 1 task 同时被本 batch 的 Stage 2 task 当作 fanin (overlap 命中) + 被 [`pto_dep_compute.h:98-104`](../../pypto/runtime/src/a5/runtime/tensormap_and_ringbuffer/runtime/pto_dep_compute.h) Step A "creator retention" 链上 (`owner_task_id`), 同一对在 swimlane 里会画出多条 dep flow。属于多渠道叠加, 不是新的逻辑误依赖。

---

## 3. 小结

| 场景 | 编译期能否定 dep | 运行期 TensorMap 行为 | 实测结果 |
| --- | --- | --- | --- |
| `block_table` 动态查表 (同一 buffer 被多个 pl.at scatter 写, 再按动态 index gather 读) | 否 | producer entry 覆盖整张 buffer, L1 byte-range 全部命中 | 16×16 **全连接**, 每个 consumer fan-in = NUM_PAGES |
| 变长 `seq_len` (每个 batch 独立的 sub-DAG, 用 per-batch `pl.create_tensor`) | 否 (循环界是动态) | 不同 batch 拿到不同 `buffer.addr`, TensorMap 按 hash 天然分桶 | **per-batch 正确隔离**, 每个 consumer fan-in = 本 batch 的 `seq_blocks_b` |

两个场景共同点都是 **编译期无法静态确定 producer-consumer 对**, 必须靠 TensorMap 在运行期 submit 阶段补全依赖。差别在于 simpler 的当前 byte-range overlap 实现:

- 用**整张 buffer** 作为 producer/consumer 输出/输入注册时, byte-range 必然完全重合 → 全连接 (block_table 这一类);
- 改用 **per-iteration 局部 buffer** (per-batch `pl.create_tensor`) 让不同分组的依赖落到不同 base_ptr, simpler 就能按 buffer 维度天然隔离 (变长 seq_len 这一类)。
