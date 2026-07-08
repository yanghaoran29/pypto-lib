# MXFP8-MXFP4 单算子 A5 运行结果汇总

## 算子测试结果整合表（按文件名字母排序）
| 序号 | 文件名 | 编译通过 | 运行通过 | 精度通过 | 错误原因 |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `decode_attention_csa.py` | √ | √ | × | Golden 不匹配 (NaN: kv_cache=17, x_out=65536) |
| 2 | `decode_attention_hca.py` | √ | √ | × | Golden 不匹配 (NaN: kv_cache=32, x_out=65536) |
| 3 | `decode_attention_swa.py` | √ | √ | × | Golden 不匹配 (NaN: kv_cache=133, x_out=131072) |
| 4 | `decode_compressor_ratio128.py` | √ | √ | × | Golden 不匹配 (kv 3.1%, cmp_kv_cache 0.003%) |
| 5 | `decode_compressor_ratio4.py` | √ | √ | × | Golden 不匹配 (kv 3.9%, cmp_kv_cache 0.004%) |
| 6 | `decode_indexer.py` | √ | √ | × | Golden 不匹配 (idx_kv_cache 0.009%) |
| 7 | `decode_indexer_compressor.py` | √ | √ | × | Golden 不匹配 (kv 62%, idx_kv_cache 0.008%) |
| 8 | `decode_sparse_attn.py` | √ | √ | × | Golden 不匹配 (attn_out 77.2%) |
| 9 | `decode_sparse_attn_hca.py` | √ | √ | × | Golden 不匹配 (attn_out 92.7%) |
| 10 | `decode_sparse_attn_swa.py` | √ | √ | × | Golden 不匹配 (88.9% 元素超出容差) |
| 11 | `expert_routed.py` | √ | √ | √ | 无 |
| 12 | `expert_shared.py` | √ | √ | √ | 无 |
| 13 | `gate.py` | √ | √ | √ | 无 |
| 14 | `hc_head.py` | √ | √ | × | Golden 不匹配 (y 31.3%) |
| 15 | `hc_post.py` | √ | √ | √ | 无 |
| 16 | `hc_pre.py` | √ | √ | × | Golden 不匹配 (comb NaN×64, 100%) |
| 17 | `mtp_projection.py` | √ | √ | √ | 无 |
| 18 | `prefill_attention_csa.py` | √ | × | × | 超时 (30s) |
| 19 | `prefill_attention_hca.py` | √ | √ | × | Golden 不匹配 (NaN: kv_cache=7, x_out=2M, Inf: x_out=16K) |
| 20 | `prefill_attention_swa.py` | √ | √ | × | Golden 不匹配 (NaN: kv_cache=8 Inf=2, x_out=1.9M Inf=41K) |
| 21 | `prefill_compressor_ratio128.py` | √ | √ | × | Golden 不匹配 (cmp_kv 0.0008%) |
| 22 | `prefill_compressor_ratio4.py` | √ | √ | × | Golden 不匹配 (cmp_kv 0.048%) |
| 23 | `prefill_indexer.py` | √ | × | × | 超时 (30s) |
| 24 | `prefill_indexer_compressor.py` | √ | √ | × | Golden 不匹配 (kv 99.5%, idx_kv_cache 0.049%) |
| 25 | `prefill_sparse_attn.py` | √ | √ | × | Golden 不匹配 (attn_out 32.2%) |
| 26 | `qkv_proj_rope.py` | √ | √ | × | Golden 不匹配 (q 45.3%, kv 10.1%, qr 41.6%, qr_scale 50%) |
| 27 | `rmsnorm.py` | √ | √ | √ | 无 |

## 算子修改记录

> 对比基准：`main` 分支 `models/deepseek/v4/<op>.py` ↔ A5 分支 `models/deepseek/v4/MXFP8-MXFP4/<op>.py`（同名算子）。

### 一、总体修改（跨算子的共性改动）

本目录是将 `main` 分支的 DeepSeek V4 单算子移植到 **A5（950）平台 + MXFP8/MXFP4 量化** 的版本。以下改动在多个算子中反复出现：

1. **目录结构**：`main` 分支算子直接位于 `models/deepseek/v4/`；A5 版本统一放入 `models/deepseek/v4/MXFP8-MXFP4/` 子目录。

2. **RoPE 去掉 gather（交错式 interleaved/GPT-J → 半分式 half-split/NeoX）**：核心目的是消除核内的 gather。交错式把旋转对存成相邻的 `(2i, 2i+1)`，施加旋转需要两次 gather —— dup-gather（半宽 cos/sin 复制到相邻 lane，`j>>1`）与 swap-gather（取配对伙伴 `j^1`），外加 `pl.arange`/`col_expand`/sign 下标构造。改为半分式后旋转对是 `(i, i+HALF)`，lo=`[:HALF]`、hi=`[HALF:]` 为连续切片，cos/sin 直接按半宽读取相乘再 `pl.concat`，gather 与下标构造全部消失。因单算子测试 kernel 与 golden 自洽即可，故 golden 同步改（`unflatten(-1,(-1,2))`+`stack.flatten` → 半宽切分+`torch.cat`），`rope_tables.py` 角度表不变。影响：`qkv_proj_rope`、`decode_compressor_ratio4/128`、`decode_indexer`、`decode_indexer_compressor`、`prefill_compressor_ratio4/128`、`prefill_indexer_compressor`、`decode_sparse_attn(_hca/_swa)`、`prefill_sparse_attn`。

3. **INT32→FP16 直接 cast 规避**：A5 不支持 `INT32→FP16` 直接转换，量化路径统一改为经 FP32 中转：`pl.cast(i32, FP16)` → `pl.cast(pl.cast(i32, FP32), FP16)`。影响所有含量化的算子（`expert_routed`/`expert_shared`、`qkv_proj_rope`、`decode_indexer`、`prefill_indexer`、`decode_sparse_attn`、`prefill_sparse_attn` 等）。

4. **`pl.spmd` → `pl.parallel` + 显式 `pl.at(level=pl.Level.CORE_GROUP)`**：部分算子的 SPMD 编程模型改写为 `pl.parallel(0, N, 1)` 循环 + 显式 CORE_GROUP scope；同时移除显式 `deps=[...]` 依赖串联和 `pl.array.create(..., pl.TASK_ID)` 的 TaskId 手工管理（改由框架自动依赖解析）。影响：`expert_routed`、`expert_shared`、`gate`、`hc_post`、`mtp_projection`、`prefill_indexer`。

5. **移除 `allow_early_resolve=True`**：A5 版本不再使用早解析标志。影响：`hc_head`、`decode_attention_csa`、`decode_sparse_attn(_hca/_swa)`、`qkv_proj_rope`。

6. **单算子测试自包含化**：`build_tensor_specs` 不再从 `decode_metadata` import 辅助函数（`block_table` / `ori_slot_mapping` / `position_ids_from_starts` / `resolve_start_positions` / `*_decode_start_set` / `kv_seq_lens_from_starts`），改为在 fixture 内用 torch 直接内联实现；`--start-pos` 语义简化为「传值则全 batch 用该单值，否则用默认覆盖模式」，去掉 8k 长上下文默认采样点。影响 decode 系 attention / compressor / indexer 各算子。

7. **`config.py` 缩减测试规模**：`max_position_embeddings` 16384→8192；删除 `DECODE_START_POS=8192`；`KV_CMP_MAX_BLOCKS` 32→8（缩小位置范围与压缩 KV 池以适配 A5）。

### 二、各算子修改（按测试表顺序）

| 算子 | 主要改动 |
| :--- | :--- |
| `config.py` | `max_position_embeddings` 16384→8192；删 `DECODE_START_POS`；`KV_CMP_MAX_BLOCKS` 32→8 |
| `decode_attention_csa.py` | 移除 `allow_early_resolve`；测试 fixture 自包含化（内联 `decode_metadata` 辅助函数、简化 `--start-pos`） |
| `decode_attention_hca.py` | 测试 fixture 自包含化（内联 `decode_metadata`、简化 `--start-pos`） |
| `decode_attention_swa.py` | 测试 fixture 自包含化（内联 `block_table`/`ori_slot_mapping`/`position_ids` 等）；INDEX-cast 写回守卫简化（先 cast 再比较 `>=0`） |
| `decode_compressor_ratio128.py` | RoPE 去掉 gather（交错→半分，kernel + golden）；测试 fixture 自包含化 |
| `decode_compressor_ratio4.py` | RoPE 去掉 gather（交错→半分，kernel + golden）；测试 fixture 自包含化 |
| `decode_indexer.py` | RoPE 去掉 gather（交错→半分）；INT32→FP16 经 FP32 中转；score 阶段 SPMD 合并（`score_kv_quant`/`score_mat`/`score_reduce` 三段合为单段 `score`）；测试 fixture 自包含化 |
| `decode_indexer_compressor.py` | RoPE 去掉 gather（交错→半分）；INT32→FP16 经 FP32 中转；测试 fixture 自包含化 |
| `decode_sparse_attn.py` | RoPE 逆变换去掉 gather（交错→半分，kernel + golden，去 dup-gather 预计算）；INT32→FP16 经 FP32 中转；移除 `allow_early_resolve` |
| `decode_sparse_attn_hca.py` | 同 `decode_sparse_attn`（RoPE 去掉 gather + FP32 中转 + 移除 `allow_early_resolve`） |
| `decode_sparse_attn_swa.py` | 同 `decode_sparse_attn`（RoPE 去掉 gather + FP32 中转） |
| `expert_routed.py` | `pl.spmd`→`pl.parallel`+`pl.at(CORE_GROUP)`；移除显式 `deps=`/`TASK_ID` 依赖管理；INT32→FP16 经 FP32 中转 |
| `expert_shared.py` | `pl.spmd`→`pl.parallel`+`pl.at(CORE_GROUP)`；INT32→FP16 经 FP32 中转 |
| `gate.py` | `pl.spmd`→`pl.parallel`+`pl.at(CORE_GROUP)`（gate 主循环 + route_hash 阶段） |
| `hc_head.py` | 移除 6 处 `allow_early_resolve`（仅此改动） |
| `hc_post.py` | `pl.spmd`→`pl.parallel`+`pl.at(CORE_GROUP)` |
| `hc_pre.py` | 结构改写：`main` 为单个融合大核 `pl.spmd(NUM_CORES=24, sync_start=True)`（`@pl.jit.inline`），A5 拆解为 9 个独立分阶段 `pl.spmd` 循环 |
| `mtp_projection.py` | 结构重写：动态形状 `T_DYN` + `@pl.jit.inline` + HC lanes `[T, HC_MULT, D]` → 静态 `[B, S, D]` + `@pl.jit`，去掉 HC 维度；`pl.spmd`→`pl.parallel` |
| `prefill_attention_csa.py` | 与 main 完全一致（无改动） |
| `prefill_attention_hca.py` | 与 main 完全一致（无改动） |
| `prefill_attention_swa.py` | 与 main 完全一致（无改动） |
| `prefill_compressor_ratio128.py` | RoPE 去掉 gather（交错→半分，kernel + golden） |
| `prefill_compressor_ratio4.py` | RoPE 去掉 gather（交错→半分，kernel + golden） |
| `prefill_indexer.py` | `pl.spmd`→`pl.parallel`（多处：qr_proj/qr_rope/hadamard_quant/weights_proj/topk 等）；INT32→FP16 经 FP32 中转 |
| `prefill_indexer_compressor.py` | RoPE 去掉 gather（交错→半分，kernel + golden） |
| `prefill_sparse_attn.py` | RoPE 逆变换去掉 gather（交错→半分，kernel + golden，去 dup-gather 预计算）；INT32→FP16 经 FP32 中转 |
| `qkv_proj_rope.py` | RoPE 去掉 gather（q 与 kv 两处 + golden，交错→半分，去核内 index/sign 构造）；INT32→FP16 经 FP32 中转；移除 `allow_early_resolve` |
| `rmsnorm.py` | 与 main 完全一致（无改动） |
