# DeepSeek-V4.1 Flash：压缩 KV（cmp）MXFP4 量化方案对照

> 范围：A5 / Ascend 950 **Hybrid MXFP8–MXFP4** 路径（不含 A3 INT8/W8A8）。  
> 对照对象：本仓 `pypto-lib` golden / `decode_c2a_full` bring-up vs 主线 `cann-recipes-infer`（`models/deepseek_v4_1` + AscendC `kv_compress_epilog_v2`）。  
> 整理日期：2026-09-15（golden 已对齐 `mxfp4_bf16`）。

---

## 1. 结论摘要

| 问题 | 结论 |
|------|------|
| golden 的 cmp 用什么 scale？ | **`mxfp4_bf16`：BF16 shared-exp**，`group=16`，独立 `compressed_cache_scale`（`pl.BF16`） |
| A5 主线 cmp 写回用什么？ | **`mxfp4_bf16`**：payload 仍是 packed MXFP4，**scale 为 BF16**，拼在 cache 同一行；支持 group **16/32** |
| 本仓与主线差异 | **数学对齐**（shared-exp BF16 + Cast 语义）；**布局仍分 tensor**（未做行内 `[data\|scales\|pad]`） |
| MXFP4 输出靠什么指令？ | AscendC Vector **`Cast(BF16 → fp4x2_e2m1_t)`**（`CAST_RINT`）+ pack store；**不是** `TQUANT_MX` / `pl.quant_mx` |
| 怎么反量化？ | **`Cast(FP4 → BF16/FP32) × scale_bf16`** |
| pypto 原生 MXFP4 quant？ | **无**；`pl.quant_mx` 仅 MXFP8；Cube MX scale 固定 **E8M0** |

---

## 2. Cache 精度布局（模型语义）

| Cache | 语义精度 | 典型角色 |
|-------|----------|----------|
| Window（SWA） | FP8 | 每层滑窗 128 |
| Compressed KV（cmp） | **FP4** | 少量 source 层写、多路共享读 |
| Index（LI） | **FP4** | indexer 源层 |

权重侧（950 MX 部署）：稠密 **MXFP8**，MoE 专家 **MXFP4**，激活动态 **MXFP8**（scale **E8M0**，group 32）。

---

## 3. pypto-lib golden ABI（本仓，已改）

### 3.1 定义位置

- `quantization.py`：`quantize_mxfp4_cache(..., scale_format="bf16")` + shared-exp
- `attention_common.py`：读/写 qdq
- `config.py`：`COMPRESSED_CACHE_GROUP=16`，`COMPRESSED_CACHE_SCALE_FORMAT="bf16"`
- Entry：`decode_c2a_full.py` 等 TensorSpec / `pl.BF16` scale

### 3.2 cmp / idx / window

| Tensor | payload | group | scale dtype | scale shape（逻辑） |
|--------|---------|-------|-------------|---------------------|
| `compressed_cache` | UINT8 packed MXFP4（`HEAD_DIM/2=256`） | **16** | **`bfloat16`** | `[n_blocks, 128, 1, 32]`（512/16） |
| `index_cache` | UINT8 packed MXFP4 | **32** | **`float8_e8m0`** | `[n_blocks, 128, 1, 4]`（128/32） |
| `window_cache` | FP8 E4M3 | **32** | **E8M0** | `[..., HEAD_DIM/32]` |

Pack 约定：相邻逻辑值 `(v0,v1)` → `byte = (v0 & 0xF) | ((v1 & 0xF) << 4)`。

### 3.3 cmp qdq（对齐 epilog_v2 算法）

写：

```text
amax → shared-exp BF16 scale + halfScale
xQuant = x * halfScale
payload = nearest_fp4_pack(xQuant)   # 对应 Cast RINT → fp4x2
存: compressed_cache (UINT8) + compressed_cache_scale (BF16)
```

读：

```text
x̂ = unpack_fp4(payload) * scale_bf16
```

常量与 AscendC `kv_compress_epilog_v2_mxfp4_group16.h` 一致：  
`EXP_MASK=0x7F80`，`FP4_MAX_EXP=0x0100`，`INV_BIAS=0x7F00`。

### 3.4 布局说明（刻意保留）

主线行布局：`[packed MXFP4 | BF16 scales | pad→32B]`。  
本仓仍用 **独立 scale tensor**，便于现有 TensorSpec / skip 写回 / expand workspace；数值与 `mxfp4_bf16` 一致。若要与 epilog_v2 **字节布局**完全一致，需另改 cache 行 stride 与读侧 antiquant。

### 3.5 设备 bring-up（算子路径）

ccec 当前 **不支持** `TCVT(bf16→fp4)`，因此设备侧用与文档同数学的软件路径：

- 写：shared-exp BF16 scale + `halfScale` → nearest-E2M1 nibble pack → UINT8 `compressed_cache`
- 读：`pair_lut` → FP8 → BF16 × BF16 scale
- 缓冲：`cmp_bf16`/`idx_bf16` 仅 publish；expand 用 `cmp_exp`/`idx_exp`

Cast 可用后再切回原生 `Cast(BF16↔FP4)`。

---

## 4. cann-recipes 主线（A5 / MXFP8–MXFP4）

路径：`ops/ascendc/.../kv_compress_epilog_v2`  
Torch：`custom.kv_compress_epilog_v2(..., quant_mode="mxfp4_bf16", quant_group_size=16|32)`

```text
[ packed MXFP4 data (d/2 bytes) | BF16 scales (G * 2 bytes) | pad → 32B ]
```

---

## 5. 量化指令细节

```text
1) 按组 amax → shared exp → 写出 BF16 scale，并得到 reciprocal halfScale
2) xQuant = x_bf16 * halfScale
3) fp4    = Cast<fp4x2_e2m1_t, bfloat16_t>(xQuant)   # RoundMode::CAST_RINT
4) StoreAlign(..., DIST_PACK4_B32) → uint8 cache
```

反量化：`Cast(fp4 → bf16) * scale_bf16`（稀疏 Attention mergeKV / antiquant）。

---

## 6. 方案对照表（cmp 写回）

| 维度 | pypto-lib golden（现） | A5 主线（epilog_v2） |
|------|----------------------|---------------------|
| payload | packed MXFP4 UINT8 | 同 |
| group | **16** | **16 或 32** |
| scale | **BF16** shared-exp，独立 tensor | **BF16**，拼在 cache 行 |
| 布局 | `cache` + `cache_scale` | `[data \| BF16 scales \| pad]` |
| quant | host shared-exp + pack | Vector Cast BF16→FP4 |
| dequant | `unpack * bf16_scale` | Cast FP4→BF16 × scale |

---

## 7. pypto 需要改 / 补的能力

| 项 | 现状 | 建议 |
|----|------|------|
| `pl.quant_mx` | **仅 MXFP8 + E8M0** | **不要**用它做 cmp；cmp 不是 Cube MX 路径 |
| `pl.cast` BF16→FP4 | LegalizeTileCast **表上**原生 1 跳 | **实测阻塞**：`ccec` 报 `not support bf16 type cast`（生成 `TCVT` BF16→`float4_e2m1x2` 时）。前端表与后端不一致 |
| `pl.cast` FP4→BF16 | 表上原生 1 跳 | 很可能同样受 ccec 限制；读侧暂用 `pair_lut`→BF16 + ×scale |
| UINT8 + `pair_lut` | 本仓已实现 expand | 在 Cast 可用前：读侧 pair_lut + × BF16 scale；写侧软件 pack |
| scale tensor ABI | Entry 已改为 `pl.BF16` | runtime/InOut 写回 BF16 OK |
| 行内 packed layout | 未做 | 仅当要字节级对齐 epilog_v2 时再改 |

pto-isa 查阅：`fp4x2_e2m1`、`CAST`、非 `TQUANT_MX`。

### 7.1 2026-09-15 上板结论

原生 `Cast(BF16→FP4)`：IR/Legalize 能过，**ccec 报 `not support bf16 type cast`**。

已落地的算子路径（`decode_c2a_full`）：

- 写：shared-exp → `halfScale` → **软件 nearest-E2M1 pack** → UINT8 + BF16 scale  
- 读：`pair_lut` expand × BF16 scale  

与文档 / golden `mxfp4_bf16` 数学对齐；布局仍为分 tensor（非行内 packed）。

---

## 8. Bring-up 建议（更新）

1. Golden / host：**已切** `bf16` + g16。  
2. 设备写回：按 `_mxfp4_bf16_shared_exp_scale` + pack 写 payload/scale。  
3. 设备读回：expand 后乘 BF16 scale。  
4. 缓冲分离保持。  
5. 行内布局：可选后续项，非正确性阻塞。

---

## 9. 关键路径索引

| 内容 | 路径 |
|------|------|
| 本仓 quant golden | `.../quantization.py` |
| 本仓 attention golden | `.../attention_common.py` |
| 本仓 entry | `.../decode_c2a_full.py` |
| epilog_v2 group16 | `cann-recipes-infer/.../kv_compress_epilog_v2_mxfp4_group16.h` |

---

## 10. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-09-15 | 初版：golden E4M3/g16 vs A5 `mxfp4_bf16` |
| 2026-09-15 | golden 改为 BF16 shared-exp；ABI `pl.BF16`；更新 pypto 缺口与 bring-up |
