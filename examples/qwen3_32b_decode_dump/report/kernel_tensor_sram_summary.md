# Qwen3 32B Decode — Kernel Local Tensor SRAM Summary

## 1) Overview

Single-layer decode forward for Qwen3 32B with **variable-length context** per
session.  Each batch item can have a different context length (up to 4096),
passed via the `seq_lens` input tensor (`[BATCH], INT32`).

## 2) Model Config

- `BATCH=16`, `MAX_SEQ=4096`, `HIDDEN=5120`
- `NUM_HEADS=64`, `NUM_KV_HEADS=8`, `HEAD_DIM=128`
- `INTERMEDIATE=25600`

## 3) Tuning Knobs

- `K_CHUNK=256`
- `Q_OUT_CHUNK=64`
- `KV_OUT_CHUNK=32`
- `SEQ_TILE=120`
- `MLP_OUT_CHUNK=32`
- `BATCH_TILE=4`

## 4) Variable Sequence Length Support

- `seq_lens: Tensor[[BATCH], INT32]` — per-session context length.
- Decode position: `pos = seq_lens[b] - 1`.
- Context length for attention: `ctx_len = seq_lens[b]`.
- All `pl.view` of GM tensors use fixed, 512-B-aligned storage shapes.

### `valid_shape` Integration (per `tensor_valid_shape.md` design)

KV-cache views carry explicit `valid_shape` annotations:

```python
k_tile = pl.view(k_cache, [SEQ_TILE, HEAD_DIM], [row0, 0],
                 valid_shape=[valid_len, HEAD_DIM])
v_tile = pl.view(v_cache, [SEQ_TILE, HEAD_DIM], [row0, 0],
                 valid_shape=[valid_len, HEAD_DIM])
```

- **Storage shape** `[SEQ_TILE, HEAD_DIM]` satisfies 512-B alignment.
- **valid_shape** `[valid_len, HEAD_DIM]` tracks the actual number of valid
  cache rows in the tail tile.

**Current workaround** (until the compiler propagates `valid_shape`):
`scores_valid = pl.view(scores, [1, valid_len], ...)` + zero-padded `exp_pad`
are used to mask garbage scores from padding rows. Once the compiler's
`ConvertTensorToBlockOps` pass forwards tensor-level `valid_shape` to
`block.load valid_shapes`, `row_max` / `row_sum` will automatically operate
on valid columns only, and the workaround can be removed.

## 5) Function-Level Statistics

| InCore Function | Local Tensor Size (B) | Buffers |
|---|---:|---:|
| `qwen3_decode_layer_incore_2_aic` | 248,256 | 17 |
| `qwen3_decode_layer_incore_0_aiv` | 195,584 | 13 |
| `qwen3_decode_layer_incore_5` | 167,424 | 5 |
| `qwen3_decode_layer_incore_4_aiv` | 167,168 | 6 |
| `qwen3_decode_layer_incore_0_aic` | 140,288 | 13 |
| `qwen3_decode_layer_incore_1_aic` | 140,288 | 21 |
| `qwen3_decode_layer_incore_3_aic` | 140,288 | 13 |
| `qwen3_decode_layer_incore_3_aiv` | 104,960 | 11 |
| `qwen3_decode_layer_incore_1_aiv` | 97,280 | 20 |
| `qwen3_decode_layer_incore_2_aiv` | 55,392 | 26 |
| `qwen3_decode_layer_incore_4_aic` | 17,408 | 8 |
| **Total** | **1,474,336** | - |

## 6) Group-Level Statistics (AIC / AIV split)

| Logical Kernel | AIC (B) | AIV (B) | Solo (B) |
|---|---:|---:|---:|
| `qwen3_decode_layer_incore_2` | 248,256 | 55,392 | 0 |
| `qwen3_decode_layer_incore_0` | 140,288 | 195,584 | 0 |
| `qwen3_decode_layer_incore_5` | 0 | 0 | 167,424 |
| `qwen3_decode_layer_incore_4` | 17,408 | 167,168 | 0 |
| `qwen3_decode_layer_incore_1` | 140,288 | 97,280 | 0 |
| `qwen3_decode_layer_incore_3` | 140,288 | 104,960 | 0 |

## 7) Constraint Check

- **AIC 256KB limit**: PASS (max AIC = `248,256 B` < 262,144)
- **AIV 192KB limit**: PASS (max AIV = `195,584 B` < 196,608)
