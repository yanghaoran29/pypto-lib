# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode C2A full attention (single-card bring-up).

Single-file ``@pl.jit.inline`` orchestration aligned with
``golden_compressed_attention(mode=FULL, ratio=2)``. This path drops
``DistributedTensor`` / TP all-reduce and writes local FP32 o_proj partials
straight to BF16 ``output`` (requires ``--tp 1``). Stages follow the Torch
reference in ``attention_common.py`` / ``golden.py`` and reuse pypto.language
patterns from V4 Pro ``qkv_proj_rope`` (RoPE scatter / RMS), ``expert_shared``
(MXFP8 ``quant_mx`` / ``matmul_mx``), and ``decode_attention_csa`` (cache
writeback / task deps). Compressed KV uses A5 ``mxfp4_bf16`` math: UINT8
packed MXFP4 + BF16 shared-exp scales (group 16). Payload conversion uses
native A5 ``Cast(BF16↔FP4)`` and a storage-only ``FP4↔UINT8`` reinterpret;
the index cache keeps its E8M0 byte scale ABI. Readers stage and expand only
needed blocks rather than materializing the full cache.
"""

import sys

import pypto.language as pl
import torch

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.attention_common import AttentionGoldenResult, golden_compressed_attention
from models.deepseek_v4_1_flash.config import AttentionMode

# ---------------------------------------------------------------------------
# Compile-time tiling (MX cube + decode capacity)
# ---------------------------------------------------------------------------
# Module-level mirrors of config — kernel bodies cannot do ATTR access.
B_DYN = C.B_DYN
CMP_BLOCKS_DYN = C.CMP_BLOCKS_DYN
COMPRESSED_CACHE_GROUP = C.COMPRESSED_CACHE_GROUP
D = C.D
DECODE_MAX_TOKENS = C.DECODE_MAX_TOKENS
HEAD_DIM = C.HEAD_DIM
INDEX_BLOCKS_DYN = C.INDEX_BLOCKS_DYN
INDEX_CACHE_GROUP = C.INDEX_CACHE_GROUP
INDEX_DIM = C.INDEX_DIM
INDEX_H = C.INDEX_H
INDEX_TOPK = C.INDEX_TOPK
LOCAL_H = C.LOCAL_H
LOCAL_O_GROUPS = C.LOCAL_O_GROUPS
LOCAL_O_WIDTH = C.LOCAL_O_WIDTH
MAX_BATCH_PER_DP = C.MAX_BATCH_PER_DP
MX_GROUP = C.MX_GROUP
ORI_BLOCKS_DYN = C.ORI_BLOCKS_DYN
O_GROUP_IN = C.O_GROUP_IN
O_LORA = C.O_LORA
Q_LORA = C.Q_LORA
ROPE_DIM = C.ROPE_DIM
STATE_HEADS = C.STATE_HEADS
TABLE_DYN = C.TABLE_DYN
TP_SIZE = C.TP_SIZE
T_DYN = C.T_DYN
WINDOW_CACHE_GROUP = C.WINDOW_CACHE_GROUP

T_MAX = DECODE_MAX_TOKENS
EPS = C.FLASH.rms_norm_eps
Q_LORA_INV = 1.0 / float(Q_LORA)
HEAD_DIM_INV = 1.0 / float(HEAD_DIM)
INDEX_DIM_INV = 1.0 / float(INDEX_DIM)
HEAD_SCALE = HEAD_DIM**-0.5
INDEX_SCALE = (INDEX_DIM**-0.5) * (INDEX_H**-0.5)
NEG_INF = -1.0e30

M_TILE = 16
K_TILE = 256
K_SCALE_GROUPS = K_TILE // MX_GROUP
N_TILE = 32
# Match the A5-proven MXFP8 cube shape used by V4 Pro shared/routed experts.
# Besides using the supported 16x256x256 instruction shape, this avoids
# splitting the 32K-wide q projection into 512 tiny orchestration tasks.
N_TILE_WIDE = 256
QUANT_K = 256
TOKEN_TILE = 8  # FP32 row needs 32B align ⇒ ≥8 cols; never materialize [TOKEN_TILE, ACT_MAX] FP32
ROPE_T_TILE = 8
HEAD_TILE = 64
RMS_D_TILE = 128  # must fit INDEX_DIM=128 static slice shape
ACT_COL_TILE = 256  # column tile for wide ACT_MAX casts/pads on Vec
ATTN_K_TILE = 16
ATTN_HEAD_CHUNK = 16  # cube M minimum is 16 on A5
ATTN_O_TILE = 128  # chunk HEAD_DIM on value matmul to shrink c2v slot_size
ATTN_STAT_PAD = 8  # pad [H,1] FP32 stats to 32B row alignment
SCORE_LEN = 512  # bring-up: keep Top-K=512 feasible while cutting sort workspace
SCORE_LEN_PAD = ((SCORE_LEN + 31) // 32) * 32
# Per-query BF16 staging (ABI caches stay dynamically sized). Each query only
# materializes its visible SCORE_LEN prefix, independently of physical block ids.
WS_ORI_BLOCKS = T_MAX
WS_CMP_BLOCKS = T_MAX * (SCORE_LEN // 128)
WS_IDX_BLOCKS = T_MAX * (SCORE_LEN // 128)
WS_ORI_ROWS = WS_ORI_BLOCKS * 128
WS_CMP_ROWS = WS_CMP_BLOCKS * 128
WS_IDX_ROWS = WS_IDX_BLOCKS * 128
# Widest activation/result dim used by MX helpers (covers TP=1 LOCAL_H * HEAD_DIM).
ACT_MAX = max(D, Q_LORA, LOCAL_H * HEAD_DIM, LOCAL_O_WIDTH, INDEX_H * INDEX_DIM)
assert T_MAX % M_TILE == 0
assert D % K_TILE == 0
assert Q_LORA % K_TILE == 0
assert HEAD_DIM % HEAD_TILE == 0
assert INDEX_TOPK <= SCORE_LEN_PAD
assert SCORE_LEN_PAD % 32 == 0
assert ACT_MAX % K_TILE == 0
assert ACT_MAX % N_TILE_WIDE == 0
assert ACT_MAX % ACT_COL_TILE == 0
assert LOCAL_H % ATTN_HEAD_CHUNK == 0
assert HEAD_DIM % ATTN_O_TILE == 0
assert 128 % ATTN_K_TILE == 0
assert INDEX_TOPK % ATTN_K_TILE == 0
assert HEAD_DIM % 2 == 0
assert INDEX_DIM % 2 == 0
assert HEAD_DIM % COMPRESSED_CACHE_GROUP == 0
CMP_PACKED = HEAD_DIM // 2  # UINT8 packed MXFP4 bytes (PR #1210 / golden ABI)
IDX_PACKED = INDEX_DIM // 2
MXFP4_PACK_CHUNK = 32
assert CMP_PACKED % MXFP4_PACK_CHUNK == 0
assert IDX_PACKED % MXFP4_PACK_CHUNK == 0
# AscendC kv_compress_epilog_v2 ``mxfp4_bf16`` BF16 exponent constants.
_BF16_EXP_MASK = 0x7F80
_BF16_INV_BIAS = 0x7F00
_FP4_E2M1_MAX_EXP = 0x0100
_FP4_SPECIAL_INV = 0x0040
_BF16_NAN = 0x7FC0
FP4_QUANT_ROWS = 16
INDEX_QUANT_ROWS = 32
_LN2 = 0.6931471805599453
_INV_LN2 = 1.4426950408889634

RING_HEAP_4G = 4 * 1024 * 1024 * 1024


@pl.jit.inline
def _mxfp4_expand_idx_block(
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    block_packed: pl.Tensor[[128, IDX_PACKED], pl.UINT8],
    block_scale: pl.Tensor[[INDEX_DIM // INDEX_CACHE_GROUP, 128], pl.FP8E8M0],
    out_bf16: pl.Tensor[[WS_IDX_ROWS, INDEX_DIM], pl.BF16],
    block_id: pl.Scalar[pl.INDEX],
    dst_row0: pl.Scalar[pl.INDEX],
):
    """Expand one index block with native FP4 TCVT and E8M0 scale bytes."""
    for r in pl.range(128):
        for c in pl.range(IDX_PACKED):
            pl.write(
                block_packed,
                [r, c],
                pl.read(index_cache, [block_id, r, 0, c]),
            )
        for g in pl.range(INDEX_DIM // INDEX_CACHE_GROUP):
            pl.write(
                block_scale,
                [g, r],
                pl.read(index_cache_scale, [block_id, r, 0, g]),
            )

    for r0 in pl.range(0, 128, INDEX_QUANT_ROWS):
        packed = pl.load(
            block_packed,
            [r0, 0],
            [INDEX_QUANT_ROWS, IDX_PACKED],
            target_memory=pl.Mem.Vec,
        )
        fp4_rows = pl.reinterpret_view(packed, pl.FP4)
        decoded = pl.cast(fp4_rows, pl.BF16, mode="rint")
        for g in pl.range(INDEX_DIM // INDEX_CACHE_GROUP):
            codes = pl.reinterpret_view(
                pl.load(
                    block_scale,
                    [g, r0],
                    [1, INDEX_QUANT_ROWS],
                    target_memory=pl.Mem.Vec,
                ),
                pl.UINT8,
            )
            # A5 has no direct UINT8→FP32 TCVT. UINT8→FP16→FP32 is a
            # supported, exact widening path for all E8M0 byte codes.
            exponent = pl.add(pl.cast(pl.cast(codes, pl.FP16), pl.FP32), -127.0)
            scale = pl.reshape(
                pl.exp(pl.mul(exponent, _LN2)),
                [INDEX_QUANT_ROWS, 1],
            )
            values = pl.slice(
                decoded,
                [INDEX_QUANT_ROWS, INDEX_CACHE_GROUP],
                [0, g * INDEX_CACHE_GROUP],
            )
            dequant = pl.cast(
                pl.row_expand_mul(pl.cast(values, pl.FP32), scale),
                pl.BF16,
                mode="rint",
            )
            out_bf16 = pl.store(
                dequant,
                [dst_row0 + r0, g * INDEX_CACHE_GROUP],
                out_bf16,
            )


@pl.jit.inline
def _mxfp4_dequant_cmp_block(
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    block_packed: pl.Tensor[[128, CMP_PACKED], pl.UINT8],
    block_scale: pl.Tensor[[HEAD_DIM // COMPRESSED_CACHE_GROUP, 128], pl.BF16],
    out_bf16: pl.Tensor[[WS_CMP_ROWS, HEAD_DIM], pl.BF16],
    block_id: pl.Scalar[pl.INDEX],
    dst_row0: pl.Scalar[pl.INDEX],
):
    """Dequant one cmp block with native FP4→BF16 TCVT and BF16 scales."""
    for r in pl.range(128):
        for c in pl.range(CMP_PACKED):
            pl.write(
                block_packed,
                [r, c],
                pl.read(compressed_cache, [block_id, r, 0, c]),
            )
        for g in pl.range(HEAD_DIM // COMPRESSED_CACHE_GROUP):
            pl.write(
                block_scale,
                [g, r],
                pl.read(compressed_cache_scale, [block_id, r, 0, g]),
            )

    for r0 in pl.range(0, 128, FP4_QUANT_ROWS):
        packed = pl.load(
            block_packed,
            [r0, 0],
            [FP4_QUANT_ROWS, CMP_PACKED],
            target_memory=pl.Mem.Vec,
        )
        fp4_rows = pl.reinterpret_view(packed, pl.FP4)
        decoded = pl.cast(fp4_rows, pl.BF16, mode="rint")
        for g in pl.range(HEAD_DIM // COMPRESSED_CACHE_GROUP):
            scale_row = pl.load(
                block_scale,
                [g, r0],
                [1, FP4_QUANT_ROWS],
                target_memory=pl.Mem.Vec,
            )
            scale = pl.reshape(pl.cast(scale_row, pl.FP32), [FP4_QUANT_ROWS, 1])
            values = pl.slice(
                decoded,
                [FP4_QUANT_ROWS, COMPRESSED_CACHE_GROUP],
                [0, g * COMPRESSED_CACHE_GROUP],
            )
            dequant = pl.cast(
                pl.row_expand_mul(pl.cast(values, pl.FP32), scale),
                pl.BF16,
                mode="rint",
            )
            out_bf16 = pl.store(
                dequant,
                [dst_row0 + r0, g * COMPRESSED_CACHE_GROUP],
                out_bf16,
            )


@pl.jit.inline
def _mxfp4_quant_cmp_rows(
    src_bf16: pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16],
    normalized_bf16: pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16],
    packed_u8: pl.Tensor[[T_MAX, CMP_PACKED], pl.UINT8],
    scale_transposed: pl.Tensor[
        [HEAD_DIM // COMPRESSED_CACHE_GROUP, T_MAX], pl.BF16
    ],
    token0: pl.Scalar[pl.INDEX],
):
    """Quantize one aligned 16-row batch with native A5 BF16→FP4 TCVT."""
    for g0 in pl.range(0, HEAD_DIM, COMPRESSED_CACHE_GROUP):
        group = pl.load(
            src_bf16,
            [token0, g0],
            [FP4_QUANT_ROWS, COMPRESSED_CACHE_GROUP],
            target_memory=pl.Mem.Vec,
        )
        group_fp32 = pl.cast(group, pl.FP32)
        reduce_tmp = pl.create_tile(
            [FP4_QUANT_ROWS, COMPRESSED_CACHE_GROUP],
            dtype=pl.FP32,
            target_memory=pl.Mem.Vec,
        )
        amax_fp32 = pl.row_max(pl.abs(group_fp32), reduce_tmp)
        amax_bf16_col = pl.cast(amax_fp32, pl.BF16, mode="rint")
        amax_bf16 = pl.reshape(amax_bf16_col, [1, FP4_QUANT_ROWS])
        bits = pl.reinterpret_view(amax_bf16, pl.INT16)
        raw_exp = pl.ands(bits, _BF16_EXP_MASK)
        tmp = pl.create_tile([1, 32], dtype=pl.UINT8, target_memory=pl.Mem.Vec)

        keep_exp = pl.cmps(raw_exp, _FP4_E2M1_MAX_EXP, cmp_type=4)
        max_exp = pl.sels(keep_exp, raw_exp, tmp, _FP4_E2M1_MAX_EXP)
        shared_exp = pl.add(max_exp, -_FP4_E2M1_MAX_EXP)

        finite = pl.cmps(raw_exp, _BF16_EXP_MASK, cmp_type=1)
        nonzero = pl.cmps(raw_exp, 0, cmp_type=1)
        scale_bits = pl.sels(finite, shared_exp, tmp, _BF16_NAN)
        scale_bits = pl.sels(nonzero, scale_bits, tmp, 0)

        half_bits = pl.add(pl.neg(shared_exp), _BF16_INV_BIAS)
        half_bits = pl.sels(finite, half_bits, tmp, _BF16_NAN)
        half_bits = pl.sels(nonzero, half_bits, tmp, 0)
        not_special = pl.cmps(shared_exp, _BF16_INV_BIAS, cmp_type=1)
        half_bits = pl.sels(not_special, half_bits, tmp, _FP4_SPECIAL_INV)
        scale = pl.reshape(
            pl.reinterpret_view(scale_bits, pl.BF16),
            [FP4_QUANT_ROWS, 1],
        )
        half_scale = pl.reshape(
            pl.reinterpret_view(half_bits, pl.BF16),
            [FP4_QUANT_ROWS, 1],
        )
        normalized = pl.cast(
            pl.row_expand_mul(group_fp32, pl.cast(half_scale, pl.FP32)),
            pl.BF16,
            mode="rint",
        )
        normalized_bf16 = pl.store(normalized, [token0, g0], normalized_bf16)
        scale_transposed = pl.store(
            pl.reshape(scale, [1, FP4_QUANT_ROWS]),
            [g0 // COMPRESSED_CACHE_GROUP, token0],
            scale_transposed,
        )

    normalized_rows = pl.load(
        normalized_bf16,
        [token0, 0],
        [FP4_QUANT_ROWS, HEAD_DIM],
        target_memory=pl.Mem.Vec,
    )
    fp4_rows = pl.cast(normalized_rows, pl.FP4, mode="rint")
    carrier = pl.reinterpret_view(fp4_rows, pl.UINT8)
    packed_u8 = pl.store(carrier, [token0, 0], packed_u8)
    return packed_u8, scale_transposed


@pl.jit.inline
def _mxfp4_quant_idx_rows(
    src_bf16: pl.Tensor[[T_MAX, INDEX_DIM], pl.BF16],
    normalized_bf16: pl.Tensor[[T_MAX, INDEX_DIM], pl.BF16],
    packed_u8: pl.Tensor[[T_MAX, IDX_PACKED], pl.UINT8],
    scale_transposed: pl.Tensor[
        [INDEX_DIM // INDEX_CACHE_GROUP, T_MAX], pl.FP8E8M0
    ],
    token0: pl.Scalar[pl.INDEX],
):
    """Quantize 32 rows to native FP4 with group-32 E8M0 scale codes."""
    for g0 in pl.range(0, INDEX_DIM, INDEX_CACHE_GROUP):
        group = pl.load(
            src_bf16,
            [token0, g0],
            [INDEX_QUANT_ROWS, INDEX_CACHE_GROUP],
            target_memory=pl.Mem.Vec,
        )
        group_fp32 = pl.cast(group, pl.FP32)
        reduce_tmp = pl.create_tile(
            [INDEX_QUANT_ROWS, INDEX_CACHE_GROUP],
            dtype=pl.FP32,
            target_memory=pl.Mem.Vec,
        )
        amax = pl.row_max(pl.abs(group_fp32), reduce_tmp)
        raw_scale = pl.maximum(pl.mul(amax, 1.0 / 6.0), 2.0**-9)
        log2_scale = pl.mul(pl.log(raw_scale, high_precision=True), _INV_LN2)
        exponent_fp32 = pl.maximum(pl.minimum(log2_scale, 128.0), -127.0)
        exponent = pl.cast(
            exponent_fp32,
            pl.INT16,
            mode="ceil",
            saturation_mode="on",
        )
        code = pl.cast(
            pl.add(exponent, 127),
            pl.UINT8,
            mode="trunc",
            saturation_mode="on",
        )
        scale_code = pl.reinterpret_view(
            pl.reshape(code, [1, INDEX_QUANT_ROWS]),
            pl.FP8E8M0,
        )
        scale_transposed = pl.store(
            scale_code,
            [g0 // INDEX_CACHE_GROUP, token0],
            scale_transposed,
        )
        reciprocal = pl.exp(pl.mul(pl.cast(exponent, pl.FP32), -_LN2))
        normalized = pl.cast(
            pl.row_expand_mul(group_fp32, reciprocal),
            pl.BF16,
            mode="rint",
        )
        normalized_bf16 = pl.store(normalized, [token0, g0], normalized_bf16)

    normalized_rows = pl.load(
        normalized_bf16,
        [token0, 0],
        [INDEX_QUANT_ROWS, INDEX_DIM],
        target_memory=pl.Mem.Vec,
    )
    fp4_rows = pl.cast(normalized_rows, pl.FP4, mode="rint")
    carrier = pl.reinterpret_view(fp4_rows, pl.UINT8)
    packed_u8 = pl.store(carrier, [token0, 0], packed_u8)
    return packed_u8, scale_transposed


MX_NT_WQ_A = (Q_LORA + N_TILE_WIDE - 1) // N_TILE_WIDE
MX_NT_WQ_B = (LOCAL_H * HEAD_DIM + N_TILE_WIDE - 1) // N_TILE_WIDE
MX_NT_WKV = (HEAD_DIM + N_TILE_WIDE - 1) // N_TILE_WIDE
MX_NT_IDX = (INDEX_H * INDEX_DIM + N_TILE_WIDE - 1) // N_TILE_WIDE
MX_NT_WO_B = (D + N_TILE_WIDE - 1) // N_TILE_WIDE


def _parse_c2a_stage() -> str:
    for index, argument in enumerate(sys.argv):
        if argument == "--stage" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if argument.startswith("--stage="):
            return argument.split("=", 1)[1]
    return "full"


C2A_STAGE = _parse_c2a_stage()
_STAGE_RANK = {"pad": 0, "qkv": 1, "window": 2, "compress": 3, "indexer": 4, "attn": 5, "full": 6}
if C2A_STAGE not in _STAGE_RANK:
    raise ValueError(f"--stage must be one of {tuple(_STAGE_RANK)}, got {C2A_STAGE!r}")
C2A_STAGE_ID = _STAGE_RANK[C2A_STAGE]


def golden_decode_c2a_full(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    request_ids: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    position_ids: torch.Tensor,
    compressed_rope_cos: torch.Tensor,
    compressed_rope_sin: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_wgate: torch.Tensor,
    compressor_state_rows: torch.Tensor,
    compressor_state: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    compressed_slots: torch.Tensor,
    index_wk: torch.Tensor,
    index_norm_weight: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> AttentionGoldenResult:
    return golden_compressed_attention(
        mode=AttentionMode.FULL,
        ratio=2,
        x=x,
        wq_a=wq_a,
        wq_a_scale=wq_a_scale,
        q_norm_weight=q_norm_weight,
        wq_b=wq_b,
        wq_b_scale=wq_b_scale,
        wkv=wkv,
        wkv_scale=wkv_scale,
        kv_norm_weight=kv_norm_weight,
        attn_sink=attn_sink,
        wo_a=wo_a,
        wo_b=wo_b,
        wo_b_scale=wo_b_scale,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        window_slots=window_slots,
        window_indices=window_indices,
        window_cache=window_cache,
        window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache,
        compressed_cache_scale=compressed_cache_scale,
        compressed_indices=None,
        compressor_wkv=compressor_wkv,
        compressor_wgate=compressor_wgate,
        compressor_norm_weight=compressor_norm_weight,
        compressor_state_rows=compressor_state_rows,
        compressor_state=compressor_state,
        compressed_slots=compressed_slots,
        position_ids=position_ids,
        compressed_lens=compressed_lens,
        compressed_rope_cos=compressed_rope_cos,
        compressed_rope_sin=compressed_rope_sin,
        index_wk=index_wk,
        index_norm_weight=index_norm_weight,
        index_wq_b=index_wq_b,
        index_wq_b_scale=index_wq_b_scale,
        index_weights_proj=index_weights_proj,
        index_cache=index_cache,
        index_cache_scale=index_cache_scale,
        index_block_table=index_block_table,
        request_ids=request_ids,
        candidate_mask=None,
    )


# ---------------------------------------------------------------------------
# Shared leaf helpers
# ---------------------------------------------------------------------------


@pl.jit.inline
def _rms_norm_rows(
    x_fp32: pl.Tensor[[T_MAX, ACT_MAX], pl.FP32],
    weight: pl.Tensor,
    out_bf16: pl.Tensor[[T_MAX, ACT_MAX], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.Scalar[pl.INT32],
):
    """Token-tiled RMSNorm written back as BF16."""
    n_tiles = (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE
    d_tiles = (width + RMS_D_TILE - 1) // RMS_D_TILE
    # Bake 1/width as a Python float — pl.recip rejects Scalar.
    if width == Q_LORA:
        width_inv = Q_LORA_INV
    elif width == INDEX_DIM:
        width_inv = INDEX_DIM_INV
    else:
        width_inv = HEAD_DIM_INV
    for blk in pl.spmd(n_tiles * d_tiles, name_hint="c2a_rms"):
        unit = pl.tile.get_block_idx()
        tg = (unit // d_tiles) * TOKEN_TILE
        d0 = (unit % d_tiles) * RMS_D_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - tg)
        cols = pl.min(RMS_D_TILE, width - d0)
        rows_i = pl.cast(rows, pl.INDEX)
        cols_i = pl.cast(cols, pl.INDEX)
        if tg < num_tokens and d0 < width:
            sq_sum = pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0)
            for rd in pl.pipeline(0, width, RMS_D_TILE, stage=1):
                chunk = pl.slice(
                    x_fp32,
                    [TOKEN_TILE, RMS_D_TILE],
                    [tg, rd],
                    valid_shape=[rows, pl.min(RMS_D_TILE, width - rd)],
                )
                sq = pl.mul(chunk, chunk)
                sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(sq), [1, TOKEN_TILE]))
            mean = pl.mul(sq_sum, width_inv)
            rms_arg = pl.add(mean, EPS)
            inv0 = pl.rsqrt(rms_arg, high_precision=True)
            # A5's current two-input TRSQRT still has roughly 1e-4 relative
            # error.  That is normally harmless, but here the BF16 RMS output
            # is immediately MXFP8-quantized and one ULP can change a group's
            # shared exponent.  One explicit FP32 Newton step makes the RMS
            # boundary stable: y <- y * (1.5 - 0.5*x*y*y).
            inv_sq = pl.mul(inv0, inv0)
            correction = pl.add(pl.mul(pl.mul(rms_arg, inv_sq), -0.5), 1.5)
            inv = pl.mul(inv0, correction)
            inv_t = pl.reshape(inv, [TOKEN_TILE, 1])
            apply = pl.slice(x_fp32, [TOKEN_TILE, RMS_D_TILE], [tg, d0], valid_shape=[rows, cols])
            w_row = pl.cast(pl.reshape(weight[d0 : d0 + RMS_D_TILE], [1, RMS_D_TILE]), pl.FP32)
            scaled = pl.col_expand_mul(pl.row_expand_mul(apply, inv_t), w_row)
            out_bf16[tg : tg + TOKEN_TILE, d0 : d0 + RMS_D_TILE] = pl.set_validshape(
                pl.cast(scaled, target_type=pl.BF16, mode="rint"), rows_i, cols_i
            )


@pl.jit.inline
def _rms_norm_rows_fp32(
    x_fp32: pl.Tensor[[T_MAX, ACT_MAX], pl.FP32],
    weight: pl.Tensor,
    out_fp32: pl.Tensor[[T_MAX, ACT_MAX], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.Scalar[pl.INT32],
):
    """Token-tiled RMSNorm retaining FP32 for a fused downstream quantizer."""
    n_tiles = (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE
    d_tiles = (width + RMS_D_TILE - 1) // RMS_D_TILE
    if width == Q_LORA:
        width_inv = Q_LORA_INV
    elif width == INDEX_DIM:
        width_inv = INDEX_DIM_INV
    else:
        width_inv = HEAD_DIM_INV
    for blk in pl.spmd(n_tiles * d_tiles, name_hint="c2a_rms_fp32"):
        unit = pl.tile.get_block_idx()
        tg = (unit // d_tiles) * TOKEN_TILE
        d0 = (unit % d_tiles) * RMS_D_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - tg)
        cols = pl.min(RMS_D_TILE, width - d0)
        rows_i = pl.cast(rows, pl.INDEX)
        cols_i = pl.cast(cols, pl.INDEX)
        if tg < num_tokens and d0 < width:
            sq_sum = pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0)
            for rd in pl.pipeline(0, width, RMS_D_TILE, stage=1):
                chunk = pl.slice(
                    x_fp32,
                    [TOKEN_TILE, RMS_D_TILE],
                    [tg, rd],
                    valid_shape=[rows, pl.min(RMS_D_TILE, width - rd)],
                )
                sq = pl.mul(chunk, chunk)
                sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(sq), [1, TOKEN_TILE]))
            mean = pl.mul(sq_sum, width_inv)
            rms_arg = pl.add(mean, EPS)
            inv0 = pl.rsqrt(rms_arg, high_precision=True)
            inv_sq = pl.mul(inv0, inv0)
            correction = pl.add(pl.mul(pl.mul(rms_arg, inv_sq), -0.5), 1.5)
            inv = pl.mul(inv0, correction)
            inv_t = pl.reshape(inv, [TOKEN_TILE, 1])
            apply = pl.slice(
                x_fp32, [TOKEN_TILE, RMS_D_TILE], [tg, d0], valid_shape=[rows, cols]
            )
            w_row = pl.cast(
                pl.reshape(weight[d0 : d0 + RMS_D_TILE], [1, RMS_D_TILE]), pl.FP32
            )
            scaled = pl.col_expand_mul(pl.row_expand_mul(apply, inv_t), w_row)
            out_fp32[tg : tg + TOKEN_TILE, d0 : d0 + RMS_D_TILE] = pl.set_validshape(
                scaled, rows_i, cols_i
            )


@pl.jit.inline
def _prepare_rope_interleaved(
    rope_cos_half: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin_half: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    cos_il: pl.Tensor[[T_MAX, ROPE_DIM], pl.FP32],
    sin_signed: pl.Tensor[[T_MAX, ROPE_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    inverse: pl.Scalar[pl.INT32],
):
    """Scatter half-dim RoPE tables into adjacent-pair interleaved form (P0101/P1010)."""
    n_tiles = (num_tokens + ROPE_T_TILE - 1) // ROPE_T_TILE
    for blk in pl.spmd(n_tiles, name_hint="c2a_rope_prep"):
        t0 = blk * ROPE_T_TILE
        rows = pl.min(ROPE_T_TILE, num_tokens - t0)
        if t0 < num_tokens:
            cos_h = pl.slice(
                rope_cos_half, [ROPE_T_TILE, ROPE_DIM // 2], [t0, 0], valid_shape=[rows, ROPE_DIM // 2]
            )
            sin_h = pl.slice(
                rope_sin_half, [ROPE_T_TILE, ROPE_DIM // 2], [t0, 0], valid_shape=[rows, ROPE_DIM // 2]
            )
            cos_tile = pl.full([ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            cos_tile = pl.tensor.scatter(cos_h, mask_pattern=pl.tile.MaskPattern.P0101, dst=cos_tile)
            cos_tile = pl.tensor.scatter(cos_h, mask_pattern=pl.tile.MaskPattern.P1010, dst=cos_tile)
            sin_neg = pl.neg(sin_h)
            # inverse RoPE flips the signed-sin construction.
            if inverse == 0:
                sin_even = sin_neg
                sin_odd = sin_h
            else:
                sin_even = sin_h
                sin_odd = sin_neg
            sin_tile = pl.full([ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            sin_tile = pl.tensor.scatter(sin_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=sin_tile)
            sin_tile = pl.tensor.scatter(sin_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=sin_tile)
            cos_il[t0 : t0 + ROPE_T_TILE, :] = pl.set_validshape(cos_tile, rows, ROPE_DIM)
            sin_signed[t0 : t0 + ROPE_T_TILE, :] = pl.set_validshape(sin_tile, rows, ROPE_DIM)


@pl.jit.inline
def _apply_rope_tail_hd(
    values: pl.Tensor[[T_MAX * LOCAL_H, HEAD_DIM], pl.BF16],
    cos_il: pl.Tensor[[T_MAX * LOCAL_H, ROPE_DIM], pl.FP32],
    sin_signed: pl.Tensor[[T_MAX * LOCAL_H, ROPE_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """In-place RoPE on HEAD_DIM rows (covers token and token×LOCAL_H layouts)."""
    rope_off = HEAD_DIM - ROPE_DIM
    n_tiles = (num_tokens + ROPE_T_TILE - 1) // ROPE_T_TILE
    for blk in pl.spmd(n_tiles, name_hint="c2a_rope_apply_hd"):
        t0 = blk * ROPE_T_TILE
        rows = pl.min(ROPE_T_TILE, num_tokens - t0)
        if t0 < num_tokens:
            chunk = pl.cast(
                values[t0 : t0 + ROPE_T_TILE, rope_off : rope_off + ROPE_DIM],
                target_type=pl.FP32,
            )
            even = pl.gather(chunk, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(chunk, mask_pattern=pl.tile.MaskPattern.P1010)
            swapped = pl.full([ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            swapped = pl.tensor.scatter(odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=swapped)
            swapped = pl.tensor.scatter(even, mask_pattern=pl.tile.MaskPattern.P1010, dst=swapped)
            cos_t = cos_il[t0 : t0 + ROPE_T_TILE, :]
            sin_t = sin_signed[t0 : t0 + ROPE_T_TILE, :]
            rot = pl.add(pl.mul(chunk, cos_t), pl.mul(swapped, sin_t))
            values[t0 : t0 + ROPE_T_TILE, rope_off : rope_off + ROPE_DIM] = pl.set_validshape(
                pl.cast(rot, target_type=pl.BF16, mode="rint"), rows, ROPE_DIM
            )


@pl.jit.inline
def _apply_rope_tail_kv(
    values: pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16],
    cos_il: pl.Tensor[[T_MAX, ROPE_DIM], pl.FP32],
    sin_signed: pl.Tensor[[T_MAX, ROPE_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """In-place RoPE on per-token HEAD_DIM rows (window KV / compressor latent)."""
    rope_off = HEAD_DIM - ROPE_DIM
    n_tiles = (num_tokens + ROPE_T_TILE - 1) // ROPE_T_TILE
    for blk in pl.spmd(n_tiles, name_hint="c2a_rope_apply_kv"):
        t0 = blk * ROPE_T_TILE
        rows = pl.min(ROPE_T_TILE, num_tokens - t0)
        if t0 < num_tokens:
            chunk = pl.cast(
                values[t0 : t0 + ROPE_T_TILE, rope_off : rope_off + ROPE_DIM],
                target_type=pl.FP32,
            )
            even = pl.gather(chunk, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(chunk, mask_pattern=pl.tile.MaskPattern.P1010)
            swapped = pl.full([ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            swapped = pl.tensor.scatter(odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=swapped)
            swapped = pl.tensor.scatter(even, mask_pattern=pl.tile.MaskPattern.P1010, dst=swapped)
            cos_t = cos_il[t0 : t0 + ROPE_T_TILE, :]
            sin_t = sin_signed[t0 : t0 + ROPE_T_TILE, :]
            rot = pl.add(pl.mul(chunk, cos_t), pl.mul(swapped, sin_t))
            values[t0 : t0 + ROPE_T_TILE, rope_off : rope_off + ROPE_DIM] = pl.set_validshape(
                pl.cast(rot, target_type=pl.BF16, mode="rint"), rows, ROPE_DIM
            )


@pl.jit.inline
def _apply_rope_tail_idx_heads(
    values: pl.Tensor[[T_MAX * INDEX_H, INDEX_DIM], pl.BF16],
    cos_il: pl.Tensor[[T_MAX * INDEX_H, ROPE_DIM], pl.FP32],
    sin_signed: pl.Tensor[[T_MAX * INDEX_H, ROPE_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """In-place RoPE on INDEX_DIM rows expanded across INDEX_H."""
    rope_off = INDEX_DIM - ROPE_DIM
    n_tiles = (num_tokens + ROPE_T_TILE - 1) // ROPE_T_TILE
    for blk in pl.spmd(n_tiles, name_hint="c2a_rope_apply_idxh"):
        t0 = blk * ROPE_T_TILE
        rows = pl.min(ROPE_T_TILE, num_tokens - t0)
        if t0 < num_tokens:
            chunk = pl.cast(
                values[t0 : t0 + ROPE_T_TILE, rope_off : rope_off + ROPE_DIM],
                target_type=pl.FP32,
            )
            even = pl.gather(chunk, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(chunk, mask_pattern=pl.tile.MaskPattern.P1010)
            swapped = pl.full([ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            swapped = pl.tensor.scatter(odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=swapped)
            swapped = pl.tensor.scatter(even, mask_pattern=pl.tile.MaskPattern.P1010, dst=swapped)
            cos_t = cos_il[t0 : t0 + ROPE_T_TILE, :]
            sin_t = sin_signed[t0 : t0 + ROPE_T_TILE, :]
            rot = pl.add(pl.mul(chunk, cos_t), pl.mul(swapped, sin_t))
            values[t0 : t0 + ROPE_T_TILE, rope_off : rope_off + ROPE_DIM] = pl.set_validshape(
                pl.cast(rot, target_type=pl.BF16, mode="rint"), rows, ROPE_DIM
            )


@pl.jit.inline
def _apply_rope_tail_idx(
    values: pl.Tensor[[T_MAX, INDEX_DIM], pl.BF16],
    cos_il: pl.Tensor[[T_MAX, ROPE_DIM], pl.FP32],
    sin_signed: pl.Tensor[[T_MAX, ROPE_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """In-place RoPE on per-token INDEX_DIM rows (index-key publish)."""
    rope_off = INDEX_DIM - ROPE_DIM
    n_tiles = (num_tokens + ROPE_T_TILE - 1) // ROPE_T_TILE
    for blk in pl.spmd(n_tiles, name_hint="c2a_rope_apply_idx"):
        t0 = blk * ROPE_T_TILE
        rows = pl.min(ROPE_T_TILE, num_tokens - t0)
        if t0 < num_tokens:
            chunk = pl.cast(
                values[t0 : t0 + ROPE_T_TILE, rope_off : rope_off + ROPE_DIM],
                target_type=pl.FP32,
            )
            even = pl.gather(chunk, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(chunk, mask_pattern=pl.tile.MaskPattern.P1010)
            swapped = pl.full([ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            swapped = pl.tensor.scatter(odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=swapped)
            swapped = pl.tensor.scatter(even, mask_pattern=pl.tile.MaskPattern.P1010, dst=swapped)
            cos_t = cos_il[t0 : t0 + ROPE_T_TILE, :]
            sin_t = sin_signed[t0 : t0 + ROPE_T_TILE, :]
            rot = pl.add(pl.mul(chunk, cos_t), pl.mul(swapped, sin_t))
            values[t0 : t0 + ROPE_T_TILE, rope_off : rope_off + ROPE_DIM] = pl.set_validshape(
                pl.cast(rot, target_type=pl.BF16, mode="rint"), rows, ROPE_DIM
            )


def _make_mxfp8_linear(
    k_dim_c: int,
    n_dim_c: int,
    act_width_c: int = ACT_MAX,
    out_width_c: int = ACT_MAX,
    act_dtype_c=pl.BF16,
):
    """Specialize MXFP8 linear; ``k_groups`` must be a Python constant for MX_A_ZZ."""
    assert k_dim_c % K_TILE == 0
    assert n_dim_c % N_TILE_WIDE == 0
    k_groups_c = k_dim_c // MX_GROUP
    act_is_fp32_c = act_dtype_c == pl.FP32
    assert k_groups_c % 2 == 0
    k_tiles_c = k_dim_c // K_TILE
    n_tiles_c = n_dim_c // N_TILE_WIDE

    if act_is_fp32_c:

        @pl.jit.inline
        def _copy_activation(
            source: pl.Tensor[[T_MAX, act_width_c], pl.FP32],
            destination: pl.Tensor[[T_MAX, act_width_c], pl.FP32],
            num_tokens: pl.Scalar[pl.INT32],
        ):
            for blk in pl.spmd(
                (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE,
                name_hint="c2a_mx_copy_fp32",
            ):
                t0 = blk * TOKEN_TILE
                rows = pl.min(TOKEN_TILE, num_tokens - t0)
                if t0 < num_tokens:
                    for c0 in pl.range(0, k_dim_c, ACT_COL_TILE):
                        cols = pl.min(ACT_COL_TILE, k_dim_c - c0)
                        destination[
                            t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE
                        ] = pl.set_validshape(
                            source[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE],
                            rows,
                            cols,
                        )

    else:

        @pl.jit.inline
        def _copy_activation(
            source: pl.Tensor[[T_MAX, act_width_c], pl.BF16],
            destination: pl.Tensor[[T_MAX, act_width_c], pl.FP32],
            num_tokens: pl.Scalar[pl.INT32],
        ):
            for blk in pl.spmd(
                (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE,
                name_hint="c2a_mx_cast",
            ):
                t0 = blk * TOKEN_TILE
                rows = pl.min(TOKEN_TILE, num_tokens - t0)
                if t0 < num_tokens:
                    for c0 in pl.range(0, k_dim_c, ACT_COL_TILE):
                        cols = pl.min(ACT_COL_TILE, k_dim_c - c0)
                        destination[
                            t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE
                        ] = pl.set_validshape(
                            pl.cast(
                                source[
                                    t0 : t0 + TOKEN_TILE,
                                    c0 : c0 + ACT_COL_TILE,
                                ],
                                target_type=pl.FP32,
                            ),
                            rows,
                            cols,
                        )

    @pl.jit.inline
    def _mxfp8_linear(
        act_bf16: pl.Tensor[[T_MAX, act_width_c], act_dtype_c],
        weight: pl.Tensor,
        weight_scale: pl.Tensor,
        out_fp32: pl.Tensor[[T_MAX, out_width_c], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        """quant_mx → [1, M*G] store + MX_A_ZZ view → matmul_mx (expert_routed)."""
        act_fp32 = pl.create_tensor([T_MAX, act_width_c], dtype=pl.FP32)
        act_mx = pl.create_tensor([T_MAX, act_width_c], dtype=pl.FP8E4M3FN)
        scale_back = pl.create_tensor([1, T_MAX * k_groups_c], dtype=pl.FP8E8M0)
        x_scale_mx = pl.tensor.view(
            scale_back,
            [T_MAX, k_groups_c],
            layout=pl.MX_A_ZZ,
        )
        _copy_activation(act_bf16, act_fp32, num_tokens)

        m_tiles = (num_tokens + M_TILE - 1) // M_TILE
        for blk in pl.spmd(m_tiles * k_tiles_c, name_hint="c2a_mx_quant"):
            unit = pl.tile.get_block_idx()
            mt = unit // k_tiles_c
            kt = unit - mt * k_tiles_c
            t0 = mt * M_TILE
            k0 = kt * K_TILE
            if t0 < num_tokens:
                x_q, x_q_scale = pl.quant_mx(pl.load(act_fp32, [t0, k0], [M_TILE, K_TILE]), group_axis=1)
                act_mx = pl.store(x_q, [t0, k0], act_mx)
                scale_back = pl.store(
                    pl.reshape(x_q_scale, [1, M_TILE * K_SCALE_GROUPS]),
                    [0, t0 * k_groups_c + kt * M_TILE * K_SCALE_GROUPS],
                    scale_back,
                )

        # The proven V4 Pro MXFP8 path uses ordinary SPMD here.  A one-slot
        # cross-core window can retain the producer chain while hundreds of
        # N tiles are queued, which made the decode probe hit the AICPU timeout.
        for blk in pl.spmd(n_tiles_c, name_hint="c2a_mx_linear"):
            n0 = blk * N_TILE_WIDE
            for mt in pl.range(m_tiles):
                t0 = mt * M_TILE
                if t0 < num_tokens:
                    xs0 = pl.load(act_mx, [t0, 0], [M_TILE, K_TILE])
                    xs_scale0 = pl.load(x_scale_mx, [t0, 0], [M_TILE, K_SCALE_GROUPS])
                    w0 = pl.load(weight, [0, n0], [K_TILE, N_TILE_WIDE])
                    w0_scale = pl.load(weight_scale, [0, n0], [K_SCALE_GROUPS, N_TILE_WIDE])
                    acc = pl.matmul_mx(xs0, xs_scale0, w0, w0_scale)
                    for kb in pl.pipeline(K_TILE, k_dim_c, K_TILE, stage=2):
                        ks = kb // MX_GROUP
                        xs_k = pl.load(act_mx, [t0, kb], [M_TILE, K_TILE])
                        xs_scale_k = pl.load(x_scale_mx, [t0, ks], [M_TILE, K_SCALE_GROUPS])
                        w_k = pl.load(weight, [kb, n0], [K_TILE, N_TILE_WIDE])
                        w_scale_k = pl.load(weight_scale, [ks, n0], [K_SCALE_GROUPS, N_TILE_WIDE])
                        acc = pl.matmul_mx_acc(acc, xs_k, xs_scale_k, w_k, w_scale_k)
                    out_fp32 = pl.store(acc, [t0, n0], out_fp32)

    return _mxfp8_linear


_mxfp8_wq_a = _make_mxfp8_linear(D, Q_LORA)
_mxfp8_wq_b = _make_mxfp8_linear(Q_LORA, LOCAL_H * HEAD_DIM)
_mxfp8_wq_b_fp32 = _make_mxfp8_linear(
    Q_LORA, LOCAL_H * HEAD_DIM, act_dtype_c=pl.FP32
)
_mxfp8_wkv = _make_mxfp8_linear(D, HEAD_DIM)
_mxfp8_idx_wq_b = _make_mxfp8_linear(Q_LORA, INDEX_H * INDEX_DIM)
_mxfp8_wo_b = _make_mxfp8_linear(LOCAL_O_WIDTH, D)
_mxfp8_wo_b_compact = _make_mxfp8_linear(
    LOCAL_O_WIDTH,
    D,
    act_width_c=LOCAL_O_WIDTH,
    out_width_c=D,
)


@pl.jit.inline
def _bf16_linear(
    act_bf16: pl.Tensor,
    weight: pl.Tensor,
    out_fp32: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    k_dim: pl.Scalar[pl.INT32],
    n_dim: pl.Scalar[pl.INT32],
):
    """BF16×BF16 → FP32 matmul used by index_wk / index_weights_proj / wo_a."""
    m_tiles = (num_tokens + M_TILE - 1) // M_TILE
    n_tiles = (n_dim + N_TILE - 1) // N_TILE
    for blk in pl.spmd(
        m_tiles * n_tiles,
        name_hint="c2a_bf16_linear",
        optimizations=[pl.cross_core_slot(slot_num=1)],
    ):
        unit = pl.tile.get_block_idx()
        mt = unit // n_tiles
        nt = unit - mt * n_tiles
        t0 = mt * M_TILE
        n0 = nt * N_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        cols = pl.min(N_TILE, n_dim - n0)
        if t0 < num_tokens and n0 < n_dim:
            x0 = pl.slice(act_bf16, [M_TILE, K_TILE], [t0, 0], valid_shape=[rows, pl.min(K_TILE, k_dim)])
            w0 = weight[0 : K_TILE, n0 : n0 + N_TILE]
            acc = pl.matmul(x0, w0, out_dtype=pl.FP32)
            for kb in pl.pipeline(K_TILE, k_dim, K_TILE, stage=1):
                xk = pl.slice(act_bf16, [M_TILE, K_TILE], [t0, kb], valid_shape=[rows, K_TILE])
                wk = weight[kb : kb + K_TILE, n0 : n0 + N_TILE]
                acc = pl.matmul_acc(acc, xk, wk)
            out_fp32[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(acc, rows, cols)


@pl.jit.inline
def _fp32_linear(
    act_bf16: pl.Tensor[[T_MAX, D], pl.BF16],
    weight: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    out_fp32: pl.Tensor[[T_MAX, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Compressor FP32 projection ``x @ w`` (weights already FP32)."""
    m_tiles = (num_tokens + M_TILE - 1) // M_TILE
    n_tiles = HEAD_DIM // N_TILE
    for blk in pl.spmd(
        m_tiles * n_tiles,
        name_hint="c2a_fp32_linear",
        optimizations=[pl.cross_core_slot(slot_num=1)],
    ):
        unit = pl.tile.get_block_idx()
        mt = unit // n_tiles
        nt = unit - mt * n_tiles
        t0 = mt * M_TILE
        n0 = nt * N_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        if t0 < num_tokens:
            x0 = pl.cast(pl.slice(act_bf16, [M_TILE, K_TILE], [t0, 0], valid_shape=[rows, K_TILE]), pl.FP32)
            w0 = weight[0:K_TILE, n0 : n0 + N_TILE]
            acc = pl.matmul(x0, w0, out_dtype=pl.FP32)
            for kb in pl.pipeline(K_TILE, D, K_TILE, stage=1):
                xk = pl.cast(pl.slice(act_bf16, [M_TILE, K_TILE], [t0, kb], valid_shape=[rows, K_TILE]), pl.FP32)
                wk = weight[kb : kb + K_TILE, n0 : n0 + N_TILE]
                acc = pl.matmul_acc(acc, xk, wk)
            out_fp32[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(acc, rows, N_TILE)


# ---------------------------------------------------------------------------
# Algorithm stages
# ---------------------------------------------------------------------------


@pl.jit.inline
def _stage_qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    qr: pl.Tensor[[T_MAX, Q_LORA], pl.BF16],
    q: pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16],
    window_kv: pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    act_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.BF16)
    out_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    n_copy = (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE
    for blk in pl.spmd(n_copy, name_hint="c2a_x_pad"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            act_pad[t0 : t0 + TOKEN_TILE, 0 : D] = pl.set_validshape(
                x[t0 : t0 + TOKEN_TILE, :], rows, D
            )

    _mxfp8_wq_a(act_pad, wq_a, wq_a_scale, out_pad, num_tokens)
    # Keep QR FP32, WQ_B FP32, and the BF16 flatten on distinct DDR
    # identities.  kv_out_pad is the same [T_MAX, ACT_MAX] FP32 shape, and
    # without a pin the reuse pass can overlay it on q_out_pad before
    # c2a_q_cast finishes — the fused Q error shows up already in the FP32
    # Cube output, while the later BF16 cast / unflat / identity RoPE hops
    # are bit-exact once that buffer is stable.
    qr_fp32_pad: pl.Tensor[[T_MAX, ACT_MAX], pl.FP32, pl.MemRef("c2a_qr_fp32")] = (
        pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    )
    _rms_norm_rows_fp32(
        out_pad, q_norm_weight, qr_fp32_pad, num_tokens, pl.cast(Q_LORA, pl.INT32)
    )
    for blk in pl.spmd(n_copy, name_hint="c2a_qr_gather"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            qr[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(
                pl.cast(
                    qr_fp32_pad[t0 : t0 + TOKEN_TILE, 0 : Q_LORA],
                    target_type=pl.BF16,
                    mode="rint",
                ),
                rows,
                Q_LORA,
            )

    q_out_pad: pl.Tensor[[T_MAX, ACT_MAX], pl.FP32, pl.MemRef("c2a_q_out")] = (
        pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    )
    _mxfp8_wq_b_fp32(qr_fp32_pad, wq_b, wq_b_scale, q_out_pad, num_tokens)
    # Snapshot Cube Q into a buffer whose last use is after KV.  q_out_pad is
    # the same [T_MAX, ACT_MAX] FP32 shape as kv_out_pad; without this copy
    # the reuse pass overlays them while c2a_q_cast is still draining the
    # 128-core WQ_B store (writeback diag: 11% → 3.3%).
    q_width = LOCAL_H * HEAD_DIM
    q_fp32_keep: pl.Tensor[
        [T_MAX, LOCAL_H * HEAD_DIM],
        pl.FP32,
        pl.MemRef("c2a_q_fp32"),
    ] = pl.create_tensor([T_MAX, LOCAL_H * HEAD_DIM], dtype=pl.FP32)
    for blk in pl.spmd(n_copy, name_hint="c2a_q_snap_fp32"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            for c0 in pl.range(0, q_width, ACT_COL_TILE):
                cols = pl.min(ACT_COL_TILE, q_width - c0)
                q_fp32_keep[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE] = pl.set_validshape(
                    q_out_pad[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE],
                    rows,
                    cols,
                )

    kv_act_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.BF16)
    kv_out_pad: pl.Tensor[[T_MAX, ACT_MAX], pl.FP32, pl.MemRef("c2a_kv_out")] = (
        pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    )
    for blk in pl.spmd(n_copy, name_hint="c2a_x_pad2"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            kv_act_pad[t0 : t0 + TOKEN_TILE, 0 : D] = pl.set_validshape(
                x[t0 : t0 + TOKEN_TILE, :], rows, D
            )
    _mxfp8_wkv(kv_act_pad, wkv, wkv_scale, kv_out_pad, num_tokens)
    kv_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.BF16)
    _rms_norm_rows(kv_out_pad, kv_norm_weight, kv_pad, num_tokens, pl.cast(HEAD_DIM, pl.INT32))
    for blk in pl.spmd(n_copy, name_hint="c2a_kv_gather"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            window_kv[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(
                kv_pad[t0 : t0 + TOKEN_TILE, 0 : HEAD_DIM], rows, HEAD_DIM
            )
    cos_il = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.FP32)
    _prepare_rope_interleaved(rope_cos, rope_sin, cos_il, sin_signed, num_tokens, pl.cast(0, pl.INT32))
    _apply_rope_tail_kv(window_kv, cos_il, sin_signed, num_tokens)

    q_flat: pl.Tensor[[T_MAX, LOCAL_H * HEAD_DIM], pl.BF16, pl.MemRef("c2a_q_flat")] = (
        pl.create_tensor([T_MAX, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    )
    for blk in pl.spmd(n_copy, name_hint="c2a_q_cast"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            for c0 in pl.range(0, q_width, ACT_COL_TILE):
                cols = pl.min(ACT_COL_TILE, q_width - c0)
                q_flat[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE] = pl.set_validshape(
                    pl.cast(
                        q_fp32_keep[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE],
                        target_type=pl.BF16,
                        mode="rint",
                    ),
                    rows,
                    cols,
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_q_unflat"):
        for t in pl.range(num_tokens):
            for h in pl.range(LOCAL_H):
                q[t : t + 1, h : h + 1, :] = pl.reshape(
                    q_flat[t : t + 1, h * HEAD_DIM : (h + 1) * HEAD_DIM],
                    [1, 1, HEAD_DIM],
                )
    # Per-head RoPE: materialize a flat HEAD_DIM workspace (reshape alone is not static for @pl.jit.inline).
    q_heads = pl.create_tensor([T_MAX * LOCAL_H, HEAD_DIM], dtype=pl.BF16)
    cos_heads = pl.create_tensor([T_MAX * LOCAL_H, ROPE_DIM], dtype=pl.FP32)
    sin_heads = pl.create_tensor([T_MAX * LOCAL_H, ROPE_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_rope_expand_heads"):
        for t in pl.range(num_tokens):
            for h in pl.range(LOCAL_H):
                dst = t * LOCAL_H + h
                q_heads[dst : dst + 1, :] = pl.reshape(q[t : t + 1, h : h + 1, :], [1, HEAD_DIM])
                cos_heads[dst : dst + 1, :] = cos_il[t : t + 1, :]
                sin_heads[dst : dst + 1, :] = sin_signed[t : t + 1, :]
    head_tokens = pl.cast(num_tokens * LOCAL_H, pl.INT32)
    _apply_rope_tail_hd(q_heads, cos_heads, sin_heads, head_tokens)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_rope_scatter_heads"):
        for t in pl.range(num_tokens):
            for h in pl.range(LOCAL_H):
                dst = t * LOCAL_H + h
                q[t : t + 1, h : h + 1, :] = pl.reshape(q_heads[dst : dst + 1, :], [1, 1, HEAD_DIM])


@pl.jit.inline
def _stage_publish_window(
    window_kv: pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0],
    window_bf16: pl.Tensor,
    window_stage_indices: pl.Tensor[[T_MAX, 128], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Dequant-on-read (needed rows only) + MXFP8 quant-on-write for window KV."""
    ori_blocks = pl.tensor.dim(window_cache, 0)
    cache_flat = pl.reshape(window_cache, [ori_blocks * 128, HEAD_DIM])
    scale_flat = pl.reshape(window_cache_scale, [ori_blocks * 128, HEAD_DIM // WINDOW_CACHE_GROUP])

    # On-demand seed: decode each referenced physical row into query-local
    # staging. E8M0 is a byte scale type, so expand its code explicitly.
    scale_scratch = pl.create_tensor([1, WINDOW_CACHE_GROUP], dtype=pl.FP8E8M0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_win_seed_needed"):
        for t in pl.range(num_tokens):
            for k in pl.range(128):
                pl.write(window_stage_indices, [t, k], pl.cast(-1, pl.INT32))
                idx = pl.read(window_indices, [t, k])
                if idx >= 0:
                    row = pl.cast(idx, pl.INDEX)
                    staged = t * 128 + k
                    for d0 in pl.range(0, HEAD_DIM, WINDOW_CACHE_GROUP):
                        scale_code = pl.read(
                            scale_flat, [row, d0 // WINDOW_CACHE_GROUP]
                        )
                        for j in pl.range(WINDOW_CACHE_GROUP):
                            pl.write(scale_scratch, [0, j], scale_code)
                        codes = pl.reinterpret_view(
                            pl.load(
                                scale_scratch,
                                [0, 0],
                                [1, WINDOW_CACHE_GROUP],
                                target_memory=pl.Mem.Vec,
                            ),
                            pl.UINT8,
                        )
                        exponent = pl.add(
                            pl.cast(pl.cast(codes, pl.FP16), pl.FP32), -127.0
                        )
                        scale = pl.exp(pl.mul(exponent, _LN2))
                        payload = pl.cast(
                            pl.load(
                                cache_flat,
                                [row, d0],
                                [1, WINDOW_CACHE_GROUP],
                                target_memory=pl.Mem.Vec,
                            ),
                            pl.FP32,
                        )
                        window_bf16 = pl.store(
                            pl.cast(pl.mul(payload, scale), pl.BF16, mode="rint"),
                            [staged, d0],
                            window_bf16,
                        )
                    pl.write(window_stage_indices, [t, k], pl.cast(staged, pl.INT32))

    # Same-dispatch publishes override stale cache contents at matching slots.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_win_overlay_pub"):
        for t in pl.range(num_tokens):
            for k in pl.range(128):
                idx = pl.read(window_indices, [t, k])
                if idx >= 0:
                    staged = t * 128 + k
                    for pt in pl.range(num_tokens):
                        slot_i64 = pl.read(window_slots, [pt])
                        if slot_i64 == pl.cast(idx, pl.INT64):
                            window_bf16[staged : staged + 1, :] = window_kv[pt : pt + 1, :]

    # Quant-on-write only published slots (not the full workspace).
    pad_buf = pl.create_tensor([M_TILE, QUANT_K], dtype=pl.FP32)
    mx_nd_flat = pl.create_tensor([1, M_TILE * QUANT_K], dtype=pl.FP8E4M3FN)
    sc_nd_flat = pl.create_tensor([1, M_TILE * (QUANT_K // MX_GROUP)], dtype=pl.FP8E8M0)
    n_scale_g = QUANT_K // WINDOW_CACHE_GROUP
    for _wb in pl.spmd(1, name_hint="c2a_win_requant"):
        for t in pl.range(num_tokens):
            slot_i64 = pl.read(window_slots, [t])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                for d0 in pl.range(0, HEAD_DIM, QUANT_K):
                    pad_buf[0:1, :] = pl.cast(
                        window_kv[t : t + 1, d0 : d0 + QUANT_K],
                        target_type=pl.FP32,
                    )
                    mx, mx_scale = pl.quant_mx(
                        pl.load(pad_buf, [0, 0], [M_TILE, QUANT_K]), group_axis=1
                    )
                    mx_nd_flat = pl.store(
                        pl.reshape(mx, [1, M_TILE * QUANT_K]), [0, 0], mx_nd_flat
                    )
                    sc_nd_flat = pl.store(
                        pl.reshape(mx_scale, [1, M_TILE * (QUANT_K // MX_GROUP)]),
                        [0, 0],
                        sc_nd_flat,
                    )
                    cache_flat = pl.store(
                        pl.load(mx_nd_flat, [0, 0], [1, QUANT_K]),
                        [slot, d0],
                        cache_flat,
                    )
                    scale_g0 = d0 // WINDOW_CACHE_GROUP
                    for g in pl.range(n_scale_g):
                        # quant_mx returns A-scales in MX_A_ZZ physical order:
                        # [M/16, G/2, 16, 2].  For logical row 0, group g is
                        # therefore at (g//2)*16*2 + g%2, not flat offset g.
                        physical_g = (g // 2) * M_TILE * 2 + g % 2
                        pl.write(
                            scale_flat,
                            [slot, scale_g0 + g],
                            pl.read(sc_nd_flat, [0, physical_g]),
                        )


@pl.jit.inline
def _stage_compressor_ratio2(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressor_state_rows: pl.Tensor[[T_DYN], pl.INT64],
    compressor_state: pl.Tensor[[MAX_BATCH_PER_DP, STATE_HEADS, HEAD_DIM], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16],
    publish_mask: pl.Tensor[[T_MAX], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Ratio-2 paged compressor: even pos stores state, odd pos pools + RMS."""
    x_pad = pl.create_tensor([T_MAX, D], dtype=pl.BF16)
    for blk in pl.spmd((num_tokens + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="c2a_cmp_x"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            x_pad[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(x[t0 : t0 + TOKEN_TILE, :], rows, D)

    kv_proj = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.FP32)
    _fp32_linear(x_pad, compressor_wkv, kv_proj, num_tokens)
    _fp32_linear(x_pad, compressor_wgate, score_proj, num_tokens)

    state_flat = pl.reshape(compressor_state, [MAX_BATCH_PER_DP * STATE_HEADS, HEAD_DIM])
    pooled_fp32 = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_ratio2_pool"):
        for t in pl.range(num_tokens):
            zero_i = pl.cast(0, pl.INT32)
            pl.write(publish_mask, [t], zero_i)
            row_i64 = pl.read(compressor_state_rows, [t])
            row = pl.cast(row_i64, pl.INDEX)
            pos = pl.read(position_ids, [t])
            # pos % 2 == 0 → store; else pool.
            pos_odd = pos - (pos // 2) * 2
            if pos_odd == 0:
                state_flat[row * STATE_HEADS + 0 : row * STATE_HEADS + 1, :] = kv_proj[t : t + 1, :]
                state_flat[row * STATE_HEADS + 1 : row * STATE_HEADS + 2, :] = score_proj[t : t + 1, :]
            else:
                prior_kv = state_flat[row * STATE_HEADS + 0 : row * STATE_HEADS + 1, :]
                prior_sc = state_flat[row * STATE_HEADS + 1 : row * STATE_HEADS + 2, :]
                cur_kv = kv_proj[t : t + 1, :]
                cur_sc = score_proj[t : t + 1, :]
                # Softmax over the 2-row score stack, then weighted sum of KV.
                s0 = prior_sc
                s1 = cur_sc
                m = pl.maximum(s0, s1)
                e0 = pl.exp(pl.sub(s0, m))
                e1 = pl.exp(pl.sub(s1, m))
                denom = pl.add(e0, e1)
                w0 = pl.div(e0, denom)
                w1 = pl.div(e1, denom)
                pooled = pl.add(pl.mul(prior_kv, w0), pl.mul(cur_kv, w1))
                pooled_fp32[t : t + 1, :] = pooled
                one_i = pl.cast(1, pl.INT32)
                pl.write(publish_mask, [t], one_i)

    pooled_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    for blk in pl.spmd((num_tokens + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="c2a_pool_pad"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            pooled_pad[t0 : t0 + TOKEN_TILE, 0 : HEAD_DIM] = pl.set_validshape(
                pooled_fp32[t0 : t0 + TOKEN_TILE, :], rows, HEAD_DIM
            )
    latent_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.BF16)
    _rms_norm_rows(pooled_pad, compressor_norm_weight, latent_pad, num_tokens, pl.cast(HEAD_DIM, pl.INT32))
    for blk in pl.spmd((num_tokens + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="c2a_latent_gather"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            latent[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(
                latent_pad[t0 : t0 + TOKEN_TILE, 0 : HEAD_DIM], rows, HEAD_DIM
            )
    # Clear latent rows that did not publish (even positions).
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_latent_mask"):
        for t in pl.range(num_tokens):
            flag = pl.read(publish_mask, [t])
            if flag == 0:
                latent[t : t + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)


@pl.jit.inline
def _stage_publish_compressed_index(
    latent: pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16],
    publish_mask: pl.Tensor[[T_MAX], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    cmp_bf16: pl.Tensor,
    idx_bf16: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """RoPE compressed latent; publish BF16 mirrors and pack cmp as ``mxfp4_bf16``."""

    # Rotate latent with compressed RoPE, then publish.
    latent_rot = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.BF16)
    for blk in pl.spmd(T_MAX // FP4_QUANT_ROWS, name_hint="c2a_latent_zero"):
        t0 = blk * FP4_QUANT_ROWS
        latent_rot[t0 : t0 + FP4_QUANT_ROWS, :] = pl.full(
            [FP4_QUANT_ROWS, HEAD_DIM], dtype=pl.BF16, value=0.0
        )
    for blk in pl.spmd((num_tokens + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="c2a_latent_copy"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            latent_rot[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(
                latent[t0 : t0 + TOKEN_TILE, :], rows, HEAD_DIM
            )
    cos_il = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.FP32)
    _prepare_rope_interleaved(
        compressed_rope_cos, compressed_rope_sin, cos_il, sin_signed, num_tokens, pl.cast(0, pl.INT32)
    )
    _apply_rope_tail_kv(latent_rot, cos_il, sin_signed, num_tokens)

    normalized_bf16 = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.BF16)
    cmp_flat = pl.create_tensor([T_MAX, CMP_PACKED], dtype=pl.UINT8)
    cmp_scale_transposed = pl.create_tensor(
        [HEAD_DIM // COMPRESSED_CACHE_GROUP, T_MAX], dtype=pl.BF16
    )
    for blk in pl.spmd(
        (num_tokens + FP4_QUANT_ROWS - 1) // FP4_QUANT_ROWS,
        name_hint="c2a_cmp_quant",
    ):
        token0 = pl.cast(blk * FP4_QUANT_ROWS, pl.INDEX)
        if token0 < num_tokens:
            cmp_flat, cmp_scale_transposed = _mxfp4_quant_cmp_rows(
                latent_rot,
                normalized_bf16,
                cmp_flat,
                cmp_scale_transposed,
                token0,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_cmp_write"):
        for t in pl.range(num_tokens):
            flag = pl.read(publish_mask, [t])
            slot_i64 = pl.read(compressed_slots, [t])
            if flag != 0:
                if slot_i64 >= 0:
                    slot = pl.cast(slot_i64, pl.INDEX)
                    block_id = slot // 128
                    block_row = slot - block_id * 128
                    cmp_bf16[t : t + 1, :] = latent_rot[t : t + 1, :]
                    # Scatter the byte carrier; cache ABI remains UINT8[..., HEAD_DIM/2].
                    for c0 in pl.range(0, CMP_PACKED, MXFP4_PACK_CHUNK):
                        tile_u8 = pl.load(
                            cmp_flat, [t, c0], [1, MXFP4_PACK_CHUNK], target_memory=pl.Mem.Vec
                        )
                        compressed_cache = pl.store(
                            tile_u8,
                            [block_id, block_row, 0, c0],
                            compressed_cache,
                        )
                    for s0 in pl.range(HEAD_DIM // COMPRESSED_CACHE_GROUP):
                        pl.write(
                            compressed_cache_scale,
                            [block_id, block_row, 0, s0],
                            pl.read(cmp_scale_transposed, [s0, t]),
                        )

    # Index key from *unrotated* latent, then RoPE with compressed tables.
    # Keep buffers at INDEX_DIM (not ACT_MAX) to fit ring-heap on bring-up.
    idx_key_fp32 = pl.create_tensor([T_MAX, INDEX_DIM], dtype=pl.FP32)
    _bf16_linear(
        latent,
        index_wk,
        idx_key_fp32,
        num_tokens,
        pl.cast(HEAD_DIM, pl.INT32),
        pl.cast(INDEX_DIM, pl.INT32),
    )
    idx_key = pl.create_tensor([T_MAX, INDEX_DIM], dtype=pl.BF16)
    n_tiles = (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE
    d_tiles = (INDEX_DIM + RMS_D_TILE - 1) // RMS_D_TILE
    for blk in pl.spmd(n_tiles * d_tiles, name_hint="c2a_idxkey_rms"):
        unit = pl.tile.get_block_idx()
        tg = (unit // d_tiles) * TOKEN_TILE
        d0 = (unit % d_tiles) * RMS_D_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - tg)
        cols = pl.min(RMS_D_TILE, INDEX_DIM - d0)
        rows_i = pl.cast(rows, pl.INDEX)
        cols_i = pl.cast(cols, pl.INDEX)
        if tg < num_tokens and d0 < INDEX_DIM:
            sq_sum = pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0)
            for rd in pl.pipeline(0, INDEX_DIM, RMS_D_TILE, stage=1):
                chunk = pl.slice(
                    idx_key_fp32,
                    [TOKEN_TILE, RMS_D_TILE],
                    [tg, rd],
                    valid_shape=[rows, pl.min(RMS_D_TILE, INDEX_DIM - rd)],
                )
                sq = pl.mul(chunk, chunk)
                sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(sq), [1, TOKEN_TILE]))
            mean = pl.mul(sq_sum, INDEX_DIM_INV)
            rms_arg = pl.add(mean, EPS)
            inv0 = pl.rsqrt(rms_arg, high_precision=True)
            inv_sq = pl.mul(inv0, inv0)
            correction = pl.add(pl.mul(pl.mul(rms_arg, inv_sq), -0.5), 1.5)
            inv = pl.mul(inv0, correction)
            inv_t = pl.reshape(inv, [TOKEN_TILE, 1])
            apply = pl.slice(
                idx_key_fp32, [TOKEN_TILE, RMS_D_TILE], [tg, d0], valid_shape=[rows, cols]
            )
            w_row = pl.cast(
                pl.reshape(index_norm_weight[d0 : d0 + RMS_D_TILE], [1, RMS_D_TILE]), pl.FP32
            )
            scaled = pl.col_expand_mul(pl.row_expand_mul(apply, inv_t), w_row)
            idx_key[tg : tg + TOKEN_TILE, d0 : d0 + RMS_D_TILE] = pl.set_validshape(
                pl.cast(scaled, target_type=pl.BF16, mode="rint"), rows_i, cols_i
            )
    _apply_rope_tail_idx(idx_key, cos_il, sin_signed, num_tokens)

    idx_normalized = pl.create_tensor([T_MAX, INDEX_DIM], dtype=pl.BF16)
    idx_flat = pl.create_tensor([T_MAX, IDX_PACKED], dtype=pl.UINT8)
    idx_scale_transposed = pl.create_tensor(
        [INDEX_DIM // INDEX_CACHE_GROUP, T_MAX], dtype=pl.FP8E8M0
    )
    for blk in pl.spmd(
        (num_tokens + INDEX_QUANT_ROWS - 1) // INDEX_QUANT_ROWS,
        name_hint="c2a_idx_quant",
    ):
        token0 = pl.cast(blk * INDEX_QUANT_ROWS, pl.INDEX)
        if token0 < num_tokens:
            idx_flat, idx_scale_transposed = _mxfp4_quant_idx_rows(
                idx_key,
                idx_normalized,
                idx_flat,
                idx_scale_transposed,
                token0,
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_idx_write"):
        for t in pl.range(num_tokens):
            flag = pl.read(publish_mask, [t])
            slot_i64 = pl.read(compressed_slots, [t])
            if flag != 0:
                if slot_i64 >= 0:
                    slot = pl.cast(slot_i64, pl.INDEX)
                    block_id = slot // 128
                    block_row = slot - block_id * 128
                    idx_bf16[t : t + 1, :] = idx_key[t : t + 1, :]
                    for c0 in pl.range(0, IDX_PACKED, MXFP4_PACK_CHUNK):
                        tile_u8 = pl.load(
                            idx_flat,
                            [t, c0],
                            [1, MXFP4_PACK_CHUNK],
                            target_memory=pl.Mem.Vec,
                        )
                        index_cache = pl.store(
                            tile_u8,
                            [block_id, block_row, 0, c0],
                            index_cache,
                        )
                    for g in pl.range(INDEX_DIM // INDEX_CACHE_GROUP):
                        pl.write(
                            index_cache_scale,
                            [block_id, block_row, 0, g],
                            pl.read(idx_scale_transposed, [g, t]),
                        )

    # ``idx_bf16`` / ``cmp_bf16`` remain same-dispatch publish mirrors;
    # readers expand packed cache into separate workspaces before overlay.


@pl.jit.inline
def _stage_paged_indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_MAX, Q_LORA], pl.BF16],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    idx_bf16: pl.Tensor,
    publish_mask: pl.Tensor[[T_MAX], pl.INT32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Score paged index keys and write physical Top-K rows (sorted ascending)."""
    act_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.BF16)
    out_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    for blk in pl.spmd((num_tokens + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="c2a_idxq_act"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            act_pad[t0 : t0 + TOKEN_TILE, 0 : Q_LORA] = pl.set_validshape(
                qr[t0 : t0 + TOKEN_TILE, :], rows, Q_LORA
            )
    _mxfp8_idx_wq_b(act_pad, index_wq_b, index_wq_b_scale, out_pad, num_tokens)
    idx_q_flat = pl.create_tensor([T_MAX, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    for blk in pl.spmd((num_tokens + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="c2a_idxq_cast"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            idx_w = INDEX_H * INDEX_DIM
            for c0 in pl.range(0, idx_w, ACT_COL_TILE):
                cols = pl.min(ACT_COL_TILE, idx_w - c0)
                chunk = pl.cast(
                    out_pad[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE],
                    target_type=pl.BF16,
                    mode="rint",
                )
                idx_q_flat[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE] = pl.set_validshape(
                    chunk,
                    rows,
                    cols,
                )
    idx_q = pl.reshape(idx_q_flat, [T_MAX, INDEX_H, INDEX_DIM])

    cos_il = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.FP32)
    _prepare_rope_interleaved(rope_cos, rope_sin, cos_il, sin_signed, num_tokens, pl.cast(0, pl.INT32))
    cos_heads = pl.create_tensor([T_MAX * INDEX_H, ROPE_DIM], dtype=pl.FP32)
    sin_heads = pl.create_tensor([T_MAX * INDEX_H, ROPE_DIM], dtype=pl.FP32)
    idx_q_heads = pl.create_tensor([T_MAX * INDEX_H, INDEX_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_idx_rope_expand"):
        for t in pl.range(num_tokens):
            for h in pl.range(INDEX_H):
                dst = t * INDEX_H + h
                idx_q_heads[dst : dst + 1, :] = pl.reshape(idx_q[t : t + 1, h : h + 1, :], [1, INDEX_DIM])
                cos_heads[dst : dst + 1, :] = cos_il[t : t + 1, :]
                sin_heads[dst : dst + 1, :] = sin_signed[t : t + 1, :]
    _apply_rope_tail_idx_heads(
        idx_q_heads, cos_heads, sin_heads, pl.cast(num_tokens * INDEX_H, pl.INT32)
    )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_idx_rope_scatter"):
        for t in pl.range(num_tokens):
            for h in pl.range(INDEX_H):
                dst = t * INDEX_H + h
                idx_q[t : t + 1, h : h + 1, :] = pl.reshape(idx_q_heads[dst : dst + 1, :], [1, 1, INDEX_DIM])

    weights_fp32 = pl.create_tensor([T_MAX, INDEX_H], dtype=pl.FP32)
    x_pad = pl.create_tensor([T_MAX, D], dtype=pl.BF16)
    for blk in pl.spmd((num_tokens + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="c2a_idx_x"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            x_pad[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(x[t0 : t0 + TOKEN_TILE, :], rows, D)
    _bf16_linear(
        x_pad,
        index_weights_proj,
        weights_fp32,
        num_tokens,
        pl.cast(D, pl.INT32),
        pl.cast(INDEX_H, pl.INT32),
    )
    # Scale weights by index_dim^{-1/2} * index_h^{-1/2}.
    for blk in pl.spmd((num_tokens + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="c2a_idx_wscale"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            weights_fp32[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(
                pl.mul(weights_fp32[t0 : t0 + TOKEN_TILE, :], INDEX_SCALE), rows, INDEX_H
            )

    table_w = pl.tensor.dim(index_block_table, 1)
    scores = pl.create_tensor([T_MAX, SCORE_LEN_PAD], dtype=pl.FP32)
    physical = pl.create_tensor([T_MAX, SCORE_LEN_PAD], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_idx_scores_init"):
        for t in pl.range(num_tokens):
            scores[t : t + 1, :] = pl.full([1, SCORE_LEN_PAD], dtype=pl.FP32, value=NEG_INF)
            physical[t : t + 1, :] = pl.full([1, SCORE_LEN_PAD], dtype=pl.INT32, value=-1)

    # Prefetch: expand UINT8 MXFP4 into a *separate* workspace; keep ``idx_bf16``
    # as publish-only (no in-place expand / backup-restore on the publish buffer).
    idx_exp = pl.create_tensor([WS_IDX_ROWS, INDEX_DIM], dtype=pl.BF16)
    block_packed = pl.create_tensor([128, IDX_PACKED], dtype=pl.UINT8)
    block_scale = pl.create_tensor(
        [INDEX_DIM // INDEX_CACHE_GROUP, 128], dtype=pl.FP8E8M0
    )
    for t in pl.range(num_tokens):
        req = pl.cast(pl.read(request_ids, [t]), pl.INDEX)
        clen = pl.read(compressed_lens, [t])
        visible = pl.min(clen, SCORE_LEN)
        n_blocks = (visible + 127) // 128
        for b in pl.range(n_blocks):
            if b < table_w:
                block_id = pl.cast(pl.read(index_block_table, [req, b]), pl.INDEX)
                for _exp in pl.spmd(1, name_hint="c2a_mxfp4_exp_idx"):
                    _mxfp4_expand_idx_block(
                        index_cache,
                        index_cache_scale,
                        block_packed,
                        block_scale,
                        idx_exp,
                        block_id,
                        t * SCORE_LEN + b * 128,
                    )
    # Overlay same-dispatch publishes onto the expand workspace for scoring.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_idx_overlay_pub"):
        for t in pl.range(num_tokens):
            req = pl.cast(pl.read(request_ids, [t]), pl.INDEX)
            visible = pl.min(pl.read(compressed_lens, [t]), SCORE_LEN)
            n_blocks = (visible + 127) // 128
            for pt in pl.range(num_tokens):
                pflag = pl.read(publish_mask, [pt])
                pslot_i64 = pl.read(compressed_slots, [pt])
                same_req = pl.read(request_ids, [pt]) == pl.read(request_ids, [t])
                if pflag != 0 and same_req:
                    if pslot_i64 >= 0:
                        pslot = pl.cast(pslot_i64, pl.INDEX)
                        pblock = pslot // 128
                        prow = pslot - pblock * 128
                        for b in pl.range(n_blocks):
                            if b < table_w:
                                mapped = pl.cast(
                                    pl.read(index_block_table, [req, b]), pl.INDEX
                                )
                                if mapped == pblock:
                                    dst = t * SCORE_LEN + b * 128 + prow
                                    idx_exp[dst : dst + 1, :] = idx_bf16[pt : pt + 1, :]

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="c2a_idx_score",
        optimizations=[pl.cross_core_slot(slot_num=1)],
    ):
        for t in pl.range(num_tokens):
            req = pl.cast(pl.read(request_ids, [t]), pl.INDEX)
            clen = pl.read(compressed_lens, [t])
            visible = pl.min(clen, SCORE_LEN)
            n_blocks = (visible + 127) // 128
            q_heads = idx_q[t : t + 1, :, :]  # [1, INDEX_H, INDEX_DIM]
            q_2d = pl.reshape(q_heads, [INDEX_H, INDEX_DIM])
            w_row = weights_fp32[t : t + 1, :]  # [1, INDEX_H]
            for b in pl.range(n_blocks):
                if b < table_w:
                    block_id = pl.cast(pl.read(index_block_table, [req, b]), pl.INDEX)
                    base = block_id * 128
                    stage_row = t * SCORE_LEN + b * 128
                    keys = idx_exp[stage_row : stage_row + 128, :]  # [128, INDEX_DIM]
                    head_sum = pl.full([1, 128], dtype=pl.FP32, value=0.0)
                    for h0 in pl.range(0, INDEX_H, 16):
                        q_h = q_2d[h0 : h0 + 16, :]
                        w_h = w_row[:, h0 : h0 + 16]
                        logits = pl.matmul(q_h, keys, out_dtype=pl.FP32, b_trans=True)
                        relu = pl.maximum(logits, 0.0)
                        head_sum = pl.add(head_sum, pl.matmul(w_h, relu, out_dtype=pl.FP32))
                    pos0 = b * 128
                    valid = pl.min(128, visible - pos0)
                    scores[t : t + 1, pos0 : pos0 + 128] = pl.set_validshape(head_sum, 1, valid)
                    for off in pl.range(128):
                        if off < valid:
                            phys = pl.cast(base + off, pl.INT32)
                            pl.write(physical, [t, pos0 + off], phys)

    # Top-K via sort32 + mrgsort (N=SCORE_LEN_PAD), then gather physical ids and ascending-sort.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_idx_topk"):
        idx_init = pl.arange(0, [1, SCORE_LEN_PAD], dtype=pl.UINT32)
        for t in pl.range(num_tokens):
            score_row = scores[t : t + 1, :]
            s = pl.sort32(score_row, idx_init)
            s = pl.mrgsort(s, block_len=64)
            s = pl.mrgsort(s, block_len=256)
            pairs = s[:, 0 : 2 * INDEX_TOPK]
            top_logic = pl.gather(pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            # Map logical → physical; invalidate non-finite / out-of-range.
            phys_row = physical[t : t + 1, :]
            gathered = pl.gather(phys_row, dim=-1, index=top_logic)
            # Ascending sort of physical ids for stable sparse gather order.
            phys_f = pl.cast(gathered, pl.FP32)
            # Replace -1 with +inf so they sort last, then restore.
            zero = pl.full([1, INDEX_TOPK], dtype=pl.FP32, value=0.0)
            is_neg = pl.minimum(pl.maximum(pl.sub(zero, phys_f), 0.0), 1.0)
            phys_sort = pl.add(phys_f, pl.mul(is_neg, 1.0e30))
            sort_init = pl.arange(0, [1, INDEX_TOPK], dtype=pl.UINT32)
            # sort32 is descending by value; we want ascending → negate first.
            # For 512 values, the interleaved pair buffer has 1024 positions:
            # sort32 leaves 64-position runs, then 64 -> 256 completes the
            # two required 4-way merge stages.  Larger block_len values are
            # not no-ops and can rotate a lone valid id into a padded lane.
            asc = pl.neg(phys_sort)
            sp = pl.sort32(asc, sort_init)
            sp = pl.mrgsort(sp, block_len=64)
            sp = pl.mrgsort(sp, block_len=256)
            order = pl.gather(sp, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            ordered = pl.gather(gathered, dim=-1, index=order)
            topk_indices[t : t + 1, :] = ordered


@pl.jit.inline
def _stage_sparse_attn_merge(
    q: pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16],
    window_bf16: pl.Tensor[[WS_ORI_ROWS, HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    cmp_bf16: pl.Tensor[[WS_CMP_ROWS, HEAD_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    attended: pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Online-softmax stats for window + compressed sources, merge with sink."""
    win_m = pl.create_tensor([T_MAX, LOCAL_H], dtype=pl.FP32)
    win_l = pl.create_tensor([T_MAX, LOCAL_H], dtype=pl.FP32)
    win_o = pl.create_tensor([T_MAX, LOCAL_H, HEAD_DIM], dtype=pl.FP32)
    cmp_m = pl.create_tensor([T_MAX, LOCAL_H], dtype=pl.FP32)
    cmp_l = pl.create_tensor([T_MAX, LOCAL_H], dtype=pl.FP32)
    cmp_o = pl.create_tensor([T_MAX, LOCAL_H, HEAD_DIM], dtype=pl.FP32)

    attn_head_groups = LOCAL_H // ATTN_HEAD_CHUNK
    # Flatten the complete static tensor before slicing.  Reshaping a
    # [1, H, D] slice directly retained valid_rows=1 on A5, so Cube consumed
    # only the first head of every ATTN_HEAD_CHUNK block.
    q_flat = pl.reshape(q, [T_MAX * LOCAL_H, HEAD_DIM])
    for unit in pl.spmd(
        num_tokens * attn_head_groups,
        name_hint="c2a_attn_window",
        optimizations=[pl.cross_core_slot(slot_num=1)],
    ):
        t = unit // attn_head_groups
        h0 = (unit - t * attn_head_groups) * ATTN_HEAD_CHUNK
        q_row0 = t * LOCAL_H + h0
        q_h = q_flat[q_row0 : q_row0 + ATTN_HEAD_CHUNK, :]
        neg_logits = pl.full(
            [ATTN_HEAD_CHUNK, ATTN_K_TILE], dtype=pl.FP32, value=NEG_INF
        )
        run_m = pl.row_max(neg_logits)
        run_l = pl.row_sum(
            pl.full([ATTN_HEAD_CHUNK, ATTN_K_TILE], dtype=pl.FP32, value=0.0)
        )
        run_o = pl.full([ATTN_HEAD_CHUNK, HEAD_DIM], dtype=pl.FP32, value=0.0)
        for c0, (run_m_iter, run_l_iter, run_o_iter) in pl.range(
            0,
            128,
            ATTN_K_TILE,
            init_values=(run_m, run_l, run_o),
        ):
            win_keys = pl.create_l1([ATTN_K_TILE, HEAD_DIM], pl.BF16)
            win_valid = pl.create_tensor([1, ATTN_K_TILE], dtype=pl.FP32)
            for k in pl.range(ATTN_K_TILE):
                idx = pl.read(window_indices, [t, c0 + k])
                if idx >= 0:
                    row = pl.cast(idx, pl.INDEX)
                    win_keys = pl.gather_row(
                        win_keys, window_bf16, [k, 0], [row, 0], [1, HEAD_DIM]
                    )
                    pl.write(win_valid, [0, k], pl.cast(1.0, pl.FP32))
                else:
                    win_keys = pl.gather_row(
                        win_keys, window_bf16, [k, 0], [0, 0], [1, HEAD_DIM]
                    )
                    pl.write(win_valid, [0, k], pl.cast(0.0, pl.FP32))
            win_logits = pl.mul(
                pl.matmul(q_h, win_keys, out_dtype=pl.FP32, b_trans=True),
                HEAD_SCALE,
            )
            win_inv = pl.sub(
                pl.full([1, ATTN_K_TILE], dtype=pl.FP32, value=1.0), win_valid
            )
            # win_inv/win_valid are [1, K] row vectors.  Generic elementwise
            # broadcasting lowers to TMUL/TADD with mismatched physical tile
            # shapes on A5; use the dedicated column-expansion operations.
            win_logits = pl.col_expand_add(
                win_logits, pl.mul(win_inv, NEG_INF)
            )
            local_m = pl.row_max(win_logits)
            new_m = pl.maximum(run_m_iter, local_m)
            win_exps = pl.exp(pl.row_expand_sub(win_logits, new_m))
            win_exps = pl.col_expand_mul(win_exps, win_valid)
            local_l = pl.row_sum(win_exps)
            scale_old = pl.exp(pl.sub(run_m_iter, new_m))
            new_l = pl.add(pl.mul(run_l_iter, scale_old), local_l)
            scaled_old_o = pl.row_expand_mul(run_o_iter, scale_old)
            win_exps_bf16 = pl.cast(win_exps, target_type=pl.BF16, mode="rint")
            local_od = pl.matmul(win_exps_bf16, win_keys, out_dtype=pl.FP32)
            new_o = pl.add(scaled_old_o, local_od)
            run_m, run_l, run_o = pl.yield_(new_m, new_l, new_o)
        win_m[t : t + 1, h0 : h0 + ATTN_HEAD_CHUNK] = pl.reshape(
            run_m, [1, ATTN_HEAD_CHUNK]
        )
        win_l[t : t + 1, h0 : h0 + ATTN_HEAD_CHUNK] = pl.reshape(
            run_l, [1, ATTN_HEAD_CHUNK]
        )
        win_o[t : t + 1, h0 : h0 + ATTN_HEAD_CHUNK, :] = pl.reshape(
            run_o, [1, ATTN_HEAD_CHUNK, HEAD_DIM]
        )

    for unit in pl.spmd(
        num_tokens * attn_head_groups,
        name_hint="c2a_attn_compressed",
        optimizations=[pl.cross_core_slot(slot_num=1)],
    ):
        t = unit // attn_head_groups
        h0 = (unit - t * attn_head_groups) * ATTN_HEAD_CHUNK
        q_row0 = t * LOCAL_H + h0
        q_h = q_flat[q_row0 : q_row0 + ATTN_HEAD_CHUNK, :]
        neg_logits = pl.full(
            [ATTN_HEAD_CHUNK, ATTN_K_TILE], dtype=pl.FP32, value=NEG_INF
        )
        run_m = pl.row_max(neg_logits)
        run_l = pl.row_sum(
            pl.full([ATTN_HEAD_CHUNK, ATTN_K_TILE], dtype=pl.FP32, value=0.0)
        )
        run_o = pl.full([ATTN_HEAD_CHUNK, HEAD_DIM], dtype=pl.FP32, value=0.0)
        for c0, (run_m_iter, run_l_iter, run_o_iter) in pl.range(
            0,
            INDEX_TOPK,
            ATTN_K_TILE,
            init_values=(run_m, run_l, run_o),
        ):
            cmp_keys = pl.create_l1([ATTN_K_TILE, HEAD_DIM], pl.BF16)
            cmp_valid = pl.create_tensor([1, ATTN_K_TILE], dtype=pl.FP32)
            for k in pl.range(ATTN_K_TILE):
                idx = pl.read(topk_indices, [t, c0 + k])
                if idx >= 0:
                    row = pl.cast(idx, pl.INDEX)
                    cmp_keys = pl.gather_row(
                        cmp_keys, cmp_bf16, [k, 0], [row, 0], [1, HEAD_DIM]
                    )
                    pl.write(cmp_valid, [0, k], pl.cast(1.0, pl.FP32))
                else:
                    cmp_keys = pl.gather_row(
                        cmp_keys, cmp_bf16, [k, 0], [0, 0], [1, HEAD_DIM]
                    )
                    pl.write(cmp_valid, [0, k], pl.cast(0.0, pl.FP32))
            cmp_logits = pl.mul(
                pl.matmul(q_h, cmp_keys, out_dtype=pl.FP32, b_trans=True),
                HEAD_SCALE,
            )
            cmp_inv = pl.sub(
                pl.full([1, ATTN_K_TILE], dtype=pl.FP32, value=1.0), cmp_valid
            )
            cmp_logits = pl.col_expand_add(
                cmp_logits, pl.mul(cmp_inv, NEG_INF)
            )
            local_m = pl.row_max(cmp_logits)
            new_m = pl.maximum(run_m_iter, local_m)
            cmp_exps = pl.exp(pl.row_expand_sub(cmp_logits, new_m))
            cmp_exps = pl.col_expand_mul(cmp_exps, cmp_valid)
            local_l = pl.row_sum(cmp_exps)
            scale_old = pl.exp(pl.sub(run_m_iter, new_m))
            new_l = pl.add(pl.mul(run_l_iter, scale_old), local_l)
            scaled_old_o = pl.row_expand_mul(run_o_iter, scale_old)
            cmp_exps_bf16 = pl.cast(cmp_exps, target_type=pl.BF16, mode="rint")
            local_od = pl.matmul(cmp_exps_bf16, cmp_keys, out_dtype=pl.FP32)
            new_o = pl.add(scaled_old_o, local_od)
            run_m, run_l, run_o = pl.yield_(new_m, new_l, new_o)
        cmp_m[t : t + 1, h0 : h0 + ATTN_HEAD_CHUNK] = pl.reshape(
            run_m, [1, ATTN_HEAD_CHUNK]
        )
        cmp_l[t : t + 1, h0 : h0 + ATTN_HEAD_CHUNK] = pl.reshape(
            run_l, [1, ATTN_HEAD_CHUNK]
        )
        cmp_o[t : t + 1, h0 : h0 + ATTN_HEAD_CHUNK, :] = pl.reshape(
            run_o, [1, ATTN_HEAD_CHUNK, HEAD_DIM]
        )

    # Merge window + compressed + sink.  Keep the scalar statistics 8-wide so
    # their FP32 rows satisfy the 32-byte vector alignment, but scale each
    # head's 512-wide numerator directly.  The transpose+col_expand form used
    # here previously only populated power-of-two lanes on A5.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_attn_merge"):
        for t in pl.range(num_tokens):
            for h0 in pl.range(0, LOCAL_H, ATTN_STAT_PAD):
                sink_t = pl.reshape(attn_sink[h0 : h0 + ATTN_STAT_PAD], [1, ATTN_STAT_PAD])
                wm_t = win_m[t : t + 1, h0 : h0 + ATTN_STAT_PAD]
                cm_t = cmp_m[t : t + 1, h0 : h0 + ATTN_STAT_PAD]
                gmax_t = pl.maximum(pl.maximum(wm_t, cm_t), sink_t)
                den_t = pl.exp(pl.sub(sink_t, gmax_t))
                wl_t = win_l[t : t + 1, h0 : h0 + ATTN_STAT_PAD]
                cl_t = cmp_l[t : t + 1, h0 : h0 + ATTN_STAT_PAD]
                den_t = pl.add(den_t, pl.mul(wl_t, pl.exp(pl.sub(wm_t, gmax_t))))
                den_t = pl.add(den_t, pl.mul(cl_t, pl.exp(pl.sub(cm_t, gmax_t))))
                scale_w = pl.exp(pl.sub(wm_t, gmax_t))
                scale_c = pl.exp(pl.sub(cm_t, gmax_t))
                inv_den = pl.recip(den_t)
                for hi in pl.range(ATTN_STAT_PAD):
                    wo_row = pl.reshape(
                        win_o[t : t + 1, h0 + hi : h0 + hi + 1, :],
                        [1, HEAD_DIM],
                    )
                    co_row = pl.reshape(
                        cmp_o[t : t + 1, h0 + hi : h0 + hi + 1, :],
                        [1, HEAD_DIM],
                    )
                    numerator = pl.add(
                        pl.mul(wo_row, pl.read(scale_w, [0, hi])),
                        pl.mul(co_row, pl.read(scale_c, [0, hi])),
                    )
                    out_row = pl.mul(numerator, pl.read(inv_den, [0, hi]))
                    attended[t : t + 1, h0 + hi : h0 + hi + 1, :] = pl.reshape(
                        pl.cast(out_row, target_type=pl.BF16, mode="rint"),
                        [1, 1, HEAD_DIM],
                    )


@pl.jit.inline
def _stage_project_output(
    attended: pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    att_project: pl.Tensor[[T_MAX, LOCAL_O_GROUPS * O_GROUP_IN], pl.BF16],
    o_latent: pl.Tensor[[T_MAX, LOCAL_O_WIDTH], pl.BF16],
    output_partial: pl.Tensor[[T_MAX, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Inverse RoPE → grouped wo_a → MXFP8 wo_b into FP32 TP partials."""
    cos_il = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.FP32)
    _prepare_rope_interleaved(rope_cos, rope_sin, cos_il, sin_signed, num_tokens, pl.cast(1, pl.INT32))
    # Materialize the projection input instead of passing a reshape alias of
    # ``attended`` to a later task.  The graph scheduler tracks tensor objects,
    # not separately-created reshape handles, so the old aliases could let
    # wo_a start before the attention merge had finished writing its output.
    rope_off = HEAD_DIM - ROPE_DIM
    for unit in pl.spmd(
        num_tokens * LOCAL_H,
        name_hint="c2a_invrope_project",
    ):
        t = unit // LOCAL_H
        h = unit - t * LOCAL_H
        col0 = h * HEAD_DIM
        head = pl.reshape(
            attended[t : t + 1, h : h + 1, :],
            [1, HEAD_DIM],
        )
        chunk = pl.cast(
            pl.slice(head, [1, ROPE_DIM], [0, rope_off]),
            target_type=pl.FP32,
        )
        even = pl.gather(chunk, mask_pattern=pl.tile.MaskPattern.P0101)
        odd = pl.gather(chunk, mask_pattern=pl.tile.MaskPattern.P1010)
        swapped = pl.full(
            [1, ROPE_DIM], dtype=pl.FP32, value=0.0
        )
        swapped = pl.tensor.scatter(
            odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=swapped
        )
        swapped = pl.tensor.scatter(
            even, mask_pattern=pl.tile.MaskPattern.P1010, dst=swapped
        )
        cos_t = cos_il[t : t + 1, :]
        sin_t = sin_signed[t : t + 1, :]
        rot = pl.add(
            pl.mul(chunk, cos_t),
            pl.mul(swapped, sin_t),
        )
        projected_head = pl.assemble(
            head,
            pl.cast(rot, target_type=pl.BF16, mode="rint"),
            [0, rope_off],
        )
        att_project[t : t + 1, col0 : col0 + HEAD_DIM] = projected_head

    # Grouped wo_a: attended [T, LOCAL_H, HEAD_DIM] → flat groups @ wo_a
    # LOCAL_H * HEAD_DIM == LOCAL_O_GROUPS * O_GROUP_IN
    att_flat = att_project
    wo_a_flat = pl.reshape(wo_a, [LOCAL_O_WIDTH, O_GROUP_IN])
    m_tiles = (num_tokens + M_TILE - 1) // M_TILE
    # Do not SPMD over every N_TILE.  Launching m_tiles * 256 mixed AIC+AIV
    # tasks dropped a different 32-col store each compile (group 0/nt=27,
    # then group 3/nt=1): those BF16 lanes stayed zero and MXFP8 wo_b blew
    # up.  Eight group-parallel cubes, each walking its 32 N tiles, keeps
    # the C2V store path in a range the scheduler actually completes.
    n_tiles_o = O_LORA // N_TILE
    for blk in pl.spmd(
        m_tiles * LOCAL_O_GROUPS,
        name_hint="c2a_wo_a",
    ):
        mt = blk // LOCAL_O_GROUPS
        g = blk - mt * LOCAL_O_GROUPS
        t0 = mt * M_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        if t0 < num_tokens:
            c0 = g * O_GROUP_IN
            xg = att_flat[t0 : t0 + M_TILE, c0 : c0 + O_GROUP_IN]
            for nt in pl.range(n_tiles_o):
                wrow0 = g * O_LORA + nt * N_TILE
                x0 = pl.slice(xg, [M_TILE, K_TILE], [0, 0], valid_shape=[rows, K_TILE])
                w0 = wo_a_flat[wrow0 : wrow0 + N_TILE, 0:K_TILE]
                acc = pl.matmul(x0, w0, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.pipeline(K_TILE, O_GROUP_IN, K_TILE, stage=1):
                    xk = pl.slice(xg, [M_TILE, K_TILE], [0, kb], valid_shape=[rows, K_TILE])
                    wk = wo_a_flat[wrow0 : wrow0 + N_TILE, kb : kb + K_TILE]
                    acc = pl.matmul_acc(acc, xk, wk, b_trans=True)
                o_latent[t0 : t0 + M_TILE, wrow0 : wrow0 + N_TILE] = pl.set_validshape(
                    pl.cast(acc, target_type=pl.BF16, mode="rint"), rows, N_TILE
                )

    # o_latent and output_partial already have the exact MX input/output
    # widths.  Keep them as direct producer/consumer tensors; routing through
    # ACT_MAX-sized copies made the composed graph lose the wo_a dependency
    # even though both leaf operators pass independently.
    _mxfp8_wo_b_compact(
        o_latent,
        wo_b,
        wo_b_scale,
        output_partial,
        num_tokens,
    )


# ---------------------------------------------------------------------------
# Top-level orchestration (single-card: no DistributedTensor / TP all-reduce)
# ---------------------------------------------------------------------------


@pl.jit.inline
def _cast_output_local(
    output_partial: pl.Tensor[[T_MAX, D], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """TP=1 local writeback: FP32 o_proj partials → BF16 output (no peer reduce)."""
    n_copy = (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE
    for blk in pl.spmd(n_copy, name_hint="c2a_out_cast"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            output[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(
                pl.cast(
                    output_partial[t0 : t0 + TOKEN_TILE, :],
                    target_type=pl.BF16,
                    mode="rint",
                ),
                rows,
                D,
            )


@pl.jit.inline
def decode_c2a_full(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_state_rows: pl.Tensor[[T_DYN], pl.INT64],
    compressor_state: pl.Tensor[[MAX_BATCH_PER_DP, STATE_HEADS, HEAD_DIM], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    t_dim = pl.tensor.dim(x, 0)
    n_tok = pl.min(num_tokens, t_dim)

    qr = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([T_MAX, LOCAL_H, HEAD_DIM], dtype=pl.BF16)
    window_kv = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.BF16)
    if C2A_STAGE_ID >= 1:
        _stage_qkv_proj_rope(
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            rope_cos,
            rope_sin,
            qr,
            q,
            window_kv,
            n_tok,
        )

    # Static workspaces; only the live prefix [0, blocks*128) is written/read.
    window_bf16 = pl.create_tensor([WS_ORI_ROWS, HEAD_DIM], dtype=pl.BF16)
    window_stage_indices = pl.create_tensor([T_MAX, 128], dtype=pl.INT32)
    # Publish mirrors only carry this dispatch's newly-compressed rows.  The
    # larger per-query cache expansions are allocated in the consumer stages.
    cmp_bf16 = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.BF16)
    idx_bf16 = pl.create_tensor([T_MAX, INDEX_DIM], dtype=pl.BF16)

    if C2A_STAGE_ID >= 2:
        _stage_publish_window(
            window_kv,
            window_slots,
            window_indices,
            window_cache,
            window_cache_scale,
            window_bf16,
            window_stage_indices,
            n_tok,
        )

    latent = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.BF16)
    publish_mask = pl.create_tensor([T_MAX], dtype=pl.INT32)
    if C2A_STAGE_ID >= 3:
        _stage_compressor_ratio2(
            x,
            position_ids,
            compressor_state_rows,
            compressor_state,
            compressor_wkv,
            compressor_wgate,
            compressor_norm_weight,
            latent,
            publish_mask,
            n_tok,
        )

        _stage_publish_compressed_index(
            latent,
            publish_mask,
            compressed_rope_cos,
            compressed_rope_sin,
            compressed_slots,
            compressed_cache,
            compressed_cache_scale,
            index_wk,
            index_norm_weight,
            index_cache,
            index_cache_scale,
            mxfp4_pair_lut,
            cmp_bf16,
            idx_bf16,
            n_tok,
        )


    if C2A_STAGE_ID >= 4:
        _stage_paged_indexer(
            x,
            qr,
            request_ids,
            compressed_lens,
            index_block_table,
            index_cache,
            index_cache_scale,
            idx_bf16,
            publish_mask,
            compressed_slots,
            index_wq_b,
            index_wq_b_scale,
            index_weights_proj,
            rope_cos,
            rope_sin,
            topk_indices,
            n_tok,
        )

    # Pin attended away from q_flat / att_project.  All three are
    # T_MAX * LOCAL_H * HEAD_DIM BF16 elements, so DDR reuse can place the
    # inv-RoPE store on the same address as the attention load without an
    # MTE dependency — the same class of corruption as the old reshape aliases.
    attended: pl.Tensor[
        [T_MAX, LOCAL_H, HEAD_DIM],
        pl.BF16,
        pl.MemRef("c2a_attended"),
    ] = pl.create_tensor([T_MAX, LOCAL_H, HEAD_DIM], dtype=pl.BF16)
    if C2A_STAGE_ID >= 5:
        # Expand compressed UINT8+BF16 scale into a separate workspace; keep ``cmp_bf16`` publish-only.
        cmp_exp = pl.create_tensor([WS_CMP_ROWS, HEAD_DIM], dtype=pl.BF16)
        cmp_block_packed = pl.create_tensor([128, CMP_PACKED], dtype=pl.UINT8)
        cmp_block_scale = pl.create_tensor(
            [HEAD_DIM // COMPRESSED_CACHE_GROUP, 128], dtype=pl.BF16
        )
        # Prefetch each query's visible blocks into its own logical staging rows.
        table_w = pl.tensor.dim(index_block_table, 1)
        for t in pl.range(n_tok):
            req = pl.cast(pl.read(request_ids, [t]), pl.INDEX)
            clen = pl.read(compressed_lens, [t])
            visible = pl.min(clen, SCORE_LEN)
            n_blocks = (visible + 127) // 128
            for b in pl.range(n_blocks):
                if b < table_w:
                    block_id = pl.cast(pl.read(index_block_table, [req, b]), pl.INDEX)
                    for _exp in pl.spmd(1, name_hint="c2a_mxfp4_dequant_cmp"):
                        _mxfp4_dequant_cmp_block(
                            compressed_cache,
                            compressed_cache_scale,
                            cmp_block_packed,
                            cmp_block_scale,
                            cmp_exp,
                            block_id,
                            t * SCORE_LEN + b * 128,
                        )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_cmp_overlay_pub"):
            for t in pl.range(n_tok):
                req = pl.cast(pl.read(request_ids, [t]), pl.INDEX)
                visible = pl.min(pl.read(compressed_lens, [t]), SCORE_LEN)
                n_blocks = (visible + 127) // 128
                for pt in pl.range(n_tok):
                    pflag = pl.read(publish_mask, [pt])
                    pslot_i64 = pl.read(compressed_slots, [pt])
                    same_req = pl.read(request_ids, [pt]) == pl.read(request_ids, [t])
                    if pflag != 0 and same_req:
                        if pslot_i64 >= 0:
                            pslot = pl.cast(pslot_i64, pl.INDEX)
                            pblock = pslot // 128
                            prow = pslot - pblock * 128
                            for b in pl.range(n_blocks):
                                if b < table_w:
                                    mapped = pl.cast(
                                        pl.read(index_block_table, [req, b]), pl.INDEX
                                    )
                                    if mapped == pblock:
                                        dst = t * SCORE_LEN + b * 128 + prow
                                        cmp_exp[dst : dst + 1, :] = cmp_bf16[pt : pt + 1, :]

        # Public Top-K values are physical cache rows. Translate them to the
        # per-query logical staging rows used by the attention gather.
        attn_cmp_indices = pl.create_tensor([T_MAX, INDEX_TOPK], dtype=pl.INT32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_cmp_stage_indices"):
            for t in pl.range(n_tok):
                req = pl.cast(pl.read(request_ids, [t]), pl.INDEX)
                visible = pl.min(pl.read(compressed_lens, [t]), SCORE_LEN)
                n_blocks = (visible + 127) // 128
                for k in pl.range(INDEX_TOPK):
                    pl.write(attn_cmp_indices, [t, k], pl.cast(-1, pl.INT32))
                    physical_i32 = pl.read(topk_indices, [t, k])
                    if physical_i32 >= 0:
                        physical = pl.cast(physical_i32, pl.INDEX)
                        pblock = physical // 128
                        prow = physical - pblock * 128
                        for b in pl.range(n_blocks):
                            if b < table_w:
                                mapped = pl.cast(
                                    pl.read(index_block_table, [req, b]), pl.INDEX
                                )
                                if mapped == pblock:
                                    staged = t * SCORE_LEN + b * 128 + prow
                                    pl.write(
                                        attn_cmp_indices,
                                        [t, k],
                                        pl.cast(staged, pl.INT32),
                                    )

        _stage_sparse_attn_merge(
            q,
            window_bf16,
            window_stage_indices,
            cmp_exp,
            attn_cmp_indices,
            attn_sink,
            attended,
            n_tok,
        )

    output_partial = pl.create_tensor([T_MAX, D], dtype=pl.FP32)
    if C2A_STAGE_ID >= 6:
        att_project: pl.Tensor[
            [T_MAX, LOCAL_O_GROUPS * O_GROUP_IN],
            pl.BF16,
            pl.MemRef("c2a_att_project"),
        ] = pl.create_tensor(
            [T_MAX, LOCAL_O_GROUPS * O_GROUP_IN], dtype=pl.BF16
        )
        o_latent: pl.Tensor[
            [T_MAX, LOCAL_O_WIDTH],
            pl.BF16,
            pl.MemRef("c2a_o_latent"),
        ] = pl.create_tensor([T_MAX, LOCAL_O_WIDTH], dtype=pl.BF16)
        _stage_project_output(
            attended,
            rope_cos,
            rope_sin,
            wo_a,
            wo_b,
            wo_b_scale,
            att_project,
            o_latent,
            output_partial,
            n_tok,
        )
        _cast_output_local(output_partial, output, n_tok)
    else:
        n_copy = (n_tok + TOKEN_TILE - 1) // TOKEN_TILE
        for blk in pl.spmd(n_copy, name_hint="c2a_stage_stub"):
            t0 = blk * TOKEN_TILE
            rows = pl.min(TOKEN_TILE, n_tok - t0)
            if t0 < n_tok:
                output[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(
                    pl.full([TOKEN_TILE, D], dtype=pl.BF16, value=0.0), rows, D
                )
                if C2A_STAGE_ID < 4:
                    topk_indices[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(
                        pl.full([TOKEN_TILE, INDEX_TOPK], dtype=pl.INT32, value=-1), rows, INDEX_TOPK
                    )


@pl.jit
def decode_c2a_full_entry(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_state_rows: pl.Tensor[[T_DYN], pl.INT64],
    compressor_state: pl.InOut[pl.Tensor[[MAX_BATCH_PER_DP, STATE_HEADS, HEAD_DIM], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Single-card ``@pl.jit`` entry: no DistributedTensor / TP all-reduce ABI."""
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    window_slots.bind_dynamic(0, T_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    window_cache_scale.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    compressed_cache_scale.bind_dynamic(0, CMP_BLOCKS_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    compressed_lens.bind_dynamic(0, T_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_cache_scale.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    compressed_rope_cos.bind_dynamic(0, T_DYN)
    compressed_rope_sin.bind_dynamic(0, T_DYN)
    compressor_state_rows.bind_dynamic(0, T_DYN)
    compressed_slots.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)

    decode_c2a_full(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        compressed_cache,
        compressed_cache_scale,
        request_ids,
        compressed_lens,
        index_cache,
        index_cache_scale,
        mxfp4_pair_lut,
        index_block_table,
        position_ids,
        compressed_rope_cos,
        compressed_rope_sin,
        compressor_wkv,
        compressor_wgate,
        compressor_state_rows,
        compressor_state,
        compressor_norm_weight,
        compressed_slots,
        index_wk,
        index_norm_weight,
        index_wq_b,
        index_wq_b_scale,
        index_weights_proj,
        topk_indices,
        output,
        num_tokens,
    )
    return output, topk_indices


def _mxfp8_weight(input_dim: int, output_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    from models.deepseek_v4_1_flash.quantization import pack_mx_b_scale

    weight = torch.randn(input_dim, output_dim).clamp(-8.0, 8.0).to(torch.float8_e4m3fn)
    scale = pack_mx_b_scale(torch.full((input_dim // 32, output_dim), 127, dtype=torch.uint8))
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None:
        scale = scale.view(e8m0)
    return weight, scale


def _e8m0_zeros(*shape: int) -> torch.Tensor:
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    codes = torch.full(shape, 127, dtype=torch.uint8)
    return codes.view(e8m0) if e8m0 is not None else codes


def _u8_zeros(*shape: int) -> torch.Tensor:
    return torch.zeros(*shape, dtype=torch.uint8)


def _import_harness():
    """Load pypto-lib ``golden`` harness, not this directory's ``golden.py``."""
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[2])
    shadowed = "golden" in sys.modules and not hasattr(sys.modules["golden"], "run")
    if shadowed:
        del sys.modules["golden"]
        for key in [k for k in sys.modules if k.startswith("golden.")]:
            del sys.modules[key]
    if sys.path[:1] != [root]:
        sys.path.insert(0, root)
    from golden import ScalarSpec, TensorSpec, ratio_allclose, run

    return ScalarSpec, TensorSpec, ratio_allclose, run


def build_tensor_specs(num_tokens: int = 2, n_blocks: int = 1):
    """Small FLASH-shaped fixture for device vs golden (prefer ``--tp 1``)."""
    ScalarSpec, TensorSpec, _, _ = _import_harness()
    from models.deepseek_v4_1_flash.quantization import build_mxfp4_pair_lut, quantize_mxfp8_cache

    torch.manual_seed(7)
    t = num_tokens
    win_payload, win_scale = quantize_mxfp8_cache(torch.zeros(n_blocks, 128, 1, HEAD_DIM))
    win_scale = _e8m0_zeros(*win_scale.shape) if win_scale.dtype == torch.uint8 else win_scale
    pair_lut = build_mxfp4_pair_lut()
    wq_a, wq_a_scale = _mxfp8_weight(D, Q_LORA)
    wq_b, wq_b_scale = _mxfp8_weight(Q_LORA, LOCAL_H * HEAD_DIM)
    wkv, wkv_scale = _mxfp8_weight(D, HEAD_DIM)
    wo_b, wo_b_scale = _mxfp8_weight(LOCAL_O_WIDTH, D)
    index_wq_b, index_wq_b_scale = _mxfp8_weight(Q_LORA, INDEX_H * INDEX_DIM)

    window_indices = torch.full((t, 128), -1, dtype=torch.int32)
    window_indices[0, 0] = 0
    if t > 1:
        window_indices[1, 0] = 0
        window_indices[1, 1] = 1

    def skip_cmp(actual, expected, **_kwargs):
        del actual, expected
        return True, ""

    skip_cmp.__name__ = "skip_cmp"
    build_tensor_specs.skip_cmp = skip_cmp

    return [
        TensorSpec("x", [t, D], torch.bfloat16, init_value=lambda: torch.randn(t, D, dtype=torch.bfloat16)),
        TensorSpec("wq_a", [D, Q_LORA], torch.float8_e4m3fn, init_value=lambda: wq_a),
        TensorSpec("wq_a_scale", [D // 32, Q_LORA], wq_a_scale.dtype, init_value=lambda: wq_a_scale),
        TensorSpec("q_norm_weight", [Q_LORA], torch.bfloat16, init_value=lambda: torch.ones(Q_LORA, dtype=torch.bfloat16)),
        TensorSpec("wq_b", [Q_LORA, LOCAL_H * HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wq_b),
        TensorSpec("wq_b_scale", [Q_LORA // 32, LOCAL_H * HEAD_DIM], wq_b_scale.dtype, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wkv),
        TensorSpec("wkv_scale", [D // 32, HEAD_DIM], wkv_scale.dtype, init_value=lambda: wkv_scale),
        TensorSpec("kv_norm_weight", [HEAD_DIM], torch.bfloat16, init_value=lambda: torch.ones(HEAD_DIM, dtype=torch.bfloat16)),
        TensorSpec("attn_sink", [LOCAL_H], torch.float32, init_value=lambda: torch.randn(LOCAL_H)),
        TensorSpec(
            "wo_a",
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN],
            torch.bfloat16,
            init_value=lambda: torch.randn(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, dtype=torch.bfloat16) * 0.02,
        ),
        TensorSpec("wo_b", [LOCAL_O_WIDTH, D], torch.float8_e4m3fn, init_value=lambda: wo_b),
        TensorSpec("wo_b_scale", [LOCAL_O_WIDTH // 32, D], wo_b_scale.dtype, init_value=lambda: wo_b_scale),
        TensorSpec("rope_cos", [t, ROPE_DIM // 2], torch.float32, init_value=lambda: torch.ones(t, ROPE_DIM // 2)),
        TensorSpec("rope_sin", [t, ROPE_DIM // 2], torch.float32, init_value=lambda: torch.zeros(t, ROPE_DIM // 2)),
        TensorSpec("window_slots", [t], torch.int64, init_value=lambda: torch.arange(t, dtype=torch.int64)),
        TensorSpec("window_indices", [t, 128], torch.int32, init_value=lambda: window_indices.clone()),
        TensorSpec("window_cache", [n_blocks, 128, 1, HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: win_payload),
        TensorSpec(
            "window_cache_scale",
            [n_blocks, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
            win_scale.dtype,
            init_value=lambda: win_scale,
        ),
        TensorSpec(
            "compressed_cache",
            [n_blocks, 128, 1, CMP_PACKED],
            torch.uint8,
            init_value=lambda: _u8_zeros(n_blocks, 128, 1, CMP_PACKED),
        ),
        TensorSpec(
            "compressed_cache_scale",
            [n_blocks, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            torch.bfloat16,
            init_value=lambda: torch.zeros(n_blocks, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP, dtype=torch.bfloat16),
        ),
        TensorSpec("request_ids", [t], torch.int32, init_value=lambda: torch.zeros(t, dtype=torch.int32)),
        TensorSpec("compressed_lens", [t], torch.int32, init_value=lambda: torch.tensor([0, 1][:t] + [0] * max(0, t - 2), dtype=torch.int32)),
        TensorSpec(
            "index_cache",
            [n_blocks, 128, 1, IDX_PACKED],
            torch.uint8,
            init_value=lambda: _u8_zeros(n_blocks, 128, 1, IDX_PACKED),
        ),
        TensorSpec(
            "index_cache_scale",
            [n_blocks, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP],
            _e8m0_zeros(1).dtype,
            init_value=lambda: _e8m0_zeros(n_blocks, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP),
        ),
        TensorSpec("mxfp4_pair_lut", [2, 256], torch.int16, init_value=lambda: pair_lut.clone()),
        TensorSpec("index_block_table", [1, 4], torch.int32, init_value=lambda: torch.zeros(1, 4, dtype=torch.int32)),
        TensorSpec("position_ids", [t], torch.int32, init_value=lambda: torch.arange(t, dtype=torch.int32)),
        TensorSpec("compressed_rope_cos", [t, ROPE_DIM // 2], torch.float32, init_value=lambda: torch.ones(t, ROPE_DIM // 2)),
        TensorSpec("compressed_rope_sin", [t, ROPE_DIM // 2], torch.float32, init_value=lambda: torch.zeros(t, ROPE_DIM // 2)),
        TensorSpec("compressor_wkv", [D, HEAD_DIM], torch.float32, init_value=lambda: torch.randn(D, HEAD_DIM) * 0.02),
        TensorSpec("compressor_wgate", [D, HEAD_DIM], torch.float32, init_value=lambda: torch.randn(D, HEAD_DIM) * 0.02),
        TensorSpec("compressor_state_rows", [t], torch.int64, init_value=lambda: torch.zeros(t, dtype=torch.int64)),
        TensorSpec(
            "compressor_state",
            [MAX_BATCH_PER_DP, STATE_HEADS, HEAD_DIM],
            torch.float32,
            init_value=lambda: torch.zeros(MAX_BATCH_PER_DP, STATE_HEADS, HEAD_DIM),
        ),
        TensorSpec("compressor_norm_weight", [HEAD_DIM], torch.bfloat16, init_value=lambda: torch.ones(HEAD_DIM, dtype=torch.bfloat16)),
        TensorSpec("compressed_slots", [t], torch.int64, init_value=lambda: torch.tensor([-1, 0][:t] + [-1] * max(0, t - 2), dtype=torch.int64)),
        TensorSpec("index_wk", [HEAD_DIM, INDEX_DIM], torch.bfloat16, init_value=lambda: torch.randn(HEAD_DIM, INDEX_DIM, dtype=torch.bfloat16) * 0.02),
        TensorSpec("index_norm_weight", [INDEX_DIM], torch.bfloat16, init_value=lambda: torch.ones(INDEX_DIM, dtype=torch.bfloat16)),
        TensorSpec("index_wq_b", [Q_LORA, INDEX_H * INDEX_DIM], torch.float8_e4m3fn, init_value=lambda: index_wq_b),
        TensorSpec("index_wq_b_scale", [Q_LORA // 32, INDEX_H * INDEX_DIM], index_wq_b_scale.dtype, init_value=lambda: index_wq_b_scale),
        TensorSpec("index_weights_proj", [D, INDEX_H], torch.bfloat16, init_value=lambda: torch.randn(D, INDEX_H, dtype=torch.bfloat16) * 0.02),
        TensorSpec("topk_indices", [t, INDEX_TOPK], torch.int32),
        TensorSpec("output", [t, D], torch.bfloat16),
        ScalarSpec("num_tokens", torch.int32, t),
    ]


def golden_decode_c2a_full_tensors(tensors) -> None:
    """Fill kernel outputs in-place from the Torch FULL/ratio-2 reference."""
    from models.deepseek_v4_1_flash.quantization import quantize_mxfp4_cache

    def copy_storage(dst: torch.Tensor, src: torch.Tensor) -> None:
        """Copy value tensors normally and one-byte carriers by raw storage."""
        source = src.contiguous()
        if dst.element_size() == 1 and source.element_size() == 1:
            dst.view(torch.uint8).copy_(source.view(torch.uint8))
        else:
            dst.copy_(source.to(dst.dtype))

    def copy_published_rows(
        dst: torch.Tensor,
        src: torch.Tensor,
        slots: torch.Tensor,
        publish: torch.Tensor,
    ) -> None:
        """Apply golden cache writes without requantizing untouched slots."""
        dst_rows = dst.reshape(dst.shape[0] * dst.shape[1], *dst.shape[2:])
        src_rows = src.reshape(src.shape[0] * src.shape[1], *src.shape[2:])
        for token in range(slots.numel()):
            slot = int(slots[token])
            if bool(publish[token]) and 0 <= slot < dst_rows.shape[0]:
                copy_storage(dst_rows[slot], src_rows[slot])

    t = int(tensors["num_tokens"])
    win = tensors["window_cache"]
    win_scale = tensors["window_cache_scale"]
    n_blocks = win.shape[0]
    cmp_bf16 = torch.zeros(n_blocks, 128, 1, HEAD_DIM, dtype=torch.bfloat16)
    idx_bf16 = torch.zeros(n_blocks, 128, 1, INDEX_DIM, dtype=torch.bfloat16)
    cmp_payload, cmp_scale = quantize_mxfp4_cache(
        cmp_bf16, group_size=COMPRESSED_CACHE_GROUP, scale_format="bf16"
    )
    idx_payload, idx_scale = quantize_mxfp4_cache(idx_bf16, group_size=32, scale_format="e8m0")
    if win_scale.dtype != torch.uint8:
        win_scale_u8 = win_scale.view(torch.uint8)
    else:
        win_scale_u8 = win_scale
    result = golden_decode_c2a_full(
        tensors["x"][:t],
        tensors["wq_a"],
        tensors["wq_a_scale"],
        tensors["q_norm_weight"],
        tensors["wq_b"],
        tensors["wq_b_scale"],
        tensors["wkv"],
        tensors["wkv_scale"],
        tensors["kv_norm_weight"],
        tensors["attn_sink"],
        tensors["wo_a"],
        tensors["wo_b"],
        tensors["wo_b_scale"],
        tensors["rope_cos"][:t],
        tensors["rope_sin"][:t],
        tensors["window_slots"][:t],
        tensors["window_indices"][:t],
        win,
        win_scale_u8,
        cmp_payload,
        cmp_scale,
        tensors["request_ids"][:t],
        tensors["compressed_lens"][:t],
        idx_payload,
        idx_scale,
        tensors["index_block_table"],
        tensors["position_ids"][:t],
        tensors["compressed_rope_cos"][:t],
        tensors["compressed_rope_sin"][:t],
        tensors["compressor_wkv"],
        tensors["compressor_wgate"],
        tensors["compressor_state_rows"][:t],
        tensors["compressor_state"],
        tensors["compressor_norm_weight"],
        tensors["compressed_slots"][:t],
        tensors["index_wk"],
        tensors["index_norm_weight"],
        tensors["index_wq_b"],
        tensors["index_wq_b_scale"],
        tensors["index_weights_proj"],
    )
    tensors["output"].zero_()
    tensors["output"][:t] = result.output.to(tensors["output"].dtype)
    tensors["topk_indices"].fill_(-1)
    if result.topk_indices is not None:
        k = min(INDEX_TOPK, result.topk_indices.shape[-1])
        tensors["topk_indices"][:t, :k] = result.topk_indices[:t, :k]
    window_slots = tensors["window_slots"][:t]
    window_publish = window_slots >= 0
    copy_published_rows(tensors["window_cache"], result.window_cache, window_slots, window_publish)
    copy_published_rows(
        tensors["window_cache_scale"], result.window_cache_scale, window_slots, window_publish
    )
    compressed_slots = tensors["compressed_slots"][:t]
    compressed_publish = (tensors["position_ids"][:t] % 2 != 0) & (compressed_slots >= 0)
    if result.compressed_cache is not None:
        copy_published_rows(
            tensors["compressed_cache"], result.compressed_cache, compressed_slots, compressed_publish
        )
    if result.compressed_cache_scale is not None:
        copy_published_rows(
            tensors["compressed_cache_scale"],
            result.compressed_cache_scale,
            compressed_slots,
            compressed_publish,
        )
    if result.index_cache is not None:
        copy_published_rows(
            tensors["index_cache"], result.index_cache, compressed_slots, compressed_publish
        )
    if result.index_cache_scale is not None:
        copy_published_rows(
            tensors["index_cache_scale"],
            result.index_cache_scale,
            compressed_slots,
            compressed_publish,
        )
    if result.compressor_state is not None:
        copy_storage(tensors["compressor_state"], result.compressor_state)


__all__ = [
    "golden_decode_c2a_full",
    "golden_decode_c2a_full_tensors",
    "decode_c2a_full",
    "decode_c2a_full_entry",
    "build_tensor_specs",
]


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="decode_c2a_full golden smoke / device vs golden")
    parser.add_argument("-p", "--platform", type=str, default=None, choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str, default="0")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--tp", type=int, default=None, help="forwarded via argv for config.TP_SIZE")
    parser.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=["pad", "qkv", "window", "compress", "indexer", "attn", "full"],
        help="bisect prefix: pad|qkv|window|compress|indexer|attn|full",
    )
    args, _unknown = parser.parse_known_args()

    if args.platform is None and not args.compile_only:
        from models.deepseek_v4_1_flash._golden_smoke import run_attention_golden

        run_attention_golden(golden_decode_c2a_full, ratio=2, mode="full")
        sys.exit(0)

    if TP_SIZE != 1:
        raise SystemExit(
            f"single-card entry requires --tp 1 (got TP_SIZE={TP_SIZE}); "
            "DistributedTensor TP all-reduce is stubbed out"
        )

    _, _, ratio_allclose, run = _import_harness()

    specs = build_tensor_specs()
    skip = build_tensor_specs.skip_cmp

    def storage_equal(actual, expected, **_kwargs):
        actual_storage = actual.contiguous()
        expected_storage = expected.contiguous()
        if actual_storage.element_size() == 1 and expected_storage.element_size() == 1:
            actual_storage = actual_storage.view(torch.uint8)
            expected_storage = expected_storage.view(torch.uint8)
        ok = torch.equal(actual_storage, expected_storage)
        if ok:
            return True, ""
        mismatch_mask = actual_storage != expected_storage
        mismatches = torch.count_nonzero(mismatch_mask).item()
        details = []
        for coord in mismatch_mask.nonzero()[:10]:
            index = tuple(int(value) for value in coord)
            details.append(
                f"{index}: actual={actual_storage[index].item()}, "
                f"expected={expected_storage[index].item()}"
            )
        first = "; ".join(details)
        return False, (
            f"    storage mismatch: {mismatches}/{actual_storage.numel()} elements\n"
            f"    first mismatches: {first}"
        )

    storage_equal.__name__ = "storage_equal"
    exact_values = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    compare_fn = {
        "output": skip if args.stage != "full" else ratio_allclose(atol=2e-2, rtol=1.0 / 32, max_error_ratio=0.05, valid_rows=2),
        "topk_indices": skip if C2A_STAGE_ID < 4 else ratio_allclose(atol=0, rtol=0, max_error_ratio=0.0, valid_rows=2),
        "window_cache": skip if C2A_STAGE_ID < 2 else storage_equal,
        "window_cache_scale": skip if C2A_STAGE_ID < 2 else storage_equal,
        "compressed_cache": skip if C2A_STAGE_ID < 3 else storage_equal,
        "compressed_cache_scale": skip if C2A_STAGE_ID < 3 else exact_values,
        "index_cache": skip if C2A_STAGE_ID < 3 else storage_equal,
        "index_cache_scale": skip if C2A_STAGE_ID < 3 else storage_equal,
        "compressor_state": skip if C2A_STAGE_ID < 3 else ratio_allclose(
            atol=2e-5, rtol=2e-5, max_error_ratio=0.0
        ),
    }
    print(f"[STAGE] {args.stage} (C2A_STAGE_ID={C2A_STAGE_ID}) single-card TP={TP_SIZE} ring_heap={RING_HEAP_4G}")
    result = run(
        fn=decode_c2a_full_entry,
        specs=specs,
        golden_fn=golden_decode_c2a_full_tensors,
        rtol=2e-2,
        atol=2e-2,
        compare_fn=compare_fn,
        config=dict(
            platform=args.platform,
            device_id=int(args.device),
            ring_heap=RING_HEAP_4G,
        ),
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
