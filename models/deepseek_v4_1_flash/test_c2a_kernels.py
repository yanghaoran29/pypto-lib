# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone @pl.jit wrappers around decode_c2a_full leaf/stage helpers.

Each case is a small ABI (few tensors, no 50-arg entry). golden_fn is omitted
so a PASS means compile+dispatch survived, not numeric match.
Force ``--tp 1`` before importing config so LOCAL_H=64 fits a single card.
"""

from __future__ import annotations

import argparse
import sys
import traceback

if not any(arg == "--tp" or arg.startswith("--tp=") for arg in sys.argv):
    sys.argv = [sys.argv[0], "--tp", "1", *sys.argv[1:]]

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash.decode_c2a_full import (
    ACT_MAX,
    CMP_BLOCKS_DYN,
    CMP_PACKED,
    COMPRESSED_CACHE_GROUP,
    D,
    DECODE_MAX_TOKENS,
    HEAD_DIM,
    INDEX_BLOCKS_DYN,
    INDEX_CACHE_GROUP,
    INDEX_DIM,
    INDEX_H,
    INDEX_TOPK,
    IDX_PACKED,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    MAX_BATCH_PER_DP,
    O_GROUP_IN,
    O_LORA,
    ORI_BLOCKS_DYN,
    Q_LORA,
    ROPE_DIM,
    STATE_HEADS,
    T_DYN,
    T_MAX,
    TABLE_DYN,
    TP_SIZE,
    WINDOW_CACHE_GROUP,
    WS_CMP_ROWS,
    WS_IDX_ROWS,
    WS_ORI_ROWS,
    B_DYN,
    _apply_rope_tail_hd,
    _apply_rope_tail_idx,
    _apply_rope_tail_kv,
    _bf16_linear,
    _e8m0_zeros,
    _fp32_linear,
    _import_harness,
    _mxfp8_idx_wq_b,
    _mxfp8_weight,
    _mxfp8_wkv,
    _mxfp8_wo_b,
    _mxfp8_wq_a,
    _mxfp8_wq_b,
    _prepare_rope_interleaved,
    _rms_norm_rows,
    _stage_compressor_ratio2,
    _stage_paged_indexer,
    _stage_project_output,
    _stage_publish_compressed_index,
    _stage_publish_window,
    _stage_qkv_proj_rope,
    _stage_sparse_attn_merge,
    _u8_zeros,
)

N_TOK = 2
N_BLOCKS = 1

ScalarSpec, TensorSpec, _, run = _import_harness()


def _skip(*_args, **_kwargs):
    return True, ""


_skip.__name__ = "skip_cmp"


def _ts(name, shape, dtype, init=None):
    return TensorSpec(name, list(shape), dtype, init_value=init)


def _sc(name, dtype, value):
    return ScalarSpec(name, dtype, value)


def _randn_bf16(*shape):
    return lambda: torch.randn(*shape, dtype=torch.bfloat16)


def _ones(*shape, dtype=torch.float32):
    return lambda: torch.ones(*shape, dtype=dtype)


def _zeros(*shape, dtype=torch.float32):
    return lambda: torch.zeros(*shape, dtype=dtype)


# ---------------------------------------------------------------------------
# Tiny ABI probes (no decode helpers)
# ---------------------------------------------------------------------------
_TINY = 32
_TINY_TILE = 16
_MX_PROBE_M = 64
_MX_PROBE_K = 256
_MX_PROBE_N = 64
_MX_PROBE_GROUPS = _MX_PROBE_K // 32


@pl.jit
def k_tiny_add(
    x: pl.Tensor[[_TINY, _TINY], pl.FP32],
    y: pl.Out[pl.Tensor[[_TINY, _TINY], pl.FP32]],
):
    for r in pl.parallel(0, _TINY, _TINY_TILE):
        for c in pl.range(0, _TINY, _TINY_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="tiny_add"):
                y[r : r + _TINY_TILE, c : c + _TINY_TILE] = pl.add(
                    x[r : r + _TINY_TILE, c : c + _TINY_TILE], 1.0
                )
    return y


@pl.jit
def k_copy_d(
    x: pl.Tensor[[T_MAX, D], pl.BF16],
    y: pl.Out[pl.Tensor[[T_MAX, D], pl.BF16]],
):
    for blk in pl.spmd((T_MAX + 7) // 8, name_hint="copy_d"):
        t0 = blk * 8
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_d_tile"):
            y[t0 : t0 + 8, :] = x[t0 : t0 + 8, :]
    return y


@pl.jit
def k_fp4_touch(
    cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8]],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
):
    """Minimal UINT8 MXFP4 InOut + pair-LUT touch (PR #1210 ABI)."""
    del mxfp4_pair_lut
    cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    blocks = pl.tensor.dim(cache, 0)
    flat = pl.reshape(cache, [blocks * 128, CMP_PACKED])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp4_touch"):
        row = flat[0:1, :]
        flat[0:1, :] = row


@pl.jit
def k_dist_unused(
    x: pl.Tensor[[_TINY, _TINY], pl.FP32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    y: pl.Out[pl.Tensor[[_TINY, _TINY], pl.FP32]],
):
    for r in pl.parallel(0, _TINY, _TINY_TILE):
        for c in pl.range(0, _TINY, _TINY_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="dist_unused"):
                y[r : r + _TINY_TILE, c : c + _TINY_TILE] = pl.add(
                    x[r : r + _TINY_TILE, c : c + _TINY_TILE], 1.0
                )
    return y


@pl.jit.incore
def _mx_probe_quant(
    a: pl.Tensor[[_MX_PROBE_M, _MX_PROBE_K], pl.FP32],
    a_quant: pl.Out[pl.Tensor[[_MX_PROBE_M, _MX_PROBE_K], pl.FP8E4M3FN]],
    a_scale: pl.Out[pl.Tensor[[1, _MX_PROBE_M * _MX_PROBE_GROUPS], pl.FP8E8M0]],
):
    """Exact shape from PyPTO's supported A5 quantized-matmul ST."""
    quant, scale = pl.quant_mx(pl.load(a, [0, 0], [_MX_PROBE_M, _MX_PROBE_K]), group_axis=1)
    a_quant = pl.store(quant, [0, 0], a_quant)
    a_scale = pl.store(
        pl.reshape(scale, [1, _MX_PROBE_M * _MX_PROBE_GROUPS]), [0, 0], a_scale
    )
    return a_quant, a_scale


@pl.jit.incore
def _mx_probe_matmul(
    a_quant: pl.Tensor[[_MX_PROBE_M, _MX_PROBE_K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[1, _MX_PROBE_M * _MX_PROBE_GROUPS], pl.FP8E8M0],
    b: pl.Tensor[[_MX_PROBE_K, _MX_PROBE_N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[1, _MX_PROBE_GROUPS * _MX_PROBE_N], pl.FP8E8M0],
    out: pl.Out[pl.Tensor[[_MX_PROBE_M, _MX_PROBE_N], pl.FP32]],
):
    a_scale_mx = pl.tensor.view(
        a_scale, [_MX_PROBE_M, _MX_PROBE_GROUPS], layout=pl.MX_A_ZZ
    )
    b_scale_mx = pl.tensor.view(
        b_scale, [_MX_PROBE_GROUPS, _MX_PROBE_N], layout=pl.MX_B_NN
    )
    lhs = pl.load(a_quant, [0, 0], [_MX_PROBE_M, _MX_PROBE_K])
    lhs_scale = pl.load(a_scale_mx, [0, 0], [_MX_PROBE_M, _MX_PROBE_GROUPS])
    rhs = pl.load(b, [0, 0], [_MX_PROBE_K, _MX_PROBE_N])
    rhs_scale = pl.load(b_scale_mx, [0, 0], [_MX_PROBE_GROUPS, _MX_PROBE_N])
    return pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)


@pl.jit
def k_mx_supported_tile(
    a: pl.Tensor[[_MX_PROBE_M, _MX_PROBE_K], pl.FP32],
    b: pl.Tensor[[_MX_PROBE_K, _MX_PROBE_N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[1, _MX_PROBE_GROUPS * _MX_PROBE_N], pl.FP8E8M0],
    a_quant: pl.Out[pl.Tensor[[_MX_PROBE_M, _MX_PROBE_K], pl.FP8E4M3FN]],
    a_scale: pl.Out[pl.Tensor[[1, _MX_PROBE_M * _MX_PROBE_GROUPS], pl.FP8E8M0]],
    out: pl.Out[pl.Tensor[[_MX_PROBE_M, _MX_PROBE_N], pl.FP32]],
):
    a_quant, a_scale = _mx_probe_quant(a, a_quant, a_scale)
    return _mx_probe_matmul(a_quant, a_scale, b, b_scale, out)


# ---------------------------------------------------------------------------
# Leaf helpers
# ---------------------------------------------------------------------------
@pl.jit
def k_rms(
    x: pl.Tensor[[T_MAX, ACT_MAX], pl.FP32],
    weight: pl.Tensor[[Q_LORA], pl.BF16],
    y: pl.Out[pl.Tensor[[T_MAX, ACT_MAX], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    _rms_norm_rows(x, weight, y, num_tokens, pl.cast(Q_LORA, pl.INT32))
    return y


@pl.jit
def k_rope_prep(
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    cos_il: pl.Out[pl.Tensor[[T_MAX, ROPE_DIM], pl.FP32]],
    sin_signed: pl.Out[pl.Tensor[[T_MAX, ROPE_DIM], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    _prepare_rope_interleaved(
        rope_cos, rope_sin, cos_il, sin_signed, num_tokens, pl.cast(0, pl.INT32)
    )
    return cos_il, sin_signed


@pl.jit
def k_rope_kv(
    values: pl.InOut[pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16]],
    cos_il: pl.Tensor[[T_MAX, ROPE_DIM], pl.FP32],
    sin_signed: pl.Tensor[[T_MAX, ROPE_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    _apply_rope_tail_kv(values, cos_il, sin_signed, num_tokens)


@pl.jit
def k_rope_hd(
    values: pl.InOut[pl.Tensor[[T_MAX * LOCAL_H, HEAD_DIM], pl.BF16]],
    cos_il: pl.Tensor[[T_MAX * LOCAL_H, ROPE_DIM], pl.FP32],
    sin_signed: pl.Tensor[[T_MAX * LOCAL_H, ROPE_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    _apply_rope_tail_hd(values, cos_il, sin_signed, num_tokens)


@pl.jit
def k_rope_idx(
    values: pl.InOut[pl.Tensor[[T_MAX, INDEX_DIM], pl.BF16]],
    cos_il: pl.Tensor[[T_MAX, ROPE_DIM], pl.FP32],
    sin_signed: pl.Tensor[[T_MAX, ROPE_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    _apply_rope_tail_idx(values, cos_il, sin_signed, num_tokens)


@pl.jit
def k_mx_wq_a(
    act: pl.Tensor[[T_MAX, ACT_MAX], pl.BF16],
    weight: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    weight_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[T_MAX, ACT_MAX], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    _mxfp8_wq_a(act, weight, weight_scale, out, num_tokens)
    return out


@pl.jit
def k_mx_wkv(
    act: pl.Tensor[[T_MAX, ACT_MAX], pl.BF16],
    weight: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    weight_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[T_MAX, ACT_MAX], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    _mxfp8_wkv(act, weight, weight_scale, out, num_tokens)
    return out


@pl.jit
def k_mx_idx(
    act: pl.Tensor[[T_MAX, ACT_MAX], pl.BF16],
    weight: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    weight_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[T_MAX, ACT_MAX], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    _mxfp8_idx_wq_b(act, weight, weight_scale, out, num_tokens)
    return out


@pl.jit
def k_mx_wo(
    act: pl.Tensor[[T_MAX, ACT_MAX], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    weight_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[T_MAX, ACT_MAX], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    _mxfp8_wo_b(act, weight, weight_scale, out, num_tokens)
    return out


@pl.jit
def k_mx_wq_b(
    act: pl.Tensor[[T_MAX, ACT_MAX], pl.BF16],
    weight: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    weight_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[T_MAX, ACT_MAX], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    _mxfp8_wq_b(act, weight, weight_scale, out, num_tokens)
    return out


@pl.jit
def k_bf16_linear(
    act: pl.Tensor[[T_MAX, D], pl.BF16],
    weight: pl.Tensor[[D, INDEX_H], pl.BF16],
    out: pl.Out[pl.Tensor[[T_MAX, INDEX_H], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    _bf16_linear(act, weight, out, num_tokens, pl.cast(D, pl.INT32), pl.cast(INDEX_H, pl.INT32))
    return out


@pl.jit
def k_fp32_linear(
    act: pl.Tensor[[T_MAX, D], pl.BF16],
    weight: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    out: pl.Out[pl.Tensor[[T_MAX, HEAD_DIM], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    _fp32_linear(act, weight, out, num_tokens)
    return out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
@pl.jit
def k_window(
    window_kv: pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]
    ],
    window_bf16: pl.InOut[pl.Tensor[[WS_ORI_ROWS, HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    window_slots.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    window_cache_scale.bind_dynamic(0, ORI_BLOCKS_DYN)
    _stage_publish_window(
        window_kv, window_slots, window_cache, window_cache_scale, window_bf16, num_tokens
    )


@pl.jit
def k_compressor(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressor_state_rows: pl.Tensor[[T_DYN], pl.INT64],
    compressor_state: pl.InOut[pl.Tensor[[MAX_BATCH_PER_DP, STATE_HEADS, HEAD_DIM], pl.FP32]],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Out[pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16]],
    publish_mask: pl.Out[pl.Tensor[[T_MAX], pl.INT32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    compressor_state_rows.bind_dynamic(0, T_DYN)
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
        num_tokens,
    )
    return latent, publish_mask


@pl.jit
def k_qkv(
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
    qr: pl.Out[pl.Tensor[[T_MAX, Q_LORA], pl.BF16]],
    q: pl.Out[pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16]],
    window_kv: pl.Out[pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
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
        num_tokens,
    )
    return qr, q, window_kv


@pl.jit
def k_pub_cmp_idx(
    latent: pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16],
    publish_mask: pl.Tensor[[T_MAX], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    cmp_bf16: pl.InOut[pl.Tensor[[WS_CMP_ROWS, HEAD_DIM], pl.BF16]],
    idx_bf16: pl.InOut[pl.Tensor[[WS_IDX_ROWS, INDEX_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    compressed_rope_cos.bind_dynamic(0, T_DYN)
    compressed_rope_sin.bind_dynamic(0, T_DYN)
    compressed_slots.bind_dynamic(0, T_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    compressed_cache_scale.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_cache_scale.bind_dynamic(0, INDEX_BLOCKS_DYN)
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
        num_tokens,
    )


@pl.jit
def k_indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_MAX, Q_LORA], pl.BF16],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    idx_bf16: pl.Tensor[[WS_IDX_ROWS, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    compressed_lens.bind_dynamic(0, T_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    _stage_paged_indexer(
        x,
        qr,
        request_ids,
        compressed_lens,
        index_block_table,
        idx_bf16,
        index_wq_b,
        index_wq_b_scale,
        index_weights_proj,
        rope_cos,
        rope_sin,
        topk_indices,
        num_tokens,
    )
    return topk_indices


@pl.jit
def k_attn(
    q: pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16],
    window_bf16: pl.Tensor[[WS_ORI_ROWS, HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    cmp_bf16: pl.Tensor[[WS_CMP_ROWS, HEAD_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    attended: pl.Out[pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    window_indices.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    _stage_sparse_attn_merge(
        q, window_bf16, window_indices, cmp_bf16, topk_indices, attn_sink, attended, num_tokens
    )
    return attended


@pl.jit
def k_oproj(
    attended: pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    output_partial: pl.Out[pl.Tensor[[T_MAX, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    att_project = pl.create_tensor(
        [T_MAX, LOCAL_O_GROUPS * O_GROUP_IN], dtype=pl.BF16
    )
    o_latent = pl.create_tensor([T_MAX, LOCAL_O_WIDTH], dtype=pl.BF16)
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
        num_tokens,
    )
    return output_partial


def _mx_specs(k_dim: int, n_dim: int):
    weight, scale = _mxfp8_weight(k_dim, n_dim)
    return [
        _ts("act", [T_MAX, ACT_MAX], torch.bfloat16, _randn_bf16(T_MAX, ACT_MAX)),
        _ts("weight", [k_dim, n_dim], torch.float8_e4m3fn, lambda w=weight: w),
        _ts("weight_scale", [k_dim // 32, n_dim], scale.dtype, lambda s=scale: s),
        _ts("out", [T_MAX, ACT_MAX], torch.float32),
        _sc("num_tokens", torch.int32, N_TOK),
    ]


def _win_payload():
    from models.deepseek_v4_1_flash.quantization import quantize_mxfp8_cache

    payload, scale = quantize_mxfp8_cache(torch.zeros(N_BLOCKS, 128, 1, HEAD_DIM))
    if scale.dtype == torch.uint8:
        scale = _e8m0_zeros(*scale.shape)
    return payload, scale


def _window_indices():
    idx = torch.full((N_TOK, 128), -1, dtype=torch.int32)
    idx[0, 0] = 0
    if N_TOK > 1:
        idx[1, 0] = 0
        idx[1, 1] = 1
    return idx


def _topk():
    idx = torch.full((N_TOK, INDEX_TOPK), -1, dtype=torch.int32)
    idx[0, 0] = 0
    return idx


def _build_cases() -> dict[str, tuple[object, list]]:
    wq_a, wq_a_s = _mxfp8_weight(D, Q_LORA)
    wq_b, wq_b_s = _mxfp8_weight(Q_LORA, LOCAL_H * HEAD_DIM)
    wkv, wkv_s = _mxfp8_weight(D, HEAD_DIM)
    wo_b, wo_b_s = _mxfp8_weight(LOCAL_O_WIDTH, D)
    idx_w, idx_s = _mxfp8_weight(Q_LORA, INDEX_H * INDEX_DIM)
    win_p, win_s = _win_payload()
    u8_cmp = _u8_zeros(N_BLOCKS, 128, 1, CMP_PACKED)
    u8_idx = _u8_zeros(N_BLOCKS, 128, 1, IDX_PACKED)
    from models.deepseek_v4_1_flash.quantization import build_mxfp4_pair_lut

    pair_lut = build_mxfp4_pair_lut()
    mask = torch.zeros(T_MAX, dtype=torch.int32)
    mask[1] = 1

    return {
        "mx_supported_tile": (
            k_mx_supported_tile,
            [
                _ts("a", [_MX_PROBE_M, _MX_PROBE_K], torch.float32, torch.randn),
                _ts(
                    "b",
                    [_MX_PROBE_K, _MX_PROBE_N],
                    torch.float8_e4m3fn,
                    lambda: torch.randn(_MX_PROBE_K, _MX_PROBE_N)
                    .clamp(-8.0, 8.0)
                    .to(torch.float8_e4m3fn),
                ),
                _ts(
                    "b_scale",
                    [1, _MX_PROBE_GROUPS * _MX_PROBE_N],
                    _e8m0_zeros(1).dtype,
                    lambda: _e8m0_zeros(1, _MX_PROBE_GROUPS * _MX_PROBE_N),
                ),
                _ts("a_quant", [_MX_PROBE_M, _MX_PROBE_K], torch.float8_e4m3fn),
                _ts(
                    "a_scale",
                    [1, _MX_PROBE_M * _MX_PROBE_GROUPS],
                    _e8m0_zeros(1).dtype,
                ),
                _ts("out", [_MX_PROBE_M, _MX_PROBE_N], torch.float32),
            ],
        ),
        "tiny_add": (
            k_tiny_add,
            [
                _ts("x", [_TINY, _TINY], torch.float32, torch.randn),
                _ts("y", [_TINY, _TINY], torch.float32),
            ],
        ),
        "dist_unused": (
            k_dist_unused,
            [
                _ts("x", [_TINY, _TINY], torch.float32, torch.randn),
                _ts("output_window", [DECODE_MAX_TOKENS, D], torch.float32),
                _ts("output_arrived", [TP_SIZE, 1], torch.int32),
                _ts("y", [_TINY, _TINY], torch.float32),
            ],
        ),
        "fp4_touch": (
            k_fp4_touch,
            [
                _ts(
                    "cache",
                    [N_BLOCKS, 128, 1, CMP_PACKED],
                    torch.uint8,
                    lambda: _u8_zeros(N_BLOCKS, 128, 1, CMP_PACKED),
                ),
                _ts("mxfp4_pair_lut", [2, 256], torch.int16, lambda: pair_lut.clone()),
            ],
        ),
        "copy_d": (
            k_copy_d,
            [
                _ts("x", [T_MAX, D], torch.bfloat16, _randn_bf16(T_MAX, D)),
                _ts("y", [T_MAX, D], torch.bfloat16),
            ],
        ),
        "rms": (
            k_rms,
            [
                _ts("x", [T_MAX, ACT_MAX], torch.float32, torch.randn),
                _ts("weight", [Q_LORA], torch.bfloat16, _ones(Q_LORA, dtype=torch.bfloat16)),
                _ts("y", [T_MAX, ACT_MAX], torch.bfloat16),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
        "rope_prep": (
            k_rope_prep,
            [
                _ts("rope_cos", [N_TOK, ROPE_DIM // 2], torch.float32, _ones(N_TOK, ROPE_DIM // 2)),
                _ts("rope_sin", [N_TOK, ROPE_DIM // 2], torch.float32, _zeros(N_TOK, ROPE_DIM // 2)),
                _ts("cos_il", [T_MAX, ROPE_DIM], torch.float32),
                _ts("sin_signed", [T_MAX, ROPE_DIM], torch.float32),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
        "rope_kv": (
            k_rope_kv,
            [
                _ts("values", [T_MAX, HEAD_DIM], torch.bfloat16, _randn_bf16(T_MAX, HEAD_DIM)),
                _ts("cos_il", [T_MAX, ROPE_DIM], torch.float32, _ones(T_MAX, ROPE_DIM)),
                _ts("sin_signed", [T_MAX, ROPE_DIM], torch.float32, _zeros(T_MAX, ROPE_DIM)),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
        "rope_hd": (
            k_rope_hd,
            [
                _ts(
                    "values",
                    [T_MAX * LOCAL_H, HEAD_DIM],
                    torch.bfloat16,
                    _randn_bf16(T_MAX * LOCAL_H, HEAD_DIM),
                ),
                _ts(
                    "cos_il",
                    [T_MAX * LOCAL_H, ROPE_DIM],
                    torch.float32,
                    _ones(T_MAX * LOCAL_H, ROPE_DIM),
                ),
                _ts(
                    "sin_signed",
                    [T_MAX * LOCAL_H, ROPE_DIM],
                    torch.float32,
                    _zeros(T_MAX * LOCAL_H, ROPE_DIM),
                ),
                _sc("num_tokens", torch.int32, min(N_TOK * LOCAL_H, 32)),
            ],
        ),
        "rope_idx": (
            k_rope_idx,
            [
                _ts("values", [T_MAX, INDEX_DIM], torch.bfloat16, _randn_bf16(T_MAX, INDEX_DIM)),
                _ts("cos_il", [T_MAX, ROPE_DIM], torch.float32, _ones(T_MAX, ROPE_DIM)),
                _ts("sin_signed", [T_MAX, ROPE_DIM], torch.float32, _zeros(T_MAX, ROPE_DIM)),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
        "mx_wq_a": (k_mx_wq_a, _mx_specs(D, Q_LORA)),
        "mx_wkv": (k_mx_wkv, _mx_specs(D, HEAD_DIM)),
        "mx_idx": (k_mx_idx, _mx_specs(Q_LORA, INDEX_H * INDEX_DIM)),
        "mx_wo": (k_mx_wo, _mx_specs(LOCAL_O_WIDTH, D)),
        "mx_wq_b": (k_mx_wq_b, _mx_specs(Q_LORA, LOCAL_H * HEAD_DIM)),
        "bf16_linear": (
            k_bf16_linear,
            [
                _ts("act", [T_MAX, D], torch.bfloat16, _randn_bf16(T_MAX, D)),
                _ts("weight", [D, INDEX_H], torch.bfloat16, _randn_bf16(D, INDEX_H)),
                _ts("out", [T_MAX, INDEX_H], torch.float32),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
        "fp32_linear": (
            k_fp32_linear,
            [
                _ts("act", [T_MAX, D], torch.bfloat16, _randn_bf16(T_MAX, D)),
                _ts("weight", [D, HEAD_DIM], torch.float32, torch.randn),
                _ts("out", [T_MAX, HEAD_DIM], torch.float32),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
        "window": (
            k_window,
            [
                _ts("window_kv", [T_MAX, HEAD_DIM], torch.bfloat16, _randn_bf16(T_MAX, HEAD_DIM)),
                _ts("window_slots", [N_TOK], torch.int64, lambda: torch.arange(N_TOK, dtype=torch.int64)),
                _ts(
                    "window_cache",
                    [N_BLOCKS, 128, 1, HEAD_DIM],
                    torch.float8_e4m3fn,
                    lambda p=win_p: p,
                ),
                _ts(
                    "window_cache_scale",
                    [N_BLOCKS, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
                    win_s.dtype,
                    lambda s=win_s: s,
                ),
                _ts("window_bf16", [WS_ORI_ROWS, HEAD_DIM], torch.bfloat16),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
        "compressor": (
            k_compressor,
            [
                _ts("x", [N_TOK, D], torch.bfloat16, _randn_bf16(N_TOK, D)),
                _ts("position_ids", [N_TOK], torch.int32, lambda: torch.arange(N_TOK, dtype=torch.int32)),
                _ts("compressor_state_rows", [N_TOK], torch.int64, _zeros(N_TOK, dtype=torch.int64)),
                _ts(
                    "compressor_state",
                    [MAX_BATCH_PER_DP, STATE_HEADS, HEAD_DIM],
                    torch.float32,
                    _zeros(MAX_BATCH_PER_DP, STATE_HEADS, HEAD_DIM),
                ),
                _ts("compressor_wkv", [D, HEAD_DIM], torch.float32, torch.randn),
                _ts("compressor_wgate", [D, HEAD_DIM], torch.float32, torch.randn),
                _ts(
                    "compressor_norm_weight",
                    [HEAD_DIM],
                    torch.bfloat16,
                    _ones(HEAD_DIM, dtype=torch.bfloat16),
                ),
                _ts("latent", [T_MAX, HEAD_DIM], torch.bfloat16),
                _ts("publish_mask", [T_MAX], torch.int32),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
        "qkv": (
            k_qkv,
            [
                _ts("x", [N_TOK, D], torch.bfloat16, _randn_bf16(N_TOK, D)),
                _ts("wq_a", [D, Q_LORA], torch.float8_e4m3fn, lambda w=wq_a: w),
                _ts("wq_a_scale", [D // 32, Q_LORA], wq_a_s.dtype, lambda s=wq_a_s: s),
                _ts("q_norm_weight", [Q_LORA], torch.bfloat16, _ones(Q_LORA, dtype=torch.bfloat16)),
                _ts("wq_b", [Q_LORA, LOCAL_H * HEAD_DIM], torch.float8_e4m3fn, lambda w=wq_b: w),
                _ts(
                    "wq_b_scale",
                    [Q_LORA // 32, LOCAL_H * HEAD_DIM],
                    wq_b_s.dtype,
                    lambda s=wq_b_s: s,
                ),
                _ts("wkv", [D, HEAD_DIM], torch.float8_e4m3fn, lambda w=wkv: w),
                _ts("wkv_scale", [D // 32, HEAD_DIM], wkv_s.dtype, lambda s=wkv_s: s),
                _ts(
                    "kv_norm_weight",
                    [HEAD_DIM],
                    torch.bfloat16,
                    _ones(HEAD_DIM, dtype=torch.bfloat16),
                ),
                _ts("rope_cos", [N_TOK, ROPE_DIM // 2], torch.float32, _ones(N_TOK, ROPE_DIM // 2)),
                _ts("rope_sin", [N_TOK, ROPE_DIM // 2], torch.float32, _zeros(N_TOK, ROPE_DIM // 2)),
                _ts("qr", [T_MAX, Q_LORA], torch.bfloat16),
                _ts("q", [T_MAX, LOCAL_H, HEAD_DIM], torch.bfloat16),
                _ts("window_kv", [T_MAX, HEAD_DIM], torch.bfloat16),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
        "pub_cmp_idx": (
            k_pub_cmp_idx,
            [
                _ts("latent", [T_MAX, HEAD_DIM], torch.bfloat16, _randn_bf16(T_MAX, HEAD_DIM)),
                _ts("publish_mask", [T_MAX], torch.int32, lambda m=mask: m.clone()),
                _ts(
                    "compressed_rope_cos",
                    [N_TOK, ROPE_DIM // 2],
                    torch.float32,
                    _ones(N_TOK, ROPE_DIM // 2),
                ),
                _ts(
                    "compressed_rope_sin",
                    [N_TOK, ROPE_DIM // 2],
                    torch.float32,
                    _zeros(N_TOK, ROPE_DIM // 2),
                ),
                _ts(
                    "compressed_slots",
                    [N_TOK],
                    torch.int64,
                    lambda: torch.tensor([-1, 0][:N_TOK], dtype=torch.int64),
                ),
                _ts(
                    "compressed_cache",
                    [N_BLOCKS, 128, 1, CMP_PACKED],
                    torch.uint8,
                    lambda t=u8_cmp: t,
                ),
                _ts(
                    "compressed_cache_scale",
                    [N_BLOCKS, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
                    torch.bfloat16,
                    _zeros(
                        N_BLOCKS,
                        128,
                        1,
                        HEAD_DIM // COMPRESSED_CACHE_GROUP,
                        dtype=torch.bfloat16,
                    ),
                ),
                _ts("index_wk", [HEAD_DIM, INDEX_DIM], torch.bfloat16, _randn_bf16(HEAD_DIM, INDEX_DIM)),
                _ts(
                    "index_norm_weight",
                    [INDEX_DIM],
                    torch.bfloat16,
                    _ones(INDEX_DIM, dtype=torch.bfloat16),
                ),
                _ts(
                    "index_cache",
                    [N_BLOCKS, 128, 1, IDX_PACKED],
                    torch.uint8,
                    lambda t=u8_idx: t,
                ),
                _ts(
                    "index_cache_scale",
                    [N_BLOCKS, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP],
                    _e8m0_zeros(1).dtype,
                    lambda: _e8m0_zeros(N_BLOCKS, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP),
                ),
                _ts("mxfp4_pair_lut", [2, 256], torch.int16, lambda: pair_lut.clone()),
                _ts("cmp_bf16", [WS_CMP_ROWS, HEAD_DIM], torch.bfloat16),
                _ts("idx_bf16", [WS_IDX_ROWS, INDEX_DIM], torch.bfloat16),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
        "indexer": (
            k_indexer,
            [
                _ts("x", [N_TOK, D], torch.bfloat16, _randn_bf16(N_TOK, D)),
                _ts("qr", [T_MAX, Q_LORA], torch.bfloat16, _randn_bf16(T_MAX, Q_LORA)),
                _ts("request_ids", [N_TOK], torch.int32, _zeros(N_TOK, dtype=torch.int32)),
                _ts(
                    "compressed_lens",
                    [N_TOK],
                    torch.int32,
                    lambda: torch.tensor([0, 1][:N_TOK], dtype=torch.int32),
                ),
                _ts("index_block_table", [1, 4], torch.int32, _zeros(1, 4, dtype=torch.int32)),
                _ts("idx_bf16", [WS_IDX_ROWS, INDEX_DIM], torch.bfloat16),
                _ts(
                    "index_wq_b",
                    [Q_LORA, INDEX_H * INDEX_DIM],
                    torch.float8_e4m3fn,
                    lambda w=idx_w: w,
                ),
                _ts(
                    "index_wq_b_scale",
                    [Q_LORA // 32, INDEX_H * INDEX_DIM],
                    idx_s.dtype,
                    lambda s=idx_s: s,
                ),
                _ts("index_weights_proj", [D, INDEX_H], torch.bfloat16, _randn_bf16(D, INDEX_H)),
                _ts("rope_cos", [N_TOK, ROPE_DIM // 2], torch.float32, _ones(N_TOK, ROPE_DIM // 2)),
                _ts("rope_sin", [N_TOK, ROPE_DIM // 2], torch.float32, _zeros(N_TOK, ROPE_DIM // 2)),
                _ts("topk_indices", [N_TOK, INDEX_TOPK], torch.int32),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
        "attn": (
            k_attn,
            [
                _ts(
                    "q",
                    [T_MAX, LOCAL_H, HEAD_DIM],
                    torch.bfloat16,
                    _randn_bf16(T_MAX, LOCAL_H, HEAD_DIM),
                ),
                _ts("window_bf16", [WS_ORI_ROWS, HEAD_DIM], torch.bfloat16, _randn_bf16(WS_ORI_ROWS, HEAD_DIM)),
                _ts("window_indices", [N_TOK, 128], torch.int32, lambda: _window_indices()),
                _ts("cmp_bf16", [WS_CMP_ROWS, HEAD_DIM], torch.bfloat16, _randn_bf16(WS_CMP_ROWS, HEAD_DIM)),
                _ts("topk_indices", [N_TOK, INDEX_TOPK], torch.int32, lambda: _topk()),
                _ts("attn_sink", [LOCAL_H], torch.float32, torch.randn),
                _ts("attended", [T_MAX, LOCAL_H, HEAD_DIM], torch.bfloat16),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
        "oproj": (
            k_oproj,
            [
                _ts(
                    "attended",
                    [T_MAX, LOCAL_H, HEAD_DIM],
                    torch.bfloat16,
                    _randn_bf16(T_MAX, LOCAL_H, HEAD_DIM),
                ),
                _ts("rope_cos", [N_TOK, ROPE_DIM // 2], torch.float32, _ones(N_TOK, ROPE_DIM // 2)),
                _ts("rope_sin", [N_TOK, ROPE_DIM // 2], torch.float32, _zeros(N_TOK, ROPE_DIM // 2)),
                _ts(
                    "wo_a",
                    [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN],
                    torch.bfloat16,
                    _randn_bf16(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN),
                ),
                _ts("wo_b", [LOCAL_O_WIDTH, D], torch.float8_e4m3fn, lambda w=wo_b: w),
                _ts("wo_b_scale", [LOCAL_O_WIDTH // 32, D], wo_b_s.dtype, lambda s=wo_b_s: s),
                _ts("output_partial", [T_MAX, D], torch.float32),
                _sc("num_tokens", torch.int32, N_TOK),
            ],
        ),
    }


GROUPS = {
    "leaves": [
        "tiny_add",
        "dist_unused",
        "fp4_touch",
        "copy_d",
        "rms",
        "rope_prep",
        "rope_kv",
        "rope_hd",
        "rope_idx",
    ],
    "mx": [
        "mx_supported_tile",
        "mx_wq_a",
        "mx_wkv",
        "mx_idx",
        "mx_wo",
        "mx_wq_b",
        "bf16_linear",
        "fp32_linear",
    ],
    "stages": ["window", "compressor", "qkv", "pub_cmp_idx", "indexer", "attn", "oproj"],
    "suspect": ["fp4_touch", "pub_cmp_idx", "attn", "dist_unused"],
}


def _classify(text: str) -> str:
    low = text.lower()
    if "halresmap" in low or "rc=7" in low:
        return "FAIL_DEVICE"
    if "poisoned" in low or "sched_error_code=100" in low or "code -100" in low:
        return "FAIL_RUNTIME"
    if any(tok in text for tok in ("507018", "21008", "AICPU", "ACL")):
        return "FAIL_RUNTIME"
    if any(tok in low for tok in ("compile", "ptoas", "legaliz", "codegen", "missing type", "misplaced")):
        return "FAIL_COMPILE"
    return "FAIL_OTHER"


def _one_liner(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def run_one(name: str, fn, specs, platform: str, device: int) -> tuple[str, str]:
    print(f"\n======== KERNEL {name} device={device} ========", flush=True)
    try:
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=None,
            config=dict(platform=platform, device_id=device),
        )
    except Exception as exc:  # noqa: BLE001 — isolate one kernel crash from the rest
        detail = traceback.format_exc()
        print(detail, flush=True)
        return _classify(detail + str(exc)), _one_liner(str(exc) or detail)
    if result.passed:
        elapsed = f"{result.execution_time:.1f}s" if result.execution_time is not None else ""
        return "PASS", elapsed
    err = result.error or "unknown"
    return _classify(err), _one_liner(err)


def _run_group_isolated(names: list[str], platform: str, device: int, out_path: str) -> int:
    """One subprocess per kernel so a 507018 poison cannot cascade."""
    import subprocess
    from pathlib import Path

    script = str(Path(__file__).resolve())
    rows: list[tuple[str, str, str]] = []
    for name in names:
        one_out = f"/tmp/c2a_kernel_one_{device}_{name}.txt"
        cmd = [
            sys.executable,
            script,
            "--tp",
            "1",
            "-p",
            platform,
            "-d",
            str(device),
            "--kernel",
            name,
            "--out",
            one_out,
            "--in-process",
        ]
        print(f"\n[C2A-KERNEL] subprocess {name}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, check=False)
        status, detail = "FAIL_OTHER", f"exit={proc.returncode}"
        try:
            with open(one_out, encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.rstrip("\n").split("\t", 2)
                    if len(parts) >= 2 and parts[0] == name:
                        status = parts[1]
                        detail = parts[2] if len(parts) > 2 else ""
                        break
        except OSError:
            pass
        if proc.returncode != 0 and status == "PASS":
            status = "FAIL_OTHER"
        rows.append((name, status, detail))
        print(f"[KERNEL] {name}\t{status}\t{detail}", flush=True)

    print("\n======== SUMMARY ========", flush=True)
    for name, status, detail in rows:
        print(f"{name:16s} {status:14s} {detail}", flush=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"# device={device} LOCAL_H={LOCAL_H} isolated=1\n")
        for name, status, detail in rows:
            handle.write(f"{name}\t{status}\t{detail}\n")
    print(f"[C2A-KERNEL] wrote {out_path}", flush=True)
    return 0 if all(status == "PASS" for _, status, _ in rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent decode_c2a_full kernel onboard probes")
    parser.add_argument("-p", "--platform", type=str, default="a5", choices=["a5", "a2a3"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--group", type=str, default="leaves", choices=sorted(GROUPS))
    parser.add_argument("--kernel", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run kernels in this process (used by the subprocess isolator).",
    )
    args, _unknown = parser.parse_known_args()

    cases = _build_cases()
    names = [args.kernel] if args.kernel else GROUPS[args.group]
    unknown = [n for n in names if n not in cases]
    if unknown:
        raise SystemExit(f"unknown kernel(s): {unknown}; have {sorted(cases)}")

    out_path = args.out or f"/tmp/c2a_kernel_npu{args.device}.txt"

    # Multi-kernel runs isolate each case in a fresh process by default.
    if len(names) > 1 and not args.in_process:
        print(
            f"[C2A-KERNEL] group={args.group} device={args.device} tp={args.tp} "
            f"LOCAL_H={LOCAL_H} ACT_MAX={ACT_MAX} kernels={names} mode=isolated",
            flush=True,
        )
        return _run_group_isolated(names, args.platform, args.device, out_path)

    print(
        f"[C2A-KERNEL] group={args.group} device={args.device} tp={args.tp} "
        f"LOCAL_H={LOCAL_H} ACT_MAX={ACT_MAX} kernels={names} mode=in-process",
        flush=True,
    )
    rows: list[tuple[str, str, str]] = []
    for name in names:
        fn, specs = cases[name]
        status, detail = run_one(name, fn, specs, args.platform, args.device)
        rows.append((name, status, detail))
        print(f"[KERNEL] {name}\t{status}\t{detail}", flush=True)

    print("\n======== SUMMARY ========", flush=True)
    for name, status, detail in rows:
        print(f"{name:16s} {status:14s} {detail}", flush=True)

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"# group={args.group} device={args.device} LOCAL_H={LOCAL_H}\n")
        for name, status, detail in rows:
            handle.write(f"{name}\t{status}\t{detail}\n")
    print(f"[C2A-KERNEL] wrote {out_path}", flush=True)
    return 0 if all(status == "PASS" for _, status, _ in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
