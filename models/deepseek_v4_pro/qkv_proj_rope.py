# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Q/KV LoRA + RoPE (dynamic shape): projects token-major
attention-normalized inputs for both decode and prefill attention paths."""


import pypto.language as pl

from config import ACTIVE as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S


# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps
MAX_SEQ_LEN = M.max_position_embeddings
T_MAX = max(DECODE_BATCH * DECODE_SEQ, PREFILL_BATCH * PREFILL_SEQ)
MX_GROUP = 32

# tiling
Q_PROJ_TILE = 128
QPROJ_MM_N_TILE = 512
Q_LORA_TILE = 512
KV_TILE = 64
QUANT_TILE = 512
T_TILE = 8
MATMUL_T_TILE = 16
QR_M_TILE = MATMUL_T_TILE
QR_N_TILE = 128
QR_K_TILE = 256
QR_SPLIT_TILE = 2
QR_K_SPLIT_TILE = D // QR_SPLIT_TILE
KV_M_TILE = MATMUL_T_TILE
KV_N_TILE = 128
KV_K_TILE = 128
KV_SPLIT_TILE = 4
KV_K_SPLIT_TILE = D // KV_SPLIT_TILE
QPROJ_M_TILE = MATMUL_T_TILE
KV_RMS_T_TILE = 8
Q_ROPE_T_TILE = 8
Q_ROPE_H_TILE = 4


def _even_pipeline_trip(name: str, trip: int) -> None:
    """Require even split-K pipeline trips for A5 accumulator-buffer selection."""
    assert trip % 2 == 0, (
        f"{name} pipeline trip count must be even, got {trip}; an odd count makes "
        "codegen emit an unsupported acc->acc pto.tmov. Retune the K tile or split-K factor."
    )


_even_pipeline_trip("qr_proj", QR_K_SPLIT_TILE // QR_K_TILE)
_even_pipeline_trip("kv_proj", KV_K_SPLIT_TILE // KV_K_TILE)
assert QPROJ_MM_N_TILE * QPROJ_M_TILE * 4 <= 128 * 1024  # L0C accumulator capacity


@pl.jit.inline
def materialize_rope_rows(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rope_cos_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(position_ids, 0)
    for rope_t0 in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint="qkv_rope_rows"):
        t0 = rope_t0 * KV_RMS_T_TILE
        for rope_dt in pl.range(KV_RMS_T_TILE):
            rope_t = t0 + rope_dt
            if rope_t < num_tokens:
                rope_pos_i32 = pl.read(position_ids, [rope_t])
                rope_pos = pl.cast(rope_pos_i32, pl.INDEX)
                rope_cos_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_cos[rope_pos : rope_pos + 1, 0:ROPE_DIM]
                rope_sin_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_sin[rope_pos : rope_pos + 1, 0:ROPE_DIM]

@pl.jit.inline
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_MAX, Q_LORA], pl.FP8E4M3FN],
    qr_scale: pl.Tensor[[1, T_MAX * (Q_LORA // MX_GROUP)], pl.FP8E8M0],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    t_dim = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [t_dim, D])
    rope_cos_view = pl.reshape(rope_cos, [t_dim, ROPE_DIM])
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])
    q_rope_cos_il = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    for qrp_idx in pl.spmd(t_dim // Q_ROPE_T_TILE, name_hint="q_rope_prepare"):
        qrp_t0 = qrp_idx * Q_ROPE_T_TILE
        qrp_cos = pl.cast(rope_cos_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, 0 : ROPE_DIM // 2], target_type=pl.FP32)
        qrp_sin = pl.cast(rope_sin_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, 0 : ROPE_DIM // 2], target_type=pl.FP32)
        qrp_cos_il = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
        qrp_cos_il = pl.tensor.scatter(qrp_cos, mask_pattern=pl.tile.MaskPattern.P0101, dst=qrp_cos_il)
        qrp_cos_il = pl.tensor.scatter(qrp_cos, mask_pattern=pl.tile.MaskPattern.P1010, dst=qrp_cos_il)
        q_rope_cos_il[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_cos_il
        qrp_sin_neg = pl.neg(qrp_sin)
        qrp_sin_signed = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
        qrp_sin_signed = pl.tensor.scatter(qrp_sin_neg, mask_pattern=pl.tile.MaskPattern.P0101, dst=qrp_sin_signed)
        qrp_sin_signed = pl.tensor.scatter(qrp_sin, mask_pattern=pl.tile.MaskPattern.P1010, dst=qrp_sin_signed)
        q_rope_sin_signed[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_sin_signed

    t_matmul = pl.max(t_dim, MATMUL_T_TILE)
    x_pad = pl.create_tensor([T_MAX, D], dtype=pl.BF16)
    for x_seed_idx in pl.spmd((T_MAX // MATMUL_T_TILE) * (D // QR_K_TILE), name_hint="qkv_x_pad_seed"):
        x_seed_t0 = (x_seed_idx // (D // QR_K_TILE)) * MATMUL_T_TILE
        x_seed_k0 = (x_seed_idx % (D // QR_K_TILE)) * QR_K_TILE
        x_pad[x_seed_t0 : x_seed_t0 + MATMUL_T_TILE, x_seed_k0 : x_seed_k0 + QR_K_TILE] = pl.full(
            [MATMUL_T_TILE, QR_K_TILE], dtype=pl.BF16, value=0.0
        )
    for x_copy_idx in pl.spmd(t_dim * (D // QR_K_TILE), name_hint="qkv_x_pad_copy"):
        x_copy_t0 = x_copy_idx // (D // QR_K_TILE)
        x_copy_k0 = (x_copy_idx % (D // QR_K_TILE)) * QR_K_TILE
        x_pad[x_copy_t0 : x_copy_t0 + 1, x_copy_k0 : x_copy_k0 + QR_K_TILE] = x_view[
            x_copy_t0 : x_copy_t0 + 1, x_copy_k0 : x_copy_k0 + QR_K_TILE
        ]

    x_mx = pl.create_tensor([T_MAX, D], dtype=pl.FP8E4M3FN)
    x_scale_backing = pl.create_tensor([1, T_MAX * (D // MX_GROUP)], dtype=pl.FP8E8M0)
    for x_quant_idx in pl.spmd((t_matmul // MATMUL_T_TILE) * (D // QR_K_TILE), name_hint="qkv_x_mx_quant"):
        x_quant_t0 = (x_quant_idx // (D // QR_K_TILE)) * MATMUL_T_TILE
        x_quant_k0 = (x_quant_idx % (D // QR_K_TILE)) * QR_K_TILE
        x_quant_tile = pl.load(x_pad, [x_quant_t0, x_quant_k0], [MATMUL_T_TILE, QR_K_TILE])
        x_quant_fp32 = pl.cast(x_quant_tile, target_type=pl.FP32, mode="none")
        x_quant_mx, x_quant_scale = pl.quant_mx(x_quant_fp32, group_axis=1)
        x_mx = pl.store(x_quant_mx, [x_quant_t0, x_quant_k0], x_mx)
        x_scale_block = x_quant_idx * MATMUL_T_TILE * (QR_K_TILE // MX_GROUP)
        x_scale_flat = pl.reshape(x_quant_scale, [1, MATMUL_T_TILE * (QR_K_TILE // MX_GROUP)])
        x_scale_backing = pl.store(x_scale_flat, [0, x_scale_block], x_scale_backing)

    x_scale_mx = pl.tensor.view(x_scale_backing, [T_MAX, D // MX_GROUP], layout=pl.MX_A_ZZ)

    qr_fp32 = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.FP32)
    for qr_mm_idx in pl.spmd(Q_LORA // QR_N_TILE, name_hint="qr_proj_mx"):
        qr_n0 = qr_mm_idx * QR_N_TILE
        for qr_tc in pl.range(t_matmul // QR_M_TILE):
            qr_t0 = qr_tc * QR_M_TILE
            qr_lhs0 = pl.load(x_mx, [qr_t0, 0], [QR_M_TILE, QR_K_TILE])
            qr_lhs_scale0 = pl.load(x_scale_mx, [qr_t0, 0], [QR_M_TILE, QR_K_TILE // MX_GROUP])
            qr_rhs0 = pl.load(wq_a, [0, qr_n0], [QR_K_TILE, QR_N_TILE])
            qr_rhs_scale0 = pl.load(wq_a_scale, [0, qr_n0], [QR_K_TILE // MX_GROUP, QR_N_TILE])
            qr_acc = pl.matmul_mx(qr_lhs0, qr_lhs_scale0, qr_rhs0, qr_rhs_scale0)
            for qr_k0 in pl.pipeline(QR_K_TILE, D, QR_K_TILE, stage=2):
                qr_ks = qr_k0 // MX_GROUP
                qr_lhs = pl.load(x_mx, [qr_t0, qr_k0], [QR_M_TILE, QR_K_TILE])
                qr_lhs_scale = pl.load(x_scale_mx, [qr_t0, qr_ks], [QR_M_TILE, QR_K_TILE // MX_GROUP])
                qr_rhs = pl.load(wq_a, [qr_k0, qr_n0], [QR_K_TILE, QR_N_TILE])
                qr_rhs_scale = pl.load(wq_a_scale, [qr_ks, qr_n0], [QR_K_TILE // MX_GROUP, QR_N_TILE])
                qr_acc = pl.matmul_mx_acc(qr_acc, qr_lhs, qr_lhs_scale, qr_rhs, qr_rhs_scale)
            qr_fp32 = pl.store(qr_acc, [qr_t0, qr_n0], qr_fp32)

    qr_norm = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.FP32)
    for qr_norm_seed_idx in pl.spmd((T_MAX // T_TILE) * (Q_LORA // QUANT_TILE), name_hint="qr_norm_seed"):
        qr_norm_seed_t0 = (qr_norm_seed_idx // (Q_LORA // QUANT_TILE)) * T_TILE
        qr_norm_seed_k0 = (qr_norm_seed_idx % (Q_LORA // QUANT_TILE)) * QUANT_TILE
        qr_norm[qr_norm_seed_t0 : qr_norm_seed_t0 + T_TILE, qr_norm_seed_k0 : qr_norm_seed_k0 + QUANT_TILE] = pl.full(
            [T_TILE, QUANT_TILE], dtype=pl.FP32, value=0.0
        )
    for qr_norm_idx in pl.spmd(t_dim // T_TILE, name_hint="qr_rms_norm"):
        qr_norm_t0 = qr_norm_idx * T_TILE
        qr_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for qr_rms_col0 in pl.pipeline(0, Q_LORA, Q_LORA_TILE, stage=2):
            qr_rms_chunk = qr_fp32[qr_norm_t0 : qr_norm_t0 + T_TILE, qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE]
            qr_sq = pl.mul(qr_rms_chunk, qr_rms_chunk)
            qr_sq_row = pl.reshape(pl.row_sum(qr_sq), [1, T_TILE])
            qr_sq_sum = pl.add(qr_sq_sum, qr_sq_row)
        qr_sq_mean = pl.mul(qr_sq_sum, 1.0 / Q_LORA)
        qr_inv_rms_row = pl.rsqrt(pl.add(qr_sq_mean, EPS), high_precision=True)
        qr_inv_rms = pl.reshape(qr_inv_rms_row, [T_TILE, 1])
        for qr_norm_col0 in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
            qr_chunk = qr_fp32[qr_norm_t0 : qr_norm_t0 + T_TILE, qr_norm_col0 : qr_norm_col0 + QUANT_TILE]
            gamma_chunk = pl.cast(gamma_cq[qr_norm_col0 : qr_norm_col0 + QUANT_TILE], target_type=pl.FP32)
            gamma_row = pl.reshape(gamma_chunk, [1, QUANT_TILE])
            qr_rms = pl.row_expand_mul(qr_chunk, qr_inv_rms)
            qr_normed = pl.col_expand_mul(qr_rms, gamma_row)
            qr_norm[qr_norm_t0 : qr_norm_t0 + T_TILE, qr_norm_col0 : qr_norm_col0 + QUANT_TILE] = qr_normed

    qr_mx = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.FP8E4M3FN)
    qr_scale_backing = pl.create_tensor([1, T_MAX * (Q_LORA // MX_GROUP)], dtype=pl.FP8E8M0)
    for qr_quant_idx in pl.spmd((T_MAX // MATMUL_T_TILE) * (Q_LORA // QUANT_TILE), name_hint="qr_mx_quant"):
        qr_quant_t0 = (qr_quant_idx // (Q_LORA // QUANT_TILE)) * MATMUL_T_TILE
        qr_quant_k0 = (qr_quant_idx % (Q_LORA // QUANT_TILE)) * QUANT_TILE
        qr_quant_tile = pl.load(qr_norm, [qr_quant_t0, qr_quant_k0], [MATMUL_T_TILE, QUANT_TILE])
        qr_quant_mx, qr_quant_scale = pl.quant_mx(qr_quant_tile, group_axis=1)
        qr_mx = pl.store(qr_quant_mx, [qr_quant_t0, qr_quant_k0], qr_mx)
        qr_scale_block = qr_quant_idx * MATMUL_T_TILE * (QUANT_TILE // MX_GROUP)
        qr_scale_flat = pl.reshape(qr_quant_scale, [1, MATMUL_T_TILE * (QUANT_TILE // MX_GROUP)])
        qr_scale_backing = pl.store(qr_scale_flat, [0, qr_scale_block], qr_scale_backing)
        qr = pl.store(qr_quant_mx, [qr_quant_t0, qr_quant_k0], qr)
        qr_scale = pl.store(qr_scale_flat, [0, qr_scale_block], qr_scale)

    qr_scale_mx = pl.tensor.view(qr_scale_backing, [T_MAX, Q_LORA // MX_GROUP], layout=pl.MX_A_ZZ)

    # RoPE: out[j] = inv_rms * (x[j] * cos[j] + x[j^1] * sign[j] * sin[j]).
    q_proj_fp32 = pl.create_tensor([T_MAX, H * HEAD_DIM], dtype=pl.FP32)
    for qproj_idx in pl.spmd((H * HEAD_DIM) // QPROJ_MM_N_TILE, name_hint="qproj_mx"):
        w_col0 = qproj_idx * QPROJ_MM_N_TILE
        for qproj_tc in pl.range(t_matmul // QPROJ_M_TILE):
            t0 = qproj_tc * QPROJ_M_TILE
            q_lhs0 = pl.load(qr_mx, [t0, 0], [QPROJ_M_TILE, Q_PROJ_TILE])
            q_lhs_scale0 = pl.load(qr_scale_mx, [t0, 0], [QPROJ_M_TILE, Q_PROJ_TILE // MX_GROUP])
            q_rhs0 = pl.load(wq_b, [0, w_col0], [Q_PROJ_TILE, QPROJ_MM_N_TILE])
            q_rhs_scale0 = pl.load(wq_b_scale, [0, w_col0], [Q_PROJ_TILE // MX_GROUP, QPROJ_MM_N_TILE])
            col_acc = pl.matmul_mx(q_lhs0, q_lhs_scale0, q_rhs0, q_rhs_scale0)
            for q_k0 in pl.pipeline(Q_PROJ_TILE, Q_LORA, Q_PROJ_TILE, stage=2):
                q_ks = q_k0 // MX_GROUP
                q_lhs = pl.load(qr_mx, [t0, q_k0], [QPROJ_M_TILE, Q_PROJ_TILE])
                q_lhs_scale = pl.load(qr_scale_mx, [t0, q_ks], [QPROJ_M_TILE, Q_PROJ_TILE // MX_GROUP])
                q_rhs = pl.load(wq_b, [q_k0, w_col0], [Q_PROJ_TILE, QPROJ_MM_N_TILE])
                q_rhs_scale = pl.load(wq_b_scale, [q_ks, w_col0], [Q_PROJ_TILE // MX_GROUP, QPROJ_MM_N_TILE])
                col_acc = pl.matmul_mx_acc(col_acc, q_lhs, q_lhs_scale, q_rhs, q_rhs_scale)
            q_proj_fp32 = pl.store(col_acc, [t0, w_col0], q_proj_fp32)

    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
    for q_ep_idx in pl.spmd((t_dim // Q_ROPE_T_TILE) * H, name_hint="qproj_rms_rope"):
        fq_tg = (q_ep_idx // H) * Q_ROPE_T_TILE
        h0 = (q_ep_idx % H) * HEAD_DIM
        q_cos_il = q_rope_cos_il[fq_tg : fq_tg + Q_ROPE_T_TILE, :]
        q_sin_signed = q_rope_sin_signed[fq_tg : fq_tg + Q_ROPE_T_TILE, :]
        q_head_dq = q_proj_fp32[fq_tg : fq_tg + Q_ROPE_T_TILE, h0 : h0 + HEAD_DIM]
        q_head_sq = pl.mul(q_head_dq, q_head_dq)
        q_head_sq_row = pl.row_sum(q_head_sq)
        q_head_sq_sum = pl.reshape(q_head_sq_row, [1, Q_ROPE_T_TILE])
        q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
        q_head_var = pl.add(q_head_sq_mean, EPS)
        q_head_inv_rms = pl.rsqrt(q_head_var, high_precision=True)
        q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])

        q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
        q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
        q_flat[fq_tg : fq_tg + Q_ROPE_T_TILE, h0 : h0 + NOPE_DIM] = q_nope_bf16

        q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
        q_rope_chunk = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
        q_rope_col0 = h0 + NOPE_DIM
        q_rope_even = pl.gather(q_rope_chunk, mask_pattern=pl.tile.MaskPattern.P0101)
        q_rope_odd = pl.gather(q_rope_chunk, mask_pattern=pl.tile.MaskPattern.P1010)
        q_rope_swapped = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
        q_rope_swapped = pl.tensor.scatter(q_rope_odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=q_rope_swapped)
        q_rope_swapped = pl.tensor.scatter(q_rope_even, mask_pattern=pl.tile.MaskPattern.P1010, dst=q_rope_swapped)
        q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
        q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
        q_rope_rot = pl.add(q_rope_base, q_rope_delta)
        q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
        q_flat[fq_tg : fq_tg + Q_ROPE_T_TILE, q_rope_col0 : q_rope_col0 + ROPE_DIM] = q_rope_bf16

    kv_fp32 = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.FP32)
    with pl.spmd(HEAD_DIM // KV_N_TILE, name_hint="kv_proj_mx", deps=[late_dep]) as _kv_tid:
        kv_idx = pl.tile.get_block_idx()
        kv_col0 = kv_idx * KV_N_TILE
        for kv_tc in pl.range(t_matmul // KV_M_TILE):
            kv_t0 = kv_tc * KV_M_TILE
            kv_lhs0 = pl.load(x_mx, [kv_t0, 0], [KV_M_TILE, KV_K_TILE])
            kv_lhs_scale0 = pl.load(x_scale_mx, [kv_t0, 0], [KV_M_TILE, KV_K_TILE // MX_GROUP])
            kv_rhs0 = pl.load(wkv, [0, kv_col0], [KV_K_TILE, KV_N_TILE])
            kv_rhs_scale0 = pl.load(wkv_scale, [0, kv_col0], [KV_K_TILE // MX_GROUP, KV_N_TILE])
            kv_acc = pl.matmul_mx(kv_lhs0, kv_lhs_scale0, kv_rhs0, kv_rhs_scale0)
            for kv_k0 in pl.pipeline(KV_K_TILE, D, KV_K_TILE, stage=2):
                kv_ks = kv_k0 // MX_GROUP
                kv_lhs = pl.load(x_mx, [kv_t0, kv_k0], [KV_M_TILE, KV_K_TILE])
                kv_lhs_scale = pl.load(x_scale_mx, [kv_t0, kv_ks], [KV_M_TILE, KV_K_TILE // MX_GROUP])
                kv_rhs = pl.load(wkv, [kv_k0, kv_col0], [KV_K_TILE, KV_N_TILE])
                kv_rhs_scale = pl.load(wkv_scale, [kv_ks, kv_col0], [KV_K_TILE // MX_GROUP, KV_N_TILE])
                kv_acc = pl.matmul_mx_acc(kv_acc, kv_lhs, kv_lhs_scale, kv_rhs, kv_rhs_scale)
            kv_fp32 = pl.store(kv_acc, [kv_t0, kv_col0], kv_fp32)

    kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])
    for tg_idx in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint="kv_rms_norm_rope"):
        tg = tg_idx * KV_RMS_T_TILE
        kv_sq_sum = pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kv_sq_col0 in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
            kv_sq = pl.mul(kv_chunk, kv_chunk)
            kv_sq_row = pl.row_sum(kv_sq)
            kv_sq_partial = pl.reshape(kv_sq_row, [1, KV_RMS_T_TILE])
            kv_sq_sum = pl.add(kv_sq_sum, kv_sq_partial)
        kv_sq_mean = pl.mul(kv_sq_sum, 1.0 / HEAD_DIM)
        kv_rms_arg = pl.add(kv_sq_mean, EPS)
        kv_inv_rms = pl.rsqrt(kv_rms_arg, high_precision=True)
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])

        for n0 in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
            gamma_kv_cast = pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32)
            gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
            kv_rms = pl.row_expand_mul(kv_chunk, kv_inv_rms_t)
            kv_normed = pl.col_expand_mul(kv_rms, gamma_kv_chunk)
            kv_normed_bf16 = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")
            kv_view[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = kv_normed_bf16

        # RoPE: out[j] = n[j] * cos_il[j] + n[j^1] * sign[j] * sin_il[j].
        gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
        gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
        kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
        kv_rope_rms = pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t)
        kv_rope_norm_chunk = pl.col_expand_mul(kv_rope_rms, gamma_rope)
        for kv_rope_row in pl.range(KV_RMS_T_TILE):
            kv_rope_t = tg + kv_rope_row
            kv_cos = pl.cast(rope_cos_view[kv_rope_t : kv_rope_t + 1, 0:ROPE_HALF], target_type=pl.FP32)
            kv_sin = pl.cast(rope_sin_view[kv_rope_t : kv_rope_t + 1, 0:ROPE_HALF], target_type=pl.FP32)
            kv_rope_norm_row = kv_rope_norm_chunk[kv_rope_row : kv_rope_row + 1, :]
            kv_rope_even = pl.gather(kv_rope_norm_row, mask_pattern=pl.tile.MaskPattern.P0101)
            kv_rope_odd = pl.gather(kv_rope_norm_row, mask_pattern=pl.tile.MaskPattern.P1010)
            kv_even_base = pl.mul(kv_rope_even, kv_cos)
            kv_odd_neg = pl.neg(kv_rope_odd)
            kv_even_delta = pl.mul(kv_odd_neg, kv_sin)
            kv_even_rot = pl.add(kv_even_base, kv_even_delta)
            kv_odd_base = pl.mul(kv_rope_odd, kv_cos)
            kv_odd_delta = pl.mul(kv_rope_even, kv_sin)
            kv_odd_rot = pl.add(kv_odd_base, kv_odd_delta)
            kv_rope_rot = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
            kv_rope_rot = pl.tensor.scatter(kv_even_rot, mask_pattern=pl.tile.MaskPattern.P0101, dst=kv_rope_rot)
            kv_rope_rot = pl.tensor.scatter(kv_odd_rot, mask_pattern=pl.tile.MaskPattern.P1010, dst=kv_rope_rot)
            kv_rope_i16 = pl.cast(kv_rope_rot, target_type=pl.BF16, mode="rint")
            kv_view[kv_rope_t : kv_rope_t + 1, NOPE_DIM : NOPE_DIM + ROPE_DIM] = kv_rope_i16

    return q


@pl.jit
def qkv_proj_rope_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Out[pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16]],
    kv: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16]],
    qr: pl.Out[pl.Tensor[[T_MAX, Q_LORA], pl.FP8E4M3FN]],
    qr_scale: pl.Out[pl.Tensor[[1, T_MAX * (Q_LORA // MX_GROUP)], pl.FP8E8M0]],
):
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    q.bind_dynamic(0, T_DYN)
    kv.bind_dynamic(0, T_DYN)

    late_dep = pl.system.task_dummy(deps=[])
    qkv_proj_rope(
        x,
        wq_a, wq_a_scale, wq_b, wq_b_scale, wkv, wkv_scale,
        rope_cos, rope_sin,
        gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale,
        late_dep,
    )
    return q


_A5_FP32_VECTOR_LANES = 64
_A5_CUBE_ACC_K = 16
_A5_CUBE_N_TILE = 128
_A5_CUBE_K_GROUP_TILE = 64


def _golden_a5_trowsum_fp32(values):
    """Mirror A5 TROWSUM's FP32 reduction order along the last dimension."""
    import torch

    if values.dtype != torch.float32:
        raise ValueError(f"A5 FP32 TROWSUM golden requires float32, got {values.dtype}")
    if values.shape[-1] % _A5_FP32_VECTOR_LANES != 0:
        raise ValueError(
            f"A5 FP32 TROWSUM width must be divisible by {_A5_FP32_VECTOR_LANES}, "
            f"got {values.shape[-1]}"
        )

    groups = values.reshape(*values.shape[:-1], -1, _A5_FP32_VECTOR_LANES)
    while groups.shape[-1] > 1:
        pairs = groups.reshape(*groups.shape[:-1], -1, 2)
        groups = pairs[..., 0] + pairs[..., 1]

    group_sums = groups[..., 0]
    total = torch.zeros_like(group_sums[..., :1])
    for group in range(group_sums.shape[-1]):
        total += group_sums[..., group:group + 1]
    return total


def _golden_a5_high_precision_rsqrt(value):
    """Match A5 high-precision FP32 rsqrt without host FP32 double rounding."""
    import torch

    return torch.rsqrt(value.to(torch.float64)).to(torch.float32)


def _golden_a5_cube_bf16_matmul(lhs, rhs):
    """Return ``lhs @ rhs`` in A5 BF16 Cube's K16 MAD accumulation order."""
    import torch

    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError("A5 Cube golden requires two rank-2 matrices")
    m_dim, k_dim = lhs.shape
    rhs_k_dim, n_dim = rhs.shape
    if k_dim != rhs_k_dim or k_dim % _A5_CUBE_ACC_K != 0:
        raise ValueError(
            f"A5 Cube golden requires equal K divisible by {_A5_CUBE_ACC_K}, "
            f"got {k_dim} and {rhs_k_dim}"
        )

    k_groups = k_dim // _A5_CUBE_ACC_K
    x_groups = lhs.to(torch.bfloat16).reshape(m_dim, k_groups, _A5_CUBE_ACC_K).double()
    w_groups = rhs.to(torch.bfloat16).T.contiguous().reshape(n_dim, k_groups, _A5_CUBE_ACC_K).double()
    out = torch.zeros(m_dim, n_dim, dtype=torch.float32, device=lhs.device)
    for n0 in range(0, n_dim, _A5_CUBE_N_TILE):
        n1 = min(n0 + _A5_CUBE_N_TILE, n_dim)
        acc = torch.zeros(m_dim, n1 - n0, dtype=torch.float32, device=lhs.device)
        for group0 in range(0, k_groups, _A5_CUBE_K_GROUP_TILE):
            group1 = min(group0 + _A5_CUBE_K_GROUP_TILE, k_groups)
            group_dots = torch.einsum(
                "mgk,ngk->mng",
                x_groups[:, group0:group1],
                w_groups[n0:n1, group0:group1],
            )
            for group in range(group1 - group0):
                acc = (acc.double() + group_dots[:, :, group]).float()
        out[:, n0:n1] = acc
    return out


def _golden_a5_split_k_bf16_matmul(lhs, rhs, *, splits, k_per_split):
    """Mirror per-split continuous K16 Cube MADs, then ascending split reduction."""
    if lhs.shape[-1] != splits * k_per_split:
        raise ValueError(
            f"split-K golden expected K={splits * k_per_split}, got {lhs.shape[-1]}"
        )
    total = None
    for split in range(splits):
        k0 = split * k_per_split
        partial = _golden_a5_cube_bf16_matmul(
            lhs[:, k0:k0 + k_per_split],
            rhs[k0:k0 + k_per_split],
        )
        total = partial if total is None else total + partial
    return total


def _golden_a5_chunked_rms_inv(values, *, chunk_size, eps):
    """A5 FP32 RMS: TROWSUM each chunk, add chunks ascending, then HP rsqrt."""
    import torch

    if values.dtype != torch.float32:
        raise ValueError(f"A5 RMS golden requires float32, got {values.dtype}")
    width = values.shape[-1]
    if width % chunk_size != 0:
        raise ValueError(f"A5 RMS width {width} must be divisible by chunk {chunk_size}")
    sq_sum = torch.zeros(
        *values.shape[:-1], 1, dtype=torch.float32, device=values.device
    )
    for k0 in range(0, width, chunk_size):
        chunk = values[..., k0:k0 + chunk_size]
        sq_sum += _golden_a5_trowsum_fp32(chunk * chunk)
    rms_arg = sq_sum * (1.0 / width) + eps
    return _golden_a5_high_precision_rsqrt(rms_arg)


def _golden_qr_rms_norm_quant(qr_fp32, gamma_cq):
    """Apply QR RMSNorm and group-32 MXFP8 quantization."""
    from mx_utils import host_quant_mxfp8

    gamma_fp32 = gamma_cq.float()
    qr_inv_rms = _golden_a5_chunked_rms_inv(qr_fp32, chunk_size=Q_LORA_TILE, eps=EPS)
    qr_normed = (qr_fp32 * qr_inv_rms) * gamma_fp32
    return host_quant_mxfp8(qr_normed, return_e8m0=True)


def _golden_a5_q_head_rms_norm(q_full):
    """Mirror the fused Q dequant kernel's HEAD_DIM TROWSUM and HP rsqrt."""
    q_inv_rms = _golden_a5_chunked_rms_inv(q_full, chunk_size=HEAD_DIM, eps=EPS)
    return q_full * q_inv_rms


def golden_qkv_proj_rope(tensors):
    """Torch reference: Q/KV LoRA + RoPE for an already attention-normalized input."""
    import torch

    full_t_dim = tensors["x"].shape[0]
    active_t_dim = full_t_dim
    if "num_tokens" in tensors:
        active_t_dim = max(0, min(int(tensors["num_tokens"]), full_t_dim))
        tensors["q"].zero_()
        tensors["kv"].zero_()
        tensors["qr"].zero_()
        tensors["qr_scale"].zero_()
        if active_t_dim == 0:
            return

    x = tensors["x"][:active_t_dim].float()
    from mx_utils import decode_e8m0_codes, host_quant_mxfp8, matmul_mx_golden

    wq_a = tensors["wq_a"]
    wq_a_scale = decode_e8m0_codes(tensors["wq_a_scale"], side="b")
    wq_b = tensors["wq_b"]
    wq_b_scale = decode_e8m0_codes(tensors["wq_b_scale"], side="b")
    wkv = tensors["wkv"]
    wkv_scale = decode_e8m0_codes(tensors["wkv_scale"], side="b")
    rope_cos = tensors["rope_cos"][:active_t_dim].float()
    rope_sin = tensors["rope_sin"][:active_t_dim].float()
    gamma_cq = tensors["gamma_cq"].float()
    gamma_ckv = tensors["gamma_ckv"].float()

    def rms_norm(x, gamma, eps=EPS):
        inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return x * inv * gamma

    def apply_rope(x_rope, cos, sin):
        # x_rope: [T, ..., ROPE_DIM] with interleaved even/odd rotary pairs.
        x_pair = x_rope.unflatten(-1, (-1, 2))
        x_even, x_odd = x_pair[..., 0], x_pair[..., 1]
        cos_v = cos[..., :ROPE_HALF]
        sin_v = sin[..., :ROPE_HALF]
        while cos_v.ndim < x_even.ndim:
            cos_v = cos_v.unsqueeze(-2)
            sin_v = sin_v.unsqueeze(-2)
        y_even = (x_even * cos_v - x_odd * sin_v).to(torch.bfloat16)
        y_odd = (x_even * sin_v + x_odd * cos_v).to(torch.bfloat16)
        return torch.stack([y_even, y_odd], dim=-1).flatten(-2)

    t_dim = active_t_dim
    token_x = x.view(t_dim, D)

    x_fp8, x_scale = host_quant_mxfp8(token_x, return_e8m0=True)
    qr_fp32 = matmul_mx_golden(x_fp8, x_scale, wq_a, wq_a_scale)
    qr_fp8, qr_scale = _golden_qr_rms_norm_quant(qr_fp32, gamma_cq)
    q_full = matmul_mx_golden(qr_fp8, qr_scale, wq_b, wq_b_scale).view(t_dim, H, HEAD_DIM)
    q_full = _golden_a5_q_head_rms_norm(q_full)  # per-head RMSNorm (no gamma)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos, rope_sin)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    kv_proj = matmul_mx_golden(x_fp8, x_scale, wkv, wkv_scale)
    kv_full = rms_norm(kv_proj, gamma_ckv)
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)               # add a pseudo head dim
    kv_rope = apply_rope(kv_rope_in, rope_cos, rope_sin).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    tensors["q"][:active_t_dim] = q_out.to(torch.bfloat16)
    tensors["kv"][:active_t_dim] = kv_out.to(torch.bfloat16)
    tensors["qr"].zero_()
    tensors["qr"][:active_t_dim] = qr_fp8
    qr_scale_codes = torch.zeros(T_MAX, Q_LORA // MX_GROUP, dtype=torch.uint8)
    qr_scale_codes[:active_t_dim] = qr_scale.contiguous().view(torch.uint8)
    from mx_utils import pack_a_scale
    tensors["qr_scale"].view(torch.uint8).copy_(pack_a_scale(qr_scale_codes).reshape(1, -1))


def _reference_q_from_quantized_qr(
    qr,
    qr_scale,
    wq_b,
    wq_b_scale,
    rope_cos,
    rope_sin,
):
    """Recompute the Q path downstream of the emitted INT8 QR boundary."""
    import torch

    from mx_utils import decode_e8m0_codes, matmul_mx_golden, unpack_a_scale

    t_dim = rope_cos.shape[0]
    qr = qr[:t_dim]
    qr_scale_codes = unpack_a_scale(qr_scale.contiguous().view(torch.uint8).reshape(T_MAX, -1))
    qr_scale = qr_scale_codes[:t_dim].view(torch.float8_e8m0fnu)
    weight_scale = decode_e8m0_codes(wq_b_scale, side="b")
    q_full = matmul_mx_golden(qr, qr_scale, wq_b, weight_scale).view(t_dim, H, HEAD_DIM)
    q_full = _golden_a5_q_head_rms_norm(q_full)

    q_pair = q_full[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    q_even, q_odd = q_pair[..., 0], q_pair[..., 1]
    cos = rope_cos.float()[..., :ROPE_HALF].unsqueeze(-2)
    sin = rope_sin.float()[..., :ROPE_HALF].unsqueeze(-2)
    y_even = (q_even * cos - q_odd * sin).to(torch.bfloat16)
    y_odd = (q_even * sin + q_odd * cos).to(torch.bfloat16)
    q_rope = torch.stack([y_even, y_odd], dim=-1).flatten(-2)
    return torch.cat([q_full[..., :NOPE_DIM], q_rope], dim=-1).to(torch.bfloat16)


def quantized_qr_compare(
    *,
    max_code_step=1,
    max_changed_ratio=0.005,
    max_changed_per_row_ratio=0.005,
    max_show=10,
):
    """Bound both the magnitude and population of QR quantization-boundary changes."""
    import torch

    if max_code_step < 0:
        raise ValueError(f"max_code_step must be non-negative, got {max_code_step}")
    if not 0.0 <= max_changed_ratio <= 1.0:
        raise ValueError(
            f"max_changed_ratio must be in [0, 1], got {max_changed_ratio}"
        )
    if not 0.0 <= max_changed_per_row_ratio <= 1.0:
        raise ValueError(
            "max_changed_per_row_ratio must be in [0, 1], got "
            f"{max_changed_per_row_ratio}"
        )

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del actual_outputs, expected_outputs, inputs, rtol, atol
        actual = actual.cpu()
        expected = expected.cpu()
        if actual.shape != expected.shape:
            return False, (
                f"    QR shape mismatch: {tuple(actual.shape)} vs "
                f"{tuple(expected.shape)}"
            )
        if actual.dtype != torch.float8_e4m3fn or expected.dtype != torch.float8_e4m3fn:
            return False, (
                f"    QR comparator requires float8_e4m3fn tensors, got "
                f"{actual.dtype} and {expected.dtype}"
            )
        actual_codes = actual.contiguous().view(torch.uint8)
        expected_codes = expected.contiguous().view(torch.uint8)
        changed = actual_codes != expected_codes
        diff = (actual.float() - expected.float()).abs()
        changed_count = int(changed.count_nonzero().item())
        changed_limit = round(max_changed_ratio * diff.numel())
        changed_per_row = changed.reshape(-1, changed.shape[-1]).sum(dim=-1)
        row_limit = int(max_changed_per_row_ratio * changed.shape[-1])
        overfull_rows = changed_per_row > row_limit
        overfull_row_count = int(overfull_rows.count_nonzero().item())
        too_large = diff > float(max_code_step)
        too_large_count = int(too_large.count_nonzero().item())
        if (
            changed_count <= changed_limit
            and overfull_row_count == 0
            and too_large_count == 0
        ):
            return True, ""

        changed_indices = changed.flatten().nonzero(as_tuple=False).flatten()
        flat_actual = actual.flatten()
        flat_expected = expected.flatten()
        flat_diff = diff.flatten()
        lines = []
        for index in changed_indices[:max_show].tolist():
            lines.append(
                f"      [{index}] actual={int(flat_actual[index])} "
                f"expected={int(flat_expected[index])} "
            f"value_diff={float(flat_diff[index]):.4g}"
            )
        return False, (
            f"    QR quantization-boundary mismatch: changed={changed_count}/"
            f"{diff.numel()} (allowed<={max_changed_ratio:.4%}, "
            f"threshold={changed_limit}), code_step>{max_code_step}: "
            f"{too_large_count}, rows>{row_limit} changed codes: "
            f"{overfull_row_count}\n"
            + "\n".join(lines)
        )

    compare.__name__ = (
        f"quantized_qr_compare(max_code_step={max_code_step},"
        f"max_changed_ratio={max_changed_ratio},"
        f"max_changed_per_row_ratio={max_changed_per_row_ratio})"
    )
    return compare


def qr_scale_compare(
    *,
    atol=2.5e-5,
    rtol=5e-3,
    max_error_ratio=0.0,
    max_show=10,
):
    """Validate QR dequant scales with aggregate and per-row bounds."""
    import torch

    if atol < 0 or rtol < 0:
        raise ValueError("QR scale tolerances must be non-negative")
    if not 0.0 <= max_error_ratio <= 1.0:
        raise ValueError(
            f"max_error_ratio must be in [0, 1], got {max_error_ratio}"
        )
    scale_atol = atol
    scale_rtol = rtol

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del actual_outputs, expected_outputs, inputs, rtol, atol
        actual = actual.cpu().to(torch.float32)
        expected = expected.cpu().to(torch.float32)
        if actual.shape != expected.shape:
            return False, (
                f"    QR scale shape mismatch: {tuple(actual.shape)} vs "
                f"{tuple(expected.shape)}"
            )
        if not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
            return False, "    QR scales contain NaN or Inf"
        if (actual <= 0).any().item() or (expected <= 0).any().item():
            return False, "    QR scales must be positive"

        diff = (actual - expected).abs()
        tolerance = scale_atol + scale_rtol * expected.abs()
        bad = diff > tolerance
        bad_count = int(bad.count_nonzero().item())
        threshold = round(max_error_ratio * actual.numel())
        hard_bad = diff > (2.0 * tolerance)
        hard_bad_count = int(hard_bad.count_nonzero().item())
        if bad_count <= threshold and hard_bad_count == 0:
            return True, ""

        bad_indices = bad.flatten().nonzero(as_tuple=False).flatten()
        flat_actual = actual.flatten()
        flat_expected = expected.flatten()
        flat_diff = diff.flatten()
        flat_tolerance = tolerance.flatten()
        lines = []
        for index in bad_indices[:max_show].tolist():
            lines.append(
                f"      [{index}] actual={float(flat_actual[index]):.8g} "
                f"expected={float(flat_expected[index]):.8g} "
                f"diff={float(flat_diff[index]):.4g} "
                f"tol={float(flat_tolerance[index]):.4g}"
            )
        return False, (
            f"    QR scale mismatch: bad={bad_count}/{actual.numel()} "
            f"(allowed<={max_error_ratio:.4%}, threshold={threshold}), "
            f"hard_bad={hard_bad_count}, atol={scale_atol}, rtol={scale_rtol}\n"
            + "\n".join(lines)
        )

    compare.__name__ = (
        f"qr_scale_compare(atol={scale_atol},rtol={scale_rtol},"
        f"max_error_ratio={max_error_ratio})"
    )
    return compare


def q_from_runtime_qr_compare(
    *,
    atol=1e-4,
    rtol=1.0 / 128,
    max_error_ratio=0.005,
):
    """Validate Q against a reference conditioned on the emitted QR codes."""
    from golden import ratio_allclose

    base_compare = ratio_allclose(atol=atol, rtol=rtol, max_error_ratio=max_error_ratio)

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del expected
        required_outputs = ("qr", "qr_scale")
        required_inputs = ("wq_b", "wq_b_scale", "rope_cos", "rope_sin")
        missing_outputs = [name for name in required_outputs if name not in actual_outputs]
        missing_inputs = [name for name in required_inputs if name not in inputs]
        if missing_outputs or missing_inputs:
            return False, (
                "    conditioned Q comparator is missing "
                f"outputs={missing_outputs}, inputs={missing_inputs}"
            )

        conditioned = _reference_q_from_quantized_qr(
            actual_outputs["qr"].cpu(),
            actual_outputs["qr_scale"].cpu(),
            inputs["wq_b"].cpu(),
            inputs["wq_b_scale"].cpu(),
            inputs["rope_cos"].cpu(),
            inputs["rope_sin"].cpu(),
        )
        if actual.shape != conditioned.shape:
            return False, (
                f"    conditioned Q shape mismatch: actual={tuple(actual.shape)} "
                f"reference={tuple(conditioned.shape)}"
            )
        ok, detail = base_compare(
            actual,
            conditioned,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )
        if ok:
            return True, ""
        return False, "    Q downstream of emitted QR does not match:\n" + detail

    compare.__name__ = (
        f"q_from_runtime_qr_compare(atol={atol},rtol={rtol},"
        f"max_error_ratio={max_error_ratio})"
    )
    return compare


def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec
    from mx_utils import gen_mxfp8_weight_kn_device

    T = B * S

    def init_x():
        return torch.empty([T, D], dtype=torch.bfloat16).uniform_(-1, 1)

    def init_cos():
        return torch.empty([T, ROPE_DIM], dtype=torch.bfloat16).uniform_(-1, 1)

    def init_sin():
        return torch.empty([T, ROPE_DIM], dtype=torch.bfloat16).uniform_(-1, 1)

    def init_gamma_cq():
        return torch.empty([Q_LORA], dtype=torch.bfloat16).uniform_(-1, 1)

    def init_gamma_ckv():
        return torch.empty([HEAD_DIM], dtype=torch.bfloat16).uniform_(-1, 1)

    wq_a, wq_a_scale = gen_mxfp8_weight_kn_device(Q_LORA, D, 0.02, seed=11)
    wq_b, wq_b_scale = gen_mxfp8_weight_kn_device(H * HEAD_DIM, Q_LORA, 0.02, seed=12)
    wkv, wkv_scale = gen_mxfp8_weight_kn_device(HEAD_DIM, D, 0.02, seed=13)

    return [
        TensorSpec("x",         [T, D],                 torch.bfloat16, init_value=init_x),
        TensorSpec("wq_a", [D, Q_LORA], torch.float8_e4m3fn, init_value=lambda: wq_a),
        TensorSpec("wq_a_scale", [D // MX_GROUP, Q_LORA], torch.float8_e8m0fnu, init_value=lambda: wq_a_scale),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wq_b),
        TensorSpec("wq_b_scale", [Q_LORA // MX_GROUP, H * HEAD_DIM], torch.float8_e8m0fnu, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wkv),
        TensorSpec("wkv_scale", [D // MX_GROUP, HEAD_DIM], torch.float8_e8m0fnu, init_value=lambda: wkv_scale),
        TensorSpec("rope_cos",  [T, ROPE_DIM],          torch.bfloat16, init_value=init_cos),
        TensorSpec("rope_sin",  [T, ROPE_DIM],          torch.bfloat16, init_value=init_sin),
        TensorSpec("gamma_cq",  [Q_LORA],               torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM],             torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("q",         [T, H, HEAD_DIM],       torch.bfloat16),
        TensorSpec("kv",        [T, HEAD_DIM],          torch.bfloat16),
        TensorSpec("qr", [T_MAX, Q_LORA], torch.float8_e4m3fn),
        TensorSpec("qr_scale", [1, T_MAX * (Q_LORA // MX_GROUP)], torch.float8_e8m0fnu),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    MODES = {
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--mode", choices=["decode", "prefill", "all"], default="all",
        help="Use decode or prefill batch sizes, or 'all' to test both.",
    )
    parser.add_argument(
        "--enable-chip-swimlane", type=int, choices=[0, 1, 2, 4], default=0,
        help="chip swimlane level: 0=off, 1=per-kernel AICore timing "
        "(prints the per-function Task Statistics table), 2=+AICPU timing.",
    )
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- qkv_proj_rope {mode_name}: B={B}, S={S} ---")
        result = run(
            fn=qkv_proj_rope_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_qkv_proj_rope,
            rtol=5e-3,
            atol=5e-3,
            compare_fn={
                "q":        q_from_runtime_qr_compare(atol=1e-4, rtol=1.0 / 128),
                "kv":       ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "qr":       quantized_qr_compare(max_code_step=1, max_changed_ratio=0.005),
                "qr_scale": qr_scale_compare(atol=2.5e-5, rtol=5e-3, max_error_ratio=0.0),
            },
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            config=dict(
                dump_passes=args.dump_passes,
                platform=args.platform,
                device_id=args.device,
                enable_chip_swimlane=args.enable_chip_swimlane,
            ),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
