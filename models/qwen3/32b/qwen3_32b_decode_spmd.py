# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B single-layer decode forward (SPMD-4 variant).

Scope 1:
  1. RMSNorm of input hidden states
  2. Q/K/V projection via matmul

Scope 2:
  1. K RoPE + cache write, V cache write, Q RoPE + pad
  2. QK matmul
  3. Softmax
  4. SV matmul
  5. Online-softmax accumulation + final normalisation

Scope 3:
  1. Output projection: attn_out × wo
  2. Residual addition with hidden_states
  3. Post-attention RMSNorm
  4. MLP: gate/up projections, SiLU activation, down projection
  5. Final residual addition
"""

import pypto.language as pl

BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 8192
INTERMEDIATE = 25600
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Scope 1 tiles
RMSNORM_K_CHUNK = 512
Q_OUT_CHUNK = 256
Q_PROJ_K_CHUNK = 128
KV_OUT_CHUNK = 256
KV_PROJ_K_CHUNK = 128
BATCH_TILE = 16

# Scope 2 tiles
Q_HEAD_BATCH = 8
Q_HEAD_PAD = 16
SEQ_TILE = 256

# Scope 3 tiles
K_CHUNK = 128
OUT_PROJ_K_CHUNK = 128
MLP_OUT_CHUNK = 256
DOWN_N_CHUNK = 256
DOWN_K_CHUNK = 128

CACHE_ROWS = BATCH * NUM_KV_HEADS * MAX_SEQ
HALF_DIM = HEAD_DIM // 2
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH
TOTAL_Q_GROUPS = NUM_KV_HEADS * Q_GROUPS
MAX_CTX_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE

# Number of SPMD cores for data-parallel partitioning.
SPMD_CORES = 4


def build_qwen3_decode_program():
    hidden = HIDDEN
    kv_hidden = KV_HIDDEN
    inter = INTERMEDIATE
    batch = BATCH
    max_seq = MAX_SEQ
    num_kv_heads = NUM_KV_HEADS
    head_dim = HEAD_DIM
    scope1_hidden_blocks = hidden // RMSNORM_K_CHUNK
    hidden_blocks = hidden // K_CHUNK
    q_out_blocks = hidden // Q_OUT_CHUNK
    kv_out_blocks = kv_hidden // KV_OUT_CHUNK
    mlp_out_blocks = inter // MLP_OUT_CHUNK
    cache_rows = CACHE_ROWS
    half_dim = HALF_DIM
    q_per_kv = Q_PER_KV
    q_groups = Q_GROUPS
    total_q_groups = TOTAL_Q_GROUPS
    attn_scale = ATTN_SCALE
    max_ctx_blocks = MAX_CTX_BLOCKS
    q_proj_inner_blocks = 4
    kv_proj_inner_blocks = 4
    q_pad_init_inner = 8
    rope_kv_inner = num_kv_heads // SPMD_CORES
    out_proj_inner_blocks = 1
    down_proj_inner_blocks = 2

    q_proj_spmd_blocks = q_out_blocks // q_proj_inner_blocks
    kv_proj_spmd_blocks = kv_out_blocks // kv_proj_inner_blocks
    q_pad_init_spmd_blocks = (batch * total_q_groups) // q_pad_init_inner
    out_proj_spmd_blocks = q_out_blocks // out_proj_inner_blocks
    down_proj_spmd_blocks = hidden_blocks // down_proj_inner_blocks

    assert q_out_blocks % q_proj_inner_blocks == 0
    assert kv_out_blocks % kv_proj_inner_blocks == 0
    assert (batch * total_q_groups) % q_pad_init_inner == 0
    assert num_kv_heads % SPMD_CORES == 0
    assert q_out_blocks % out_proj_inner_blocks == 0
    assert hidden_blocks % down_proj_inner_blocks == 0

    @pl.program
    class Qwen3Decode:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_decode(
            self,
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            wq: pl.Tensor[[hidden, hidden], pl.BF16],
            wk: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            v_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            wo: pl.Tensor[[hidden, hidden], pl.BF16],
            post_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            w_gate: pl.Tensor[[hidden, inter], pl.BF16],
            w_up: pl.Tensor[[hidden, inter], pl.BF16],
            w_down: pl.Tensor[[inter, hidden], pl.BF16],
            out: pl.Out[pl.Tensor[[batch, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, hidden], pl.BF16]:
            # Intermediate FP32 tensors between scope 1 and scope 2.
            q_proj = pl.create_tensor([batch, hidden], dtype=pl.FP32)
            k_proj = pl.create_tensor([batch, kv_hidden], dtype=pl.FP32)
            v_proj = pl.create_tensor([batch, kv_hidden], dtype=pl.FP32)

            # ── Scope 1: input RMSNorm + Q/K/V projection ──
            for b0 in pl.parallel(0, batch, BATCH_TILE):
                normed_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.BF16)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                    partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.pipeline(scope1_hidden_blocks, stage=4):
                        k0 = kb * RMSNORM_K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, RMSNORM_K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        partial_sq = pl.add(
                            partial_sq,
                            pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_TILE]),
                        )
                    variance = pl.reshape(
                        pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                        [BATCH_TILE, 1],
                    )
                    inv_rms = pl.recip(pl.sqrt(variance))

                    for kb in pl.pipeline(scope1_hidden_blocks, stage=4):
                        k0 = kb * RMSNORM_K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, RMSNORM_K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        gamma = pl.slice(input_rms_weight, [1, RMSNORM_K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                        normed_tile = pl.assemble(normed_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                # Q projection — SPMD over output columns.
                for ob0 in pl.spmd(q_proj_spmd_blocks, level=pl.Level.CORE_GROUP, name_hint="q_proj"):
                    for j in pl.range(q_proj_inner_blocks):
                        ob = ob0 * q_proj_inner_blocks + j
                        q0 = ob * Q_OUT_CHUNK
                        tile_a = pl.slice(normed_tile, [BATCH_TILE, Q_PROJ_K_CHUNK], [0, 0])
                        tile_b = pl.slice(wq, [Q_PROJ_K_CHUNK, Q_OUT_CHUNK], [0, q0])
                        q_acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)

                        tile_a_1 = pl.slice(normed_tile, [BATCH_TILE, Q_PROJ_K_CHUNK], [0, Q_PROJ_K_CHUNK])
                        tile_b_1 = pl.slice(wq, [Q_PROJ_K_CHUNK, Q_OUT_CHUNK], [Q_PROJ_K_CHUNK, q0])
                        q_acc = pl.matmul_acc(q_acc, tile_a_1, tile_b_1)

                        for kb in pl.pipeline(2, scope1_hidden_blocks, stage=2):
                            k0 = kb * Q_PROJ_K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [BATCH_TILE, Q_PROJ_K_CHUNK], [0, k0])
                            tile_b_i = pl.slice(wq, [Q_PROJ_K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                            q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_b_i)
                        q_proj = pl.assemble(q_proj, q_acc, [b0, q0])

                # K/V projection — SPMD over output columns.
                for ob0 in pl.spmd(kv_proj_spmd_blocks, level=pl.Level.CORE_GROUP, name_hint="kv_proj"):
                    for j in pl.range(kv_proj_inner_blocks):
                        ob = ob0 * kv_proj_inner_blocks + j
                        kv0 = ob * KV_OUT_CHUNK
                        k_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                        v_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                        for kb in pl.pipeline(0, HIDDEN // KV_PROJ_K_CHUNK, stage=2):
                            k0 = kb * KV_PROJ_K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [BATCH_TILE, KV_PROJ_K_CHUNK], [0, k0])
                            tile_wk_i = pl.slice(wk, [KV_PROJ_K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            tile_wv_i = pl.slice(wv, [KV_PROJ_K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            if k0 == 0:
                                k_acc = pl.matmul(tile_a_i, tile_wk_i, out_dtype=pl.FP32)
                                v_acc = pl.matmul(tile_a_i, tile_wv_i, out_dtype=pl.FP32)
                            else:
                                k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                                v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                        k_proj = pl.assemble(k_proj, k_acc, [b0, kv0])
                        v_proj = pl.assemble(v_proj, v_acc, [b0, kv0])

            # ── Scope 2: RoPE + KV cache update + grouped-query attention ──
            all_q_padded = pl.create_tensor([batch * total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16)
            for ob0 in pl.spmd(q_pad_init_spmd_blocks, level=pl.Level.CORE_GROUP, name_hint="q_pad_init"):
                for j in pl.range(q_pad_init_inner):
                    idx = ob0 * q_pad_init_inner + j
                    all_q_padded = pl.assemble(
                        all_q_padded,
                        pl.cast(pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, head_dim], dtype=pl.FP32, value=0.0), target_type=pl.BF16),
                        [idx * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                    )

            attn_out = pl.create_tensor([batch, hidden], dtype=pl.BF16)
            for b in pl.parallel(batch):
                ctx_len = pl.read(seq_lens, [b])
                pos = ctx_len - 1
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                cos_row = pl.slice(rope_cos, [1, head_dim], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, head_dim], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                cos_hi = pl.slice(cos_row, [1, half_dim], [0, half_dim])
                sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                sin_hi = pl.slice(sin_row, [1, half_dim], [0, half_dim])

                # Stage 1: K RoPE + cache update + V cache + Q RoPE + pad.
                for ki0 in pl.spmd(SPMD_CORES, level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
                    for j in pl.range(rope_kv_inner):
                        ki = ki0 * rope_kv_inner + j
                        kv_col = ki * head_dim
                        k_lo = pl.slice(k_proj, [1, half_dim], [b, kv_col])
                        k_hi = pl.slice(k_proj, [1, half_dim], [b, kv_col + half_dim])
                        rot_lo = pl.sub(
                            pl.col_expand_mul(k_lo, cos_lo),
                            pl.col_expand_mul(k_hi, sin_lo),
                        )
                        rot_hi = pl.add(
                            pl.col_expand_mul(k_hi, cos_hi),
                            pl.col_expand_mul(k_lo, sin_hi),
                        )
                        cache_row = b * num_kv_heads * max_seq + ki * max_seq + pos
                        k_cache = pl.assemble(k_cache, pl.cast(rot_lo, target_type=pl.BF16), [cache_row, 0])
                        k_cache = pl.assemble(k_cache, pl.cast(rot_hi, target_type=pl.BF16), [cache_row, half_dim])
                        v_cache = pl.assemble(
                            v_cache,
                            pl.cast(pl.slice(v_proj, [1, head_dim], [b, ki * head_dim]), target_type=pl.BF16),
                            [cache_row, 0],
                        )
                        q_base = ki * q_per_kv
                        for qi in pl.range(Q_HEAD_BATCH):
                            q_col = (q_base + qi) * head_dim
                            q_lo = pl.slice(q_proj, [1, half_dim], [b, q_col])
                            q_hi = pl.slice(q_proj, [1, half_dim], [b, q_col + half_dim])
                            rot_lo_bf16 = pl.cast(
                                pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo)),
                                target_type=pl.BF16,
                            )
                            rot_hi_bf16 = pl.cast(
                                pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi)),
                                target_type=pl.BF16,
                            )
                            all_q_padded = pl.assemble(all_q_padded, rot_lo_bf16, [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD + qi, 0])
                            all_q_padded = pl.assemble(all_q_padded, rot_hi_bf16, [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD + qi, half_dim])

                attn_row = pl.create_tensor([1, hidden], dtype=pl.BF16)
                for gi in pl.parallel(total_q_groups):
                    kvh = gi // q_groups
                    qg = gi - kvh * q_groups
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_padded = pl.slice(all_q_padded, [Q_HEAD_PAD, head_dim], [b * total_q_groups * Q_HEAD_PAD + gi * Q_HEAD_PAD, 0])

                    # Stage 2: QK matmul.
                    all_raw_scores = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                    all_exp_padded = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16)
                    all_oi_tmp = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, head_dim], dtype=pl.FP32)
                    all_cur_mi = pl.create_tensor([max_ctx_blocks * Q_HEAD_BATCH, 1], dtype=pl.FP32)
                    all_cur_li = pl.create_tensor([max_ctx_blocks * Q_HEAD_BATCH, 1], dtype=pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_matmul"):
                        for sb in pl.range(ctx_blocks):
                            s0 = sb * SEQ_TILE
                            cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0
                            k_tile = pl.slice(
                                k_cache,
                                [SEQ_TILE, head_dim],
                                [cache_row0, 0],
                            )
                            raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                            all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

                    # Stage 3: softmax.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax"):
                        for sb in pl.range(ctx_blocks):
                            s0 = sb * SEQ_TILE
                            valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                            scores_valid = pl.slice(
                                all_raw_scores,
                                [Q_HEAD_BATCH, SEQ_TILE],
                                [sb * Q_HEAD_PAD, 0],
                                valid_shape=[Q_HEAD_BATCH, valid_len],
                            )
                            scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                            scores = pl.mul(scores_padded, attn_scale)
                            cur_mi = pl.row_max(scores)
                            exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                            exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                            cur_li = pl.row_sum(exp_scores_fp32)
                            all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0])
                            all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_BATCH, 0])
                            all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_BATCH, 0])

                    # Stage 4: SV matmul.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_matmul"):
                        for sb in pl.range(ctx_blocks):
                            s0 = sb * SEQ_TILE
                            cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0
                            exp_tile = pl.slice(
                                all_exp_padded,
                                [Q_HEAD_PAD, SEQ_TILE],
                                [sb * Q_HEAD_PAD, 0],
                            )
                            v_tile = pl.slice(
                                v_cache,
                                [SEQ_TILE, head_dim],
                                [cache_row0, 0],
                            )
                            oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
                            all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])

                    # Stage 5: online softmax accumulation and normalisation.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax"):
                        oi = pl.slice(all_oi_tmp, [Q_HEAD_BATCH, head_dim], [0, 0])
                        mi = pl.slice(all_cur_mi, [Q_HEAD_BATCH, 1], [0, 0])
                        li = pl.slice(all_cur_li, [Q_HEAD_BATCH, 1], [0, 0])
                        for sb in pl.range(1, ctx_blocks):
                            oi_tmp_valid = pl.slice(all_oi_tmp, [Q_HEAD_BATCH, head_dim], [sb * Q_HEAD_PAD, 0])
                            cur_mi = pl.slice(all_cur_mi, [Q_HEAD_BATCH, 1], [sb * Q_HEAD_BATCH, 0])
                            cur_li = pl.slice(all_cur_li, [Q_HEAD_BATCH, 1], [sb * Q_HEAD_BATCH, 0])
                            mi_new = pl.maximum(mi, cur_mi)
                            alpha = pl.exp(pl.sub(mi, mi_new))
                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                            li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                            oi = pl.add(pl.row_expand_mul(oi, alpha),
                                        pl.row_expand_mul(oi_tmp_valid, beta))
                            mi = mi_new
                        ctx = pl.row_expand_div(oi, li)
                        ctx_flat = pl.reshape(ctx, [1, Q_HEAD_BATCH * head_dim])
                        ctx_flat_bf16 = pl.cast(ctx_flat, target_type=pl.BF16)
                        attn_row = pl.assemble(
                            attn_row, ctx_flat_bf16, [0, q_base * head_dim],
                        )

                attn_out = pl.assemble(attn_out, attn_row, [b, 0])

            # ── Scope 3: output projection + residual + post RMSNorm + MLP + residual ──
            for b0 in pl.parallel(0, batch, BATCH_TILE):
                resid1_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.FP32)

                # Stage 1 & 2: Output projection + residual addition with hidden_states.
                for o0 in pl.parallel(0, hidden, Q_OUT_CHUNK):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj_residual"):
                        hidden_chunk = pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0])
                        o_acc = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                        for kb in pl.pipeline(0, hidden_blocks, stage=2):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0])
                            w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            if k0 == 0:
                                o_acc = pl.matmul(a_chunk, w_chunk, out_dtype=pl.FP32)
                            else:
                                o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

                        resid = pl.cast(hidden_chunk, target_type=pl.FP32)
                        resid_sum = pl.add(o_acc, resid)
                        resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

                # Stage 3: Post-attention RMSNorm.
                post_norm_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.BF16)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.pipeline(hidden_blocks, stage=2):
                        k0 = kb * K_CHUNK
                        resid_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, BATCH_TILE]),
                        )
                    inv_rms_s3 = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))

                    for kb in pl.pipeline(hidden_blocks, stage=2):
                        k0 = kb * K_CHUNK
                        resid_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        post_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        post_normed = pl.col_expand_mul(
                            pl.row_expand_mul(resid_chunk, pl.reshape(inv_rms_s3, [BATCH_TILE, 1])),
                            post_gamma,
                        )
                        normed_bf16 = pl.cast(post_normed, target_type=pl.BF16)
                        post_norm_tile = pl.assemble(post_norm_tile, normed_bf16, [0, k0])

                # Stage 4 & 5 & 6: MLP gate/up projections + SiLU (align decode.py).
                mlp_tile = pl.create_tensor([BATCH_TILE, inter], dtype=pl.BF16)
                for o0 in pl.parallel(0, inter, MLP_OUT_CHUNK):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj"):
                        gate_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                        for kb in pl.pipeline(0, hidden_blocks, stage=2):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            if k0 == 0:
                                gate_acc = pl.matmul(post_chunk, wg, out_dtype=pl.FP32)
                            else:
                                gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj"):
                        up_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                        for kb in pl.pipeline(0, hidden_blocks, stage=2):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            if k0 == 0:
                                up_acc = pl.matmul(post_chunk, wu, out_dtype=pl.FP32)
                            else:
                                up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu"):
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_tile = pl.assemble(mlp_tile, pl.cast(mlp_chunk, target_type=pl.BF16), [0, o0])

                # Stage 7 & 8: Down projection + final residual writeback.
                for ob0 in pl.spmd(down_proj_spmd_blocks, level=pl.Level.CORE_GROUP, name_hint="down_proj_residual"):
                    for j in pl.range(down_proj_inner_blocks):
                        dob = ob0 * down_proj_inner_blocks + j
                        d0 = dob * K_CHUNK
                        mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, 0])
                        w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [0, d0])
                        resid1_tile_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0])

                        down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)

                        mlp_chunk_1 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, MLP_OUT_CHUNK])
                        w_down_chunk_1 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [MLP_OUT_CHUNK, d0])
                        down_acc = pl.matmul_acc(down_acc, mlp_chunk_1, w_down_chunk_1)

                        for ob in pl.pipeline(2, mlp_out_blocks, stage=2):
                            o0 = ob * MLP_OUT_CHUNK
                            down_mlp_chunk_bf16 = pl.slice(
                                mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, o0]
                            )
                            w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                            down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)

                        out_chunk = pl.add(down_acc, resid1_tile_chunk)
                        out_chunk_cast = pl.cast(out_chunk, target_type=pl.BF16)
                        out = pl.assemble(out, out_chunk_cast, [b0, d0])

            return out

    return Qwen3Decode


def build_tensor_specs(use_max_seq: bool = False):
    import torch
    from golden import TensorSpec

    hidden = HIDDEN
    kv_hidden = KV_HIDDEN
    inter = INTERMEDIATE
    cache_rows = CACHE_ROWS

    def init_hidden_states():
        return torch.rand(BATCH, HIDDEN) - 0.5

    def init_rms_weight():
        return torch.rand(1, HIDDEN) - 0.5

    def init_wq():
        return torch.rand(HIDDEN, HIDDEN) / HIDDEN ** 0.5

    def init_wk():
        return torch.rand(HIDDEN, KV_HIDDEN) / HIDDEN ** 0.5

    def init_wv():
        return torch.rand(HIDDEN, KV_HIDDEN) / HIDDEN ** 0.5

    def init_seq_lens():
        if use_max_seq:
            return torch.full((BATCH,), MAX_SEQ, dtype=torch.int32)
        return torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32)

    def init_rope_cos():
        return torch.rand(MAX_SEQ, HEAD_DIM) - 0.5

    def init_rope_sin():
        return torch.rand(MAX_SEQ, HEAD_DIM) - 0.5

    def init_k_cache():
        return torch.rand(CACHE_ROWS, HEAD_DIM) - 0.5

    def init_v_cache():
        return torch.rand(CACHE_ROWS, HEAD_DIM) - 0.5

    def init_wo():
        return (torch.rand(HIDDEN, HIDDEN) - 0.5) / HIDDEN ** 0.5

    def init_post_rms_weight():
        return torch.ones(1, HIDDEN)

    def init_w_gate():
        return (torch.rand(HIDDEN, INTERMEDIATE) - 0.5) / HIDDEN ** 0.5

    def init_w_up():
        return (torch.rand(HIDDEN, INTERMEDIATE) - 0.5) / HIDDEN ** 0.5

    def init_w_down():
        return (torch.rand(INTERMEDIATE, HIDDEN) - 0.5) / INTERMEDIATE ** 0.5

    return [
        TensorSpec("hidden_states", [BATCH, HIDDEN], torch.bfloat16,
                   init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, HIDDEN], torch.float32,
                   init_value=init_rms_weight),
        TensorSpec("wq", [HIDDEN, HIDDEN], torch.bfloat16,
                   init_value=init_wq),
        TensorSpec("wk", [HIDDEN, KV_HIDDEN], torch.bfloat16,
                   init_value=init_wk),
        TensorSpec("wv", [HIDDEN, KV_HIDDEN], torch.bfloat16,
                   init_value=init_wv),
        TensorSpec("seq_lens", [BATCH], torch.int32, init_value=init_seq_lens),
        TensorSpec("rope_cos", [MAX_SEQ, HEAD_DIM], torch.float32,
                   init_value=init_rope_cos),
        TensorSpec("rope_sin", [MAX_SEQ, HEAD_DIM], torch.float32,
                   init_value=init_rope_sin),
        TensorSpec("k_cache", [CACHE_ROWS, HEAD_DIM], torch.bfloat16,
                   init_value=init_k_cache),
        TensorSpec("v_cache", [CACHE_ROWS, HEAD_DIM], torch.bfloat16,
                   init_value=init_v_cache),
        TensorSpec("wo", [HIDDEN, HIDDEN], torch.bfloat16,
                   init_value=init_wo),
        TensorSpec("post_rms_weight", [1, HIDDEN], torch.float32,
                   init_value=init_post_rms_weight),
        TensorSpec("w_gate", [HIDDEN, INTERMEDIATE], torch.bfloat16,
                   init_value=init_w_gate),
        TensorSpec("w_up", [HIDDEN, INTERMEDIATE], torch.bfloat16,
                   init_value=init_w_up),
        TensorSpec("w_down", [INTERMEDIATE, HIDDEN], torch.bfloat16,
                   init_value=init_w_down),
        TensorSpec("out", [BATCH, HIDDEN], torch.bfloat16, is_output=True),
    ]


def golden_qwen3_decode_scope1(tensors):
    """PyTorch reference for scope 1 only: fill ``q_proj`` / ``k_proj`` / ``v_proj`` (FP32)."""
    import torch

    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    kv_hidden = wk.shape[1]

    q_proj = torch.zeros(batch, hidden_size, dtype=torch.float32)
    k_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)
    v_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)

    for b0 in range(0, batch, BATCH_TILE):
        b_end = min(b0 + BATCH_TILE, batch)
        x_tile = hidden_states[b0:b_end, :].float()

        sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, RMSNORM_K_CHUNK):
            x_chunk = x_tile[:, k0:k0 + RMSNORM_K_CHUNK]
            sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)
        variance = sq_sum / hidden_size + EPS
        rms = torch.sqrt(variance)
        normed = (x_tile / rms * input_rms_weight.float()).bfloat16()

        q_proj[b0:b_end, :] = (normed.float() @ wq.float()).float()
        k_proj[b0:b_end, :] = (normed.float() @ wk.float()).float()
        v_proj[b0:b_end, :] = (normed.float() @ wv.float()).float()

    tensors["q_proj"][:] = q_proj
    tensors["k_proj"][:] = k_proj
    tensors["v_proj"][:] = v_proj


def golden_qwen3_decode_scope2(tensors):
    """PyTorch reference for scope 2 only: ``attn_out`` (BF16), same attention path as full golden."""
    import math

    import torch

    q_proj = tensors["q_proj"].float()
    k_proj = tensors["k_proj"].float()
    v_proj = tensors["v_proj"].float()
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()

    batch = q_proj.shape[0]
    hidden_size = q_proj.shape[1]
    kv_hidden = k_proj.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden_size // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)

    attn_out = torch.zeros(batch, hidden_size, dtype=torch.float32)

    for b in range(batch):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_proj[b].view(num_kv_heads, head_dim)
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat([k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi], dim=-1)

        for ki in range(num_kv_heads):
            cr = b * num_kv_heads * max_seq + ki * max_seq + pos
            k_cache[cr, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cr, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(torch.bfloat16)

        q_heads = q_proj[b].view(num_heads, head_dim)
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat([q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi], dim=-1)

        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * SEQ_TILE
                    valid_len = min(SEQ_TILE, ctx_len - s0)
                    cb = b * num_kv_heads * max_seq + kvh * max_seq + s0

                    k_tile = k_cache[cb : cb + SEQ_TILE, :]
                    v_tile = v_cache[cb : cb + SEQ_TILE, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < SEQ_TILE:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale

                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)

                    oi_tmp = exp_scores_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi = oi_tmp
                        li = cur_li
                        mi = cur_mi
                    else:
                        mi_new = torch.maximum(mi, cur_mi)
                        alpha = torch.exp(mi - mi_new)
                        beta = torch.exp(cur_mi - mi_new)
                        li = alpha * li + beta * cur_li
                        oi = oi * alpha + oi_tmp * beta
                        mi = mi_new

                ctx = oi / li
                for qi in range(Q_HEAD_BATCH):
                    qh = q_base + qi
                    attn_out[b, qh * head_dim : (qh + 1) * head_dim] = ctx[qi]

    tensors["attn_out"][:] = attn_out.to(torch.bfloat16)


def golden_qwen3_decode_scope3(tensors):
    """PyTorch reference for scope 3 only: MLP path into ``out`` (BF16)."""
    import torch

    hidden_states = tensors["hidden_states"]
    attn_out = tensors["attn_out"].float()
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]
    eps = 1e-6

    o_proj = torch.matmul(attn_out, wo.float())
    resid1 = o_proj + hidden_states.float()

    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())

    tensors["out"][:] = (down + resid1).bfloat16()


def golden_qwen3_decode(tensors):
    """PyTorch reference: scope1 (RMSNorm + projection), scope2 (attention), scope3 (output + MLP)."""
    import torch

    hidden = tensors["hidden_states"].shape[1]
    kv_hidden = tensors["wk"].shape[1]
    batch = tensors["hidden_states"].shape[0]

    q_proj = torch.zeros(batch, hidden, dtype=torch.float32)
    k_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)
    v_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)
    s1 = {
        "hidden_states": tensors["hidden_states"],
        "input_rms_weight": tensors["input_rms_weight"],
        "wq": tensors["wq"],
        "wk": tensors["wk"],
        "wv": tensors["wv"],
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
    }
    golden_qwen3_decode_scope1(s1)

    attn_buf = torch.zeros(batch, hidden, dtype=torch.bfloat16)
    s2 = {
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "seq_lens": tensors["seq_lens"],
        "rope_cos": tensors["rope_cos"],
        "rope_sin": tensors["rope_sin"],
        "k_cache": tensors["k_cache"],
        "v_cache": tensors["v_cache"],
        "attn_out": attn_buf,
    }
    golden_qwen3_decode_scope2(s2)

    s3 = {
        "hidden_states": tensors["hidden_states"],
        "attn_out": attn_buf,
        "wo": tensors["wo"],
        "post_rms_weight": tensors["post_rms_weight"],
        "w_gate": tensors["w_gate"],
        "w_up": tensors["w_up"],
        "w_down": tensors["w_down"],
        "out": tensors["out"],
    }
    golden_qwen3_decode_scope3(s3)




if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    parser.add_argument("--max-seq", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_qwen3_decode_program(),
        specs=build_tensor_specs(use_max_seq=args.max_seq),
        golden_fn=golden_qwen3_decode,
        config=RunConfig(
            rtol=3e-3,
            atol=3e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)