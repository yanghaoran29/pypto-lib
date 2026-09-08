# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared MXFP8 grouped output projections for DeepSeek-V4-Pro attention."""

import pypto.language as pl

from config import ACTIVE as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


D = M.hidden_size
HEAD_DIM = M.head_dim
H = M.num_attention_heads
O_GROUPS = M.o_groups
O_LORA = M.o_lora_rank
O_GROUP_IN = (H // O_GROUPS) * HEAD_DIM
O_LORA_TOTAL = O_GROUPS * O_LORA
MX_GROUP = 32

DECODE_T = DECODE_BATCH * DECODE_SEQ
PREFILL_T = PREFILL_BATCH * PREFILL_SEQ
MM_M_TILE = 16
A_K_TILE = 256
A_N_TILE = 128
B_K_TILE = 256
B_N_TILE = 256
DECODE_T_PAD = ((DECODE_T + MM_M_TILE - 1) // MM_M_TILE) * MM_M_TILE


@pl.jit.inline
def decode_output_proj_mx(
    o_packed: pl.Tensor[[O_GROUPS * DECODE_T, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[O_GROUPS * (O_GROUP_IN // MX_GROUP), O_LORA], pl.FP8E8M0, pl.MX_B_NN],
    wo_b: pl.Tensor[[O_LORA_TOTAL, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[O_LORA_TOTAL // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    attn_out: pl.Tensor[[DECODE_T, D], pl.BF16],
):
    """Project decode attention output through two MXFP8 Cube matmuls."""
    o_a_pad = pl.create_tensor([O_GROUPS * DECODE_T_PAD, O_GROUP_IN], dtype=pl.BF16)
    for pad_a_idx in pl.spmd(O_GROUPS * (O_GROUP_IN // A_K_TILE), name_hint="o_proj_a_pad"):
        pad_a_group = pad_a_idx // (O_GROUP_IN // A_K_TILE)
        pad_a_k0 = (pad_a_idx % (O_GROUP_IN // A_K_TILE)) * A_K_TILE
        o_a_pad[
            pad_a_group * DECODE_T_PAD : (pad_a_group + 1) * DECODE_T_PAD,
            pad_a_k0 : pad_a_k0 + A_K_TILE,
        ] = pl.full([MM_M_TILE, A_K_TILE], dtype=pl.BF16, value=0.0)
    for copy_a_idx in pl.spmd(O_GROUPS * (O_GROUP_IN // A_K_TILE), name_hint="o_proj_a_copy"):
        copy_a_group = copy_a_idx // (O_GROUP_IN // A_K_TILE)
        copy_a_k0 = (copy_a_idx % (O_GROUP_IN // A_K_TILE)) * A_K_TILE
        o_a_pad[
            copy_a_group * DECODE_T_PAD : copy_a_group * DECODE_T_PAD + DECODE_T,
            copy_a_k0 : copy_a_k0 + A_K_TILE,
        ] = o_packed[
            copy_a_group * DECODE_T : (copy_a_group + 1) * DECODE_T,
            copy_a_k0 : copy_a_k0 + A_K_TILE,
        ]

    o_a_mx = pl.create_tensor([O_GROUPS * DECODE_T_PAD, O_GROUP_IN], dtype=pl.FP8E4M3FN)
    o_a_scale_backing = pl.create_tensor(
        [1, O_GROUPS * DECODE_T_PAD * (O_GROUP_IN // MX_GROUP)],
        dtype=pl.FP8E8M0,
    )
    for quant_a_idx in pl.spmd(O_GROUPS * (O_GROUP_IN // A_K_TILE), name_hint="o_proj_a_quant_mx"):
        quant_a_group = quant_a_idx // (O_GROUP_IN // A_K_TILE)
        quant_a_k0 = (quant_a_idx % (O_GROUP_IN // A_K_TILE)) * A_K_TILE
        quant_a_src = pl.load(
            o_a_pad,
            [quant_a_group * DECODE_T_PAD, quant_a_k0],
            [MM_M_TILE, A_K_TILE],
        )
        quant_a_fp32 = pl.cast(quant_a_src, target_type=pl.FP32, mode="none")
        quant_a_data, quant_a_scale = pl.quant_mx(quant_a_fp32, group_axis=1)
        quant_a_row = quant_a_group * DECODE_T_PAD
        o_a_mx = pl.store(quant_a_data, [quant_a_row, quant_a_k0], o_a_mx)
        quant_a_scale_block = quant_a_idx * MM_M_TILE * (A_K_TILE // MX_GROUP)
        quant_a_scale_flat = pl.reshape(quant_a_scale, [1, MM_M_TILE * (A_K_TILE // MX_GROUP)])
        o_a_scale_backing = pl.store(
            quant_a_scale_flat,
            [0, quant_a_scale_block],
            o_a_scale_backing,
        )

    o_a_scale = pl.tensor.view(
        o_a_scale_backing,
        [O_GROUPS * DECODE_T_PAD, O_GROUP_IN // MX_GROUP],
        layout=pl.MX_A_ZZ,
    )
    wo_a_flat = pl.reshape(wo_a, [O_GROUPS * O_GROUP_IN, O_LORA])
    o_r = pl.create_tensor([DECODE_T_PAD, O_LORA_TOTAL], dtype=pl.FP32)
    with pl.spmd(O_LORA // A_N_TILE, name_hint="o_proj_a_mx") as _proj_a_tid:
        proj_a_n0 = pl.tile.get_block_idx() * A_N_TILE
        for proj_a_group in pl.range(O_GROUPS):
            proj_a_row = proj_a_group * DECODE_T_PAD
            proj_a_scale_row = proj_a_group * (O_GROUP_IN // MX_GROUP)
            proj_a_lhs0 = pl.load(o_a_mx, [proj_a_row, 0], [MM_M_TILE, A_K_TILE])
            proj_a_lhs_scale0 = pl.load(
                o_a_scale,
                [proj_a_row, 0],
                [MM_M_TILE, A_K_TILE // MX_GROUP],
            )
            proj_a_rhs0 = pl.load(
                wo_a_flat,
                [proj_a_group * O_GROUP_IN, proj_a_n0],
                [A_K_TILE, A_N_TILE],
            )
            proj_a_rhs_scale0 = pl.load(
                wo_a_scale,
                [proj_a_scale_row, proj_a_n0],
                [A_K_TILE // MX_GROUP, A_N_TILE],
            )
            proj_a_acc = pl.matmul_mx(
                proj_a_lhs0,
                proj_a_lhs_scale0,
                proj_a_rhs0,
                proj_a_rhs_scale0,
            )
            for proj_a_k0 in pl.range(A_K_TILE, O_GROUP_IN, A_K_TILE):
                proj_a_ks = proj_a_k0 // MX_GROUP
                proj_a_lhs = pl.load(o_a_mx, [proj_a_row, proj_a_k0], [MM_M_TILE, A_K_TILE])
                proj_a_lhs_scale = pl.load(
                    o_a_scale,
                    [proj_a_row, proj_a_ks],
                    [MM_M_TILE, A_K_TILE // MX_GROUP],
                )
                proj_a_rhs = pl.load(
                    wo_a_flat,
                    [proj_a_group * O_GROUP_IN + proj_a_k0, proj_a_n0],
                    [A_K_TILE, A_N_TILE],
                )
                proj_a_rhs_scale = pl.load(
                    wo_a_scale,
                    [proj_a_scale_row + proj_a_ks, proj_a_n0],
                    [A_K_TILE // MX_GROUP, A_N_TILE],
                )
                proj_a_acc = pl.matmul_mx_acc(
                    proj_a_acc,
                    proj_a_lhs,
                    proj_a_lhs_scale,
                    proj_a_rhs,
                    proj_a_rhs_scale,
                )
            o_r = pl.store(proj_a_acc, [0, proj_a_group * O_LORA + proj_a_n0], o_r)

    o_b_mx = pl.create_tensor([DECODE_T_PAD, O_LORA_TOTAL], dtype=pl.FP8E4M3FN)
    o_b_scale_backing = pl.create_tensor(
        [1, DECODE_T_PAD * (O_LORA_TOTAL // MX_GROUP)],
        dtype=pl.FP8E8M0,
    )
    for quant_b_idx in pl.spmd(O_LORA_TOTAL // B_K_TILE, name_hint="o_proj_b_quant_mx"):
        quant_b_k0 = quant_b_idx * B_K_TILE
        quant_b_src = pl.load(o_r, [0, quant_b_k0], [MM_M_TILE, B_K_TILE])
        quant_b_data, quant_b_scale = pl.quant_mx(quant_b_src, group_axis=1)
        o_b_mx = pl.store(quant_b_data, [0, quant_b_k0], o_b_mx)
        quant_b_scale_block = quant_b_idx * MM_M_TILE * (B_K_TILE // MX_GROUP)
        quant_b_scale_flat = pl.reshape(quant_b_scale, [1, MM_M_TILE * (B_K_TILE // MX_GROUP)])
        o_b_scale_backing = pl.store(
            quant_b_scale_flat,
            [0, quant_b_scale_block],
            o_b_scale_backing,
        )

    o_b_scale = pl.tensor.view(
        o_b_scale_backing,
        [DECODE_T_PAD, O_LORA_TOTAL // MX_GROUP],
        layout=pl.MX_A_ZZ,
    )
    for proj_b_idx in pl.spmd(D // B_N_TILE, name_hint="o_proj_b_mx"):
        proj_b_n0 = proj_b_idx * B_N_TILE
        proj_b_lhs0 = pl.load(o_b_mx, [0, 0], [MM_M_TILE, B_K_TILE])
        proj_b_lhs_scale0 = pl.load(o_b_scale, [0, 0], [MM_M_TILE, B_K_TILE // MX_GROUP])
        proj_b_rhs0 = pl.load(wo_b, [0, proj_b_n0], [B_K_TILE, B_N_TILE])
        proj_b_rhs_scale0 = pl.load(
            wo_b_scale,
            [0, proj_b_n0],
            [B_K_TILE // MX_GROUP, B_N_TILE],
        )
        proj_b_acc = pl.matmul_mx(
            proj_b_lhs0,
            proj_b_lhs_scale0,
            proj_b_rhs0,
            proj_b_rhs_scale0,
        )
        for proj_b_k0 in pl.range(B_K_TILE, O_LORA_TOTAL, B_K_TILE):
            proj_b_ks = proj_b_k0 // MX_GROUP
            proj_b_lhs = pl.load(o_b_mx, [0, proj_b_k0], [MM_M_TILE, B_K_TILE])
            proj_b_lhs_scale = pl.load(
                o_b_scale,
                [0, proj_b_ks],
                [MM_M_TILE, B_K_TILE // MX_GROUP],
            )
            proj_b_rhs = pl.load(wo_b, [proj_b_k0, proj_b_n0], [B_K_TILE, B_N_TILE])
            proj_b_rhs_scale = pl.load(
                wo_b_scale,
                [proj_b_ks, proj_b_n0],
                [B_K_TILE // MX_GROUP, B_N_TILE],
            )
            proj_b_acc = pl.matmul_mx_acc(
                proj_b_acc,
                proj_b_lhs,
                proj_b_lhs_scale,
                proj_b_rhs,
                proj_b_rhs_scale,
            )
        proj_b_out = pl.cast(proj_b_acc, target_type=pl.BF16, mode="rint")
        attn_out = pl.store(
            proj_b_out[0:DECODE_T, 0:B_N_TILE],
            [0, proj_b_n0],
            attn_out,
        )

    return attn_out


@pl.jit.inline
def prefill_output_proj_mx(
    o_packed: pl.Tensor[[O_GROUPS * PREFILL_T, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[O_GROUPS * (O_GROUP_IN // MX_GROUP), O_LORA], pl.FP8E8M0, pl.MX_B_NN],
    wo_b: pl.Tensor[[O_LORA_TOTAL, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[O_LORA_TOTAL // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    attn_out: pl.Tensor[[PREFILL_T, D], pl.BF16],
):
    """Project prefill attention output through two MXFP8 Cube matmuls."""
    o_a_mx = pl.create_tensor([O_GROUPS * PREFILL_T, O_GROUP_IN], dtype=pl.FP8E4M3FN)
    o_a_scale_backing = pl.create_tensor(
        [1, O_GROUPS * PREFILL_T * (O_GROUP_IN // MX_GROUP)],
        dtype=pl.FP8E8M0,
    )
    for quant_a_idx in pl.spmd(
        O_GROUPS * (PREFILL_T // MM_M_TILE) * (O_GROUP_IN // A_K_TILE),
        name_hint="o_proj_a_quant_mx",
    ):
        quant_a_group = quant_a_idx // ((PREFILL_T // MM_M_TILE) * (O_GROUP_IN // A_K_TILE))
        quant_a_local = quant_a_idx % ((PREFILL_T // MM_M_TILE) * (O_GROUP_IN // A_K_TILE))
        quant_a_t0 = (quant_a_local // (O_GROUP_IN // A_K_TILE)) * MM_M_TILE
        quant_a_k0 = (quant_a_local % (O_GROUP_IN // A_K_TILE)) * A_K_TILE
        quant_a_row = quant_a_group * PREFILL_T + quant_a_t0
        quant_a_src = pl.load(o_packed, [quant_a_row, quant_a_k0], [MM_M_TILE, A_K_TILE])
        quant_a_fp32 = pl.cast(quant_a_src, target_type=pl.FP32, mode="none")
        quant_a_data, quant_a_scale = pl.quant_mx(quant_a_fp32, group_axis=1)
        o_a_mx = pl.store(quant_a_data, [quant_a_row, quant_a_k0], o_a_mx)
        quant_a_scale_block = quant_a_idx * MM_M_TILE * (A_K_TILE // MX_GROUP)
        quant_a_scale_flat = pl.reshape(quant_a_scale, [1, MM_M_TILE * (A_K_TILE // MX_GROUP)])
        o_a_scale_backing = pl.store(
            quant_a_scale_flat,
            [0, quant_a_scale_block],
            o_a_scale_backing,
        )

    o_a_scale = pl.tensor.view(
        o_a_scale_backing,
        [O_GROUPS * PREFILL_T, O_GROUP_IN // MX_GROUP],
        layout=pl.MX_A_ZZ,
    )
    wo_a_flat = pl.reshape(wo_a, [O_GROUPS * O_GROUP_IN, O_LORA])
    o_r = pl.create_tensor([PREFILL_T, O_LORA_TOTAL], dtype=pl.FP32)
    with pl.spmd(O_LORA // A_N_TILE, name_hint="o_proj_a_mx") as _proj_a_tid:
        proj_a_n0 = pl.tile.get_block_idx() * A_N_TILE
        for proj_a_group in pl.range(O_GROUPS):
            for proj_a_t0 in pl.pipeline(0, PREFILL_T, MM_M_TILE, stage=2):
                proj_a_row = proj_a_group * PREFILL_T + proj_a_t0
                proj_a_scale_row = proj_a_group * (O_GROUP_IN // MX_GROUP)
                proj_a_lhs0 = pl.load(o_a_mx, [proj_a_row, 0], [MM_M_TILE, A_K_TILE])
                proj_a_lhs_scale0 = pl.load(
                    o_a_scale,
                    [proj_a_row, 0],
                    [MM_M_TILE, A_K_TILE // MX_GROUP],
                )
                proj_a_rhs0 = pl.load(
                    wo_a_flat,
                    [proj_a_group * O_GROUP_IN, proj_a_n0],
                    [A_K_TILE, A_N_TILE],
                )
                proj_a_rhs_scale0 = pl.load(
                    wo_a_scale,
                    [proj_a_scale_row, proj_a_n0],
                    [A_K_TILE // MX_GROUP, A_N_TILE],
                )
                proj_a_acc = pl.matmul_mx(
                    proj_a_lhs0,
                    proj_a_lhs_scale0,
                    proj_a_rhs0,
                    proj_a_rhs_scale0,
                )
                for proj_a_k0 in pl.range(A_K_TILE, O_GROUP_IN, A_K_TILE):
                    proj_a_ks = proj_a_k0 // MX_GROUP
                    proj_a_lhs = pl.load(o_a_mx, [proj_a_row, proj_a_k0], [MM_M_TILE, A_K_TILE])
                    proj_a_lhs_scale = pl.load(
                        o_a_scale,
                        [proj_a_row, proj_a_ks],
                        [MM_M_TILE, A_K_TILE // MX_GROUP],
                    )
                    proj_a_rhs = pl.load(
                        wo_a_flat,
                        [proj_a_group * O_GROUP_IN + proj_a_k0, proj_a_n0],
                        [A_K_TILE, A_N_TILE],
                    )
                    proj_a_rhs_scale = pl.load(
                        wo_a_scale,
                        [proj_a_scale_row + proj_a_ks, proj_a_n0],
                        [A_K_TILE // MX_GROUP, A_N_TILE],
                    )
                    proj_a_acc = pl.matmul_mx_acc(
                        proj_a_acc,
                        proj_a_lhs,
                        proj_a_lhs_scale,
                        proj_a_rhs,
                        proj_a_rhs_scale,
                    )
                o_r = pl.store(
                    proj_a_acc,
                    [proj_a_t0, proj_a_group * O_LORA + proj_a_n0],
                    o_r,
                )

    o_b_mx = pl.create_tensor([PREFILL_T, O_LORA_TOTAL], dtype=pl.FP8E4M3FN)
    o_b_scale_backing = pl.create_tensor(
        [1, PREFILL_T * (O_LORA_TOTAL // MX_GROUP)],
        dtype=pl.FP8E8M0,
    )
    for quant_b_idx in pl.spmd(
        (PREFILL_T // MM_M_TILE) * (O_LORA_TOTAL // B_K_TILE),
        name_hint="o_proj_b_quant_mx",
    ):
        quant_b_t0 = (quant_b_idx // (O_LORA_TOTAL // B_K_TILE)) * MM_M_TILE
        quant_b_k0 = (quant_b_idx % (O_LORA_TOTAL // B_K_TILE)) * B_K_TILE
        quant_b_src = pl.load(o_r, [quant_b_t0, quant_b_k0], [MM_M_TILE, B_K_TILE])
        quant_b_data, quant_b_scale = pl.quant_mx(quant_b_src, group_axis=1)
        o_b_mx = pl.store(quant_b_data, [quant_b_t0, quant_b_k0], o_b_mx)
        quant_b_scale_block = quant_b_idx * MM_M_TILE * (B_K_TILE // MX_GROUP)
        quant_b_scale_flat = pl.reshape(quant_b_scale, [1, MM_M_TILE * (B_K_TILE // MX_GROUP)])
        o_b_scale_backing = pl.store(
            quant_b_scale_flat,
            [0, quant_b_scale_block],
            o_b_scale_backing,
        )

    o_b_scale = pl.tensor.view(
        o_b_scale_backing,
        [PREFILL_T, O_LORA_TOTAL // MX_GROUP],
        layout=pl.MX_A_ZZ,
    )
    with pl.spmd(D // B_N_TILE, name_hint="o_proj_b_mx") as _proj_b_tid:
        proj_b_n0 = pl.tile.get_block_idx() * B_N_TILE
        for proj_b_t0 in pl.pipeline(0, PREFILL_T, MM_M_TILE, stage=2):
            proj_b_lhs0 = pl.load(o_b_mx, [proj_b_t0, 0], [MM_M_TILE, B_K_TILE])
            proj_b_lhs_scale0 = pl.load(
                o_b_scale,
                [proj_b_t0, 0],
                [MM_M_TILE, B_K_TILE // MX_GROUP],
            )
            proj_b_rhs0 = pl.load(wo_b, [0, proj_b_n0], [B_K_TILE, B_N_TILE])
            proj_b_rhs_scale0 = pl.load(
                wo_b_scale,
                [0, proj_b_n0],
                [B_K_TILE // MX_GROUP, B_N_TILE],
            )
            proj_b_acc = pl.matmul_mx(
                proj_b_lhs0,
                proj_b_lhs_scale0,
                proj_b_rhs0,
                proj_b_rhs_scale0,
            )
            for proj_b_k0 in pl.range(B_K_TILE, O_LORA_TOTAL, B_K_TILE):
                proj_b_ks = proj_b_k0 // MX_GROUP
                proj_b_lhs = pl.load(o_b_mx, [proj_b_t0, proj_b_k0], [MM_M_TILE, B_K_TILE])
                proj_b_lhs_scale = pl.load(
                    o_b_scale,
                    [proj_b_t0, proj_b_ks],
                    [MM_M_TILE, B_K_TILE // MX_GROUP],
                )
                proj_b_rhs = pl.load(wo_b, [proj_b_k0, proj_b_n0], [B_K_TILE, B_N_TILE])
                proj_b_rhs_scale = pl.load(
                    wo_b_scale,
                    [proj_b_ks, proj_b_n0],
                    [B_K_TILE // MX_GROUP, B_N_TILE],
                )
                proj_b_acc = pl.matmul_mx_acc(
                    proj_b_acc,
                    proj_b_lhs,
                    proj_b_lhs_scale,
                    proj_b_rhs,
                    proj_b_rhs_scale,
                )
            proj_b_out = pl.cast(proj_b_acc, target_type=pl.BF16, mode="rint")
            attn_out = pl.store(proj_b_out, [proj_b_t0, proj_b_n0], attn_out)

    return attn_out


def golden_output_proj_mx(o_model, wo_a, wo_a_scale, wo_b, wo_b_scale):
    """Torch reference for the shared grouped MXFP8 output projection."""
    import torch

    from mx_utils import decode_e8m0_codes, host_quant_mxfp8, matmul_mx_golden

    token_count = o_model.shape[0]
    wo_a_scale_logical = decode_e8m0_codes(wo_a_scale, side="b")
    wo_b_scale_logical = decode_e8m0_codes(wo_b_scale, side="b")
    group_results = []
    for group in range(O_GROUPS):
        group_input, group_input_scale = host_quant_mxfp8(
            o_model[:, group, :],
            return_e8m0=True,
        )
        group_scale0 = group * (O_GROUP_IN // MX_GROUP)
        group_scale1 = group_scale0 + O_GROUP_IN // MX_GROUP
        group_result = matmul_mx_golden(
            group_input,
            group_input_scale,
            wo_a[group],
            wo_a_scale_logical[group_scale0:group_scale1],
        )
        group_results.append(group_result)

    o_r = torch.stack(group_results, dim=1).reshape(token_count, O_LORA_TOTAL)
    o_r_mx, o_r_scale = host_quant_mxfp8(o_r, return_e8m0=True)
    return matmul_mx_golden(o_r_mx, o_r_scale, wo_b, wo_b_scale_logical)
