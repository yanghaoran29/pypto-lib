# Copyright (c) PyPTO Contributors.
"""Pipeline probe bisect: compare each stage output vs golden intermediates."""

from __future__ import annotations

import os
import sys

import torch
import pypto.language as pl

# Force full pipeline when imported (stages read C2A_STAGE from argv at import).
# Match decode_c2a_full.py / test_c2a_kernels.py: single-card bring-up is TP=1
# (LOCAL_H=64).  Without this, config defaults to TP=4 and the probe diagnoses
# a different graph than the failing full entry.
if "--stage" not in sys.argv:
    sys.argv.extend(["--stage", "full"])
if not any(arg == "--tp" or arg.startswith("--tp=") for arg in sys.argv):
    sys.argv.extend(["--tp", "1"])

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.attention_common import _project_output
from models.deepseek_v4_1_flash.decode_c2a_full import (
    ACT_MAX,
    CMP_PACKED,
    D,
    HEAD_DIM,
    IDX_PACKED,
    INDEX_DIM,
    INDEX_TOPK,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    RING_HEAP_4G,
    ROPE_DIM,
    SCORE_LEN,
    T_MAX,
    TOKEN_TILE,
    WS_CMP_ROWS,
    WS_IDX_ROWS,
    WS_ORI_ROWS,
    _import_harness,
    _mxfp4_dequant_cmp_block,
    _mxfp8_wq_a,
    _stage_compressor_ratio2,
    _stage_paged_indexer,
    _stage_project_output,
    _stage_publish_compressed_index,
    _stage_publish_window,
    _stage_qkv_proj_rope,
    _stage_sparse_attn_merge,
    build_tensor_specs,
)
from models.deepseek_v4_1_flash.golden import (
    compressor_ratio2_paged,
    paged_indexer,
    paged_sparse_attention,
    publish_cache,
    publish_index_key,
    qkv_proj_rope,
    rope_interleave,
)
from models.deepseek_v4_1_flash.quantization import (
    MX_GROUP,
    _quantize_mxfp8_activation,
    decode_e8m0,
    dequantize_mxfp4_cache,
    dequantize_mxfp8_cache,
    quantize_mxfp4_cache,
    quantize_mxfp8_cache,
    unpack_mx_b_scale,
)

T_DYN = C.T_DYN
ORI_BLOCKS_DYN = C.ORI_BLOCKS_DYN
CMP_BLOCKS_DYN = C.CMP_BLOCKS_DYN
INDEX_BLOCKS_DYN = C.INDEX_BLOCKS_DYN
B_DYN = C.B_DYN
TABLE_DYN = C.TABLE_DYN
WINDOW_CACHE_GROUP = C.WINDOW_CACHE_GROUP
COMPRESSED_CACHE_GROUP = C.COMPRESSED_CACHE_GROUP
INDEX_CACHE_GROUP = C.INDEX_CACHE_GROUP
MAX_BATCH_PER_DP = C.MAX_BATCH_PER_DP
STATE_HEADS = C.STATE_HEADS
INDEX_H = C.INDEX_H


# Probe helpers live inline in decode_c2a_probe_entry (typed copies).


@pl.jit
def decode_c2a_probe_entry(
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
    # Probes (device stage outputs)
    probe_wqa_raw: pl.Out[pl.Tensor[[T_MAX, Q_LORA], pl.FP32]],
    probe_qr: pl.Out[pl.Tensor[[T_MAX, Q_LORA], pl.BF16]],
    probe_q: pl.Out[pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16]],
    probe_window_kv: pl.Out[pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16]],
    probe_window_bf16: pl.Out[pl.Tensor[[WS_ORI_ROWS, HEAD_DIM], pl.BF16]],
    probe_latent: pl.Out[pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16]],
    probe_cmp_bf16: pl.Out[pl.Tensor[[WS_CMP_ROWS, HEAD_DIM], pl.BF16]],
    probe_idx_bf16: pl.Out[pl.Tensor[[WS_IDX_ROWS, INDEX_DIM], pl.BF16]],
    probe_attended: pl.Out[pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16]],
    probe_att_project: pl.Out[
        pl.Tensor[[T_MAX, LOCAL_O_GROUPS * O_GROUP_IN], pl.BF16]
    ],
    probe_o_latent: pl.Out[pl.Tensor[[T_MAX, LOCAL_O_WIDTH], pl.BF16]],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
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

    t_dim = pl.tensor.dim(x, 0)
    n_tok = pl.min(num_tokens, t_dim)

    # Raw WQ_A output, before RMSNorm, to distinguish Cube accumulation from
    # RMS reduction/rsqrt differences in the Q quantization discontinuity.
    raw_act_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.BF16)
    raw_out_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    for blk in pl.spmd((n_tok + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="c2a_probe_raw_pad"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, n_tok - t0)
        if t0 < n_tok:
            raw_act_pad[t0 : t0 + TOKEN_TILE, 0:D] = pl.set_validshape(
                x[t0 : t0 + TOKEN_TILE, :], rows, D
            )
    _mxfp8_wq_a(raw_act_pad, wq_a, wq_a_scale, raw_out_pad, n_tok)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_probe_wqa_raw"):
        for t in pl.range(n_tok):
            probe_wqa_raw[t : t + 1, :] = raw_out_pad[t : t + 1, 0:Q_LORA]

    qr = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([T_MAX, LOCAL_H, HEAD_DIM], dtype=pl.BF16)
    window_kv = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.BF16)
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
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_probe_qkv"):
        for t in pl.range(n_tok):
            probe_qr[t : t + 1, :] = qr[t : t + 1, :]
            probe_window_kv[t : t + 1, :] = window_kv[t : t + 1, :]
            probe_q[t : t + 1, :, :] = q[t : t + 1, :, :]

    window_bf16 = pl.create_tensor([WS_ORI_ROWS, HEAD_DIM], dtype=pl.BF16)
    window_stage_indices = pl.create_tensor([T_MAX, 128], dtype=pl.INT32)
    cmp_bf16 = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.BF16)
    idx_bf16 = pl.create_tensor([T_MAX, INDEX_DIM], dtype=pl.BF16)
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
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_probe_win"):
        for r in pl.range(WS_ORI_ROWS):
            probe_window_bf16[r : r + 1, :] = window_bf16[r : r + 1, :]

    latent = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.BF16)
    publish_mask = pl.create_tensor([T_MAX], dtype=pl.INT32)
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
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_probe_lat"):
        for t in pl.range(n_tok):
            probe_latent[t : t + 1, :] = latent[t : t + 1, :]

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
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_probe_pub"):
        for r in pl.range(n_tok):
            probe_cmp_bf16[r : r + 1, :] = cmp_bf16[r : r + 1, :]
        for r in pl.range(n_tok):
            probe_idx_bf16[r : r + 1, :] = idx_bf16[r : r + 1, :]


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

    # Expand compressed UINT8 into a separate workspace; keep cmp_bf16 publish-only.
    cmp_exp = pl.create_tensor([WS_CMP_ROWS, HEAD_DIM], dtype=pl.BF16)
    cmp_block_packed = pl.create_tensor([128, CMP_PACKED], dtype=pl.UINT8)
    cmp_block_scale = pl.create_tensor(
        [HEAD_DIM // COMPRESSED_CACHE_GROUP, 128], dtype=pl.BF16
    )
    table_w = pl.tensor.dim(index_block_table, 1)
    for t in pl.range(n_tok):
        req = pl.cast(pl.read(request_ids, [t]), pl.INDEX)
        clen = pl.read(compressed_lens, [t])
        visible = pl.min(clen, SCORE_LEN)
        n_blocks = (visible + 127) // 128
        for b in pl.range(n_blocks):
            if b < table_w:
                block_id = pl.cast(pl.read(index_block_table, [req, b]), pl.INDEX)
                for _exp in pl.spmd(1, name_hint="c2a_probe_exp_cmp"):
                    _mxfp4_dequant_cmp_block(
                        compressed_cache,
                        compressed_cache_scale,
                        cmp_block_packed,
                        cmp_block_scale,
                        cmp_exp,
                        block_id,
                        t * SCORE_LEN + b * 128,
                    )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_probe_cmp_overlay"):
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
                                mapped = pl.cast(pl.read(index_block_table, [req, b]), pl.INDEX)
                                if mapped == pblock:
                                    dst = t * SCORE_LEN + b * 128 + prow
                                    cmp_exp[dst : dst + 1, :] = cmp_bf16[pt : pt + 1, :]

    attn_cmp_indices = pl.create_tensor([T_MAX, INDEX_TOPK], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_probe_cmp_stage_indices"):
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
                            mapped = pl.cast(pl.read(index_block_table, [req, b]), pl.INDEX)
                            if mapped == pblock:
                                staged = t * SCORE_LEN + b * 128 + prow
                                pl.write(attn_cmp_indices, [t, k], pl.cast(staged, pl.INT32))

    attended = pl.create_tensor([T_MAX, LOCAL_H, HEAD_DIM], dtype=pl.BF16)
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
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c2a_probe_att"):
        for t in pl.range(n_tok):
            probe_attended[t : t + 1, :, :] = attended[t : t + 1, :, :]

    output_partial = pl.create_tensor([T_MAX, D], dtype=pl.FP32)
    _stage_project_output(
        attended,
        rope_cos,
        rope_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        probe_att_project,
        probe_o_latent,
        output_partial,
        n_tok,
    )
    n_copy = (n_tok + TOKEN_TILE - 1) // TOKEN_TILE
    for blk in pl.spmd(n_copy, name_hint="c2a_probe_out"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, n_tok - t0)
        if t0 < n_tok:
            output[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(
                pl.cast(
                    output_partial[t0 : t0 + TOKEN_TILE, :],
                    target_type=pl.BF16,
                    mode="rint",
                ),
                rows,
                D,
            )


def _flat_cache(cache_3or4d: torch.Tensor, rows: int, dim: int) -> torch.Tensor:
    flat = cache_3or4d.reshape(-1, dim)
    out = torch.zeros(rows, dim, dtype=flat.dtype)
    n = min(rows, flat.shape[0])
    out[:n] = flat[:n]
    return out


PROBE_REF: dict[str, torch.Tensor] = {}


def golden_probe_tensors(tensors: dict) -> None:
    """Fill probes using golden steps; BF16 mirrors match device (pre-MXFP4 qdq)."""
    t = int(tensors["num_tokens"])
    # Keep the exact fixture inputs used after compilation.  The harness fills
    # random inputs only after the JIT has consumed RNG state, so reconstructing
    # them in a separate Python process does not reproduce the board run.
    for name in (
        "x",
        "wq_a",
        "wq_a_scale",
        "q_norm_weight",
        "wq_b",
        "wq_b_scale",
        "wkv",
        "wkv_scale",
        "kv_norm_weight",
    ):
        value = tensors[name]
        PROBE_REF[name] = value[:t].detach().cpu() if name == "x" else value.detach().cpu()

    activation, activation_scale = _quantize_mxfp8_activation(tensors["x"][:t])
    weight_scale = decode_e8m0(unpack_mx_b_scale(tensors["wq_a_scale"]))
    weight_groups = tensors["wq_a"].float().unflatten(0, (-1, MX_GROUP))
    wqa_partials = torch.einsum(
        "...gk,gkn->...gn", activation.float(), weight_groups
    )
    wqa_raw = (
        wqa_partials * activation_scale.unsqueeze(-1) * weight_scale
    ).sum(dim=-2)
    tensors["probe_wqa_raw"].zero_()
    tensors["probe_wqa_raw"][:t] = wqa_raw
    PROBE_REF["wqa_raw"] = wqa_raw.detach().cpu()
    query, window_kv, qr = qkv_proj_rope(
        tensors["x"][:t],
        tensors["wq_a"],
        tensors["wq_a_scale"],
        tensors["q_norm_weight"],
        tensors["wq_b"],
        tensors["wq_b_scale"],
        tensors["wkv"],
        tensors["wkv_scale"],
        tensors["kv_norm_weight"],
        tensors["rope_cos"][:t],
        tensors["rope_sin"][:t],
    )
    PROBE_REF["qr"] = qr.detach().cpu()
    PROBE_REF["query"] = query.detach().cpu()
    PROBE_REF["window_kv"] = window_kv.detach().cpu()
    tensors["probe_qr"].zero_()
    tensors["probe_q"].zero_()
    tensors["probe_window_kv"].zero_()
    tensors["probe_qr"][:t] = qr.to(tensors["probe_qr"].dtype)
    tensors["probe_q"][:t] = query.to(tensors["probe_q"].dtype)
    tensors["probe_window_kv"][:t] = window_kv.to(tensors["probe_window_kv"].dtype)

    # Window BF16 mirror: query-local staging, matching _stage_publish_window.
    win_bf16 = torch.zeros(WS_ORI_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    win_val = dequantize_mxfp8_cache(
        tensors["window_cache"], tensors["window_cache_scale"]
    ).to(torch.bfloat16)
    win_flat = win_val.reshape(-1, HEAD_DIM)
    idx = tensors["window_indices"][:t]
    window_stage_indices = torch.full_like(idx, -1)
    for ti in range(t):
        for k in range(idx.shape[1]):
            r = int(idx[ti, k])
            if 0 <= r < win_flat.shape[0]:
                staged = ti * 128 + k
                win_bf16[staged] = win_flat[r]
                window_stage_indices[ti, k] = staged
    slots = tensors["window_slots"][:t]
    for ti in range(t):
        for k in range(idx.shape[1]):
            r = int(idx[ti, k])
            if r >= 0:
                for pt in range(t):
                    if int(slots[pt]) == r:
                        win_bf16[ti * 128 + k] = window_kv[pt].to(torch.bfloat16)
    tensors["probe_window_bf16"].copy_(win_bf16)

    state = tensors["compressor_state"].clone()
    latent, publish_mask = compressor_ratio2_paged(
        tensors["x"][:t],
        tensors["position_ids"][:t],
        tensors["compressor_state_rows"][:t],
        state,
        tensors["compressor_wkv"],
        tensors["compressor_wgate"],
        tensors["compressor_norm_weight"],
    )
    tensors["compressor_state"].copy_(state)
    tensors["probe_latent"].zero_()
    tensors["probe_latent"][:t] = latent.to(tensors["probe_latent"].dtype)

    # Compressed / index BF16 mirrors (device writes BF16; UINT8 pack still stub).
    cos_c = tensors["compressed_rope_cos"][:t]
    sin_c = tensors["compressed_rope_sin"][:t]
    rd = cos_c.shape[-1] * 2
    rotated = torch.cat(
        (latent[..., :-rd], rope_interleave(latent[..., -rd:], cos_c, sin_c)), dim=-1
    )
    cmp_bf16 = torch.zeros(WS_CMP_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    for ti in range(t):
        if bool(publish_mask[ti]) and int(tensors["compressed_slots"][ti]) >= 0:
            cmp_bf16[ti] = rotated[ti].to(torch.bfloat16)
    tensors["probe_cmp_bf16"].copy_(cmp_bf16)

    idx_bf16 = torch.zeros(WS_IDX_ROWS, INDEX_DIM, dtype=torch.bfloat16)
    idx_work = torch.zeros(1, 128, 1, INDEX_DIM, dtype=torch.bfloat16)
    publish_index_key(
        latent,
        publish_mask,
        tensors["index_wk"],
        tensors["index_norm_weight"],
        cos_c,
        sin_c,
        tensors["compressed_slots"][:t],
        idx_work,
    )
    idx_flat = idx_work.reshape(128, INDEX_DIM)
    for ti in range(t):
        slot = int(tensors["compressed_slots"][ti])
        if bool(publish_mask[ti]) and 0 <= slot < idx_flat.shape[0]:
            idx_bf16[ti] = idx_flat[slot].to(torch.bfloat16)
    tensors["probe_idx_bf16"].copy_(idx_bf16)

    # Indexer against BF16 index mirror (device expands stub UINT8 then restores publish).
    _scores, topk = paged_indexer(
        tensors["x"][:t],
        qr,
        tensors["request_ids"][:t],
        idx_work,
        tensors["index_block_table"],
        tensors["compressed_lens"][:t],
        tensors["index_wq_b"],
        tensors["index_wq_b_scale"],
        tensors["index_weights_proj"],
        tensors["rope_cos"][:t],
        tensors["rope_sin"][:t],
        candidates=None,
        topk=INDEX_TOPK,
    )
    tensors["topk_indices"].fill_(-1)
    tensors["topk_indices"][:t] = topk.to(torch.int32)

    # Device-style attended: query-local staging plus exact same-dispatch overlay.
    cmp_stage = torch.zeros(WS_CMP_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    attn_cmp_indices = torch.full_like(topk, -1)
    block_table = tensors["index_block_table"]
    request_ids = tensors["request_ids"][:t]
    compressed_lens = tensors["compressed_lens"][:t]
    compressed_slots = tensors["compressed_slots"][:t]
    for ti in range(t):
        req = int(request_ids[ti])
        visible = min(int(compressed_lens[ti]), SCORE_LEN)
        n_blocks = (visible + 127) // 128
        for pt in range(t):
            slot = int(compressed_slots[pt])
            if bool(publish_mask[pt]) and int(request_ids[pt]) == req and slot >= 0:
                pblock, prow = divmod(slot, 128)
                for b in range(min(n_blocks, block_table.shape[1])):
                    if int(block_table[req, b]) == pblock:
                        cmp_stage[ti * SCORE_LEN + b * 128 + prow] = rotated[pt].to(torch.bfloat16)
        for k in range(topk.shape[1]):
            physical = int(topk[ti, k])
            if physical >= 0:
                pblock, prow = divmod(physical, 128)
                for b in range(min(n_blocks, block_table.shape[1])):
                    if int(block_table[req, b]) == pblock:
                        attn_cmp_indices[ti, k] = ti * SCORE_LEN + b * 128 + prow
    win_4d = win_bf16.view(1, WS_ORI_ROWS, 1, HEAD_DIM)
    cmp_4d = cmp_stage.view(1, WS_CMP_ROWS, 1, HEAD_DIM)
    attended_dev = paged_sparse_attention(
        query,
        win_4d,
        window_stage_indices,
        cmp_4d,
        attn_cmp_indices,
        tensors["attn_sink"],
    )
    tensors["probe_attended"].zero_()
    tensors["probe_attended"][:t] = attended_dev.to(tensors["probe_attended"].dtype)

    # oproj from device-style attended (isolates oproj from MXFP4 qdq gap).
    out = _project_output(
        attended_dev,
        tensors["rope_cos"][:t],
        tensors["rope_sin"][:t],
        tensors["wo_a"],
        tensors["wo_b"],
        tensors["wo_b_scale"],
    )
    tensors["output"].zero_()
    tensors["output"][:t] = out.to(tensors["output"].dtype)
    rope_dim = tensors["rope_cos"].shape[-1] * 2
    attended_tail = rope_interleave(
        attended_dev[..., -rope_dim:],
        tensors["rope_cos"][:t],
        tensors["rope_sin"][:t],
        inverse=True,
    )
    attended_projected = torch.cat(
        (attended_dev[..., :-rope_dim], attended_tail), dim=-1
    ).flatten(-2)
    grouped = attended_projected.unflatten(
        -1, (LOCAL_O_GROUPS, O_GROUP_IN)
    )
    o_latent = torch.einsum("tgd,grd->tgr", grouped, tensors["wo_a"])
    tensors["probe_att_project"].zero_()
    tensors["probe_att_project"][:t] = attended_projected.to(torch.bfloat16)
    tensors["probe_o_latent"].zero_()
    tensors["probe_o_latent"][:t] = o_latent.to(torch.bfloat16).flatten(-2)

    # Reference-only: full golden qdq path (not compared on device probes).
    updated_window = win_val.clone()
    publish_cache(updated_window, window_kv, tensors["window_slots"][:t])
    win_payload, win_scale = quantize_mxfp8_cache(updated_window)
    quantized_window = dequantize_mxfp8_cache(win_payload, win_scale).to(query.dtype)
    cmp_page = torch.zeros(1, 128, 1, HEAD_DIM, dtype=torch.bfloat16)
    for ti in range(t):
        if bool(publish_mask[ti]) and int(tensors["compressed_slots"][ti]) >= 0:
            s = int(tensors["compressed_slots"][ti])
            cmp_page[0, s, 0] = rotated[ti].to(torch.bfloat16)
    cmp_payload, cmp_scale = quantize_mxfp4_cache(
        cmp_page, group_size=COMPRESSED_CACHE_GROUP, scale_format="bf16"
    )
    quantized_cmp = dequantize_mxfp4_cache(
        cmp_payload, cmp_scale, group_size=COMPRESSED_CACHE_GROUP, scale_format="bf16"
    ).to(query.dtype)
    attended_gold = paged_sparse_attention(
        query,
        quantized_window,
        tensors["window_indices"][:t],
        quantized_cmp,
        topk,
        tensors["attn_sink"],
    )
    out_qdq = _project_output(
        attended_gold,
        tensors["rope_cos"][:t],
        tensors["rope_sin"][:t],
        tensors["wo_a"],
        tensors["wo_b"],
        tensors["wo_b_scale"],
    )
    PROBE_REF["output_qdq"] = out_qdq.detach().cpu()
    PROBE_REF["attended_qdq"] = attended_gold.detach().cpu()
    PROBE_REF["output_devstyle"] = out.detach().cpu()
    PROBE_REF["rope_cos"] = tensors["rope_cos"][:t].detach().cpu()
    PROBE_REF["rope_sin"] = tensors["rope_sin"][:t].detach().cpu()
    PROBE_REF["wo_a"] = tensors["wo_a"].detach().cpu()
    PROBE_REF["wo_b"] = tensors["wo_b"].detach().cpu()
    PROBE_REF["wo_b_scale"] = tensors["wo_b_scale"].detach().cpu()


def build_probe_specs(num_tokens: int = 2, n_blocks: int = 1):
    ScalarSpec, TensorSpec, _, _ = _import_harness()
    base = build_tensor_specs(num_tokens=num_tokens, n_blocks=n_blocks)
    # Insert probes before topk_indices / output / num_tokens.
    probes = [
        TensorSpec("probe_wqa_raw", [T_MAX, Q_LORA], torch.float32),
        TensorSpec("probe_qr", [T_MAX, Q_LORA], torch.bfloat16),
        TensorSpec("probe_q", [T_MAX, LOCAL_H, HEAD_DIM], torch.bfloat16),
        TensorSpec("probe_window_kv", [T_MAX, HEAD_DIM], torch.bfloat16),
        TensorSpec("probe_window_bf16", [WS_ORI_ROWS, HEAD_DIM], torch.bfloat16),
        TensorSpec("probe_latent", [T_MAX, HEAD_DIM], torch.bfloat16),
        TensorSpec("probe_cmp_bf16", [WS_CMP_ROWS, HEAD_DIM], torch.bfloat16),
        TensorSpec("probe_idx_bf16", [WS_IDX_ROWS, INDEX_DIM], torch.bfloat16),
        TensorSpec("probe_attended", [T_MAX, LOCAL_H, HEAD_DIM], torch.bfloat16),
        TensorSpec(
            "probe_att_project",
            [T_MAX, LOCAL_O_GROUPS * O_GROUP_IN],
            torch.bfloat16,
        ),
        TensorSpec("probe_o_latent", [T_MAX, LOCAL_O_WIDTH], torch.bfloat16),
    ]
    # base ends with topk, output, num_tokens
    return base[:-3] + probes + base[-3:]


def _stats(name: str, actual: torch.Tensor, expected: torch.Tensor, atol=0.02, rtol=1 / 32):
    a = actual.detach().cpu().float().reshape(-1)
    e = expected.detach().cpu().float().reshape(-1)
    if a.numel() != e.numel():
        return f"{name:18s} SHAPE_MISMATCH act={tuple(actual.shape)} exp={tuple(expected.shape)}"
    if actual.dtype in (torch.int32, torch.int64):
        neq = (actual.cpu().to(torch.int64).reshape(-1) != expected.cpu().to(torch.int64).reshape(-1)).sum().item()
        return f"{name:18s} mismatch={neq}/{a.numel()} ({100 * neq / max(a.numel(), 1):.3f}%)"
    diff = (a - e).abs()
    tol = atol + rtol * e.abs()
    bad = (diff > tol).sum().item()
    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), e.unsqueeze(0)))
    return (
        f"{name:18s} bad={bad}/{a.numel()} ({100 * bad / max(a.numel(), 1):.2f}%) "
        f"max|d|={float(diff.max()):.5g} mean|d|={float(diff.mean()):.5g} cos={cos:.6f}"
    )


def main() -> int:
    device = int(os.environ.get("TASK_DEVICE", "2").split(",")[0])
    _, _, _, run = _import_harness()
    specs = build_probe_specs()
    dump: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def skip(a, e, **k):
        return True, ""

    def capture(name):
        def _fn(a, e, **k):
            dump[name] = (a.detach().cpu().clone(), e.detach().cpu().clone())
            return True, ""

        _fn.__name__ = f"cap_{name}"
        return _fn

    names = [
        "probe_wqa_raw",
        "probe_qr",
        "probe_q",
        "probe_window_kv",
        "probe_window_bf16",
        "probe_latent",
        "probe_cmp_bf16",
        "probe_idx_bf16",
        "topk_indices",
        "probe_attended",
        "probe_att_project",
        "probe_o_latent",
        "output",
        "compressor_state",
    ]
    compare_fn = {n: capture(n) for n in names}
    for n in (
        "window_cache",
        "window_cache_scale",
        "compressed_cache",
        "compressed_cache_scale",
        "index_cache",
        "index_cache_scale",
    ):
        compare_fn[n] = skip

    print(f"[PROBE] device={device} ring_heap={RING_HEAP_4G}")
    result = run(
        fn=decode_c2a_probe_entry,
        specs=specs,
        golden_fn=golden_probe_tensors,
        rtol=2e-2,
        atol=2e-2,
        compare_fn=compare_fn,
        config=dict(platform="a5", device_id=device, ring_heap=RING_HEAP_4G),
    )
    print(f"[PROBE] run passed={result.passed} (compare always capture)")
    torch.save(dump, "/tmp/c2a_probe_intermediates.pt")
    torch.save(PROBE_REF, "/tmp/c2a_probe_refs.pt")

    order = [
        ("0.qkv/wqa_raw", "probe_wqa_raw"),
        ("1.qkv/qr", "probe_qr"),
        ("1.qkv/q", "probe_q"),
        ("1.qkv/window_kv", "probe_window_kv"),
        ("2.window/bf16", "probe_window_bf16"),
        ("3.compress/latent", "probe_latent"),
        ("3.compress/state", "compressor_state"),
        ("3.publish/cmp_bf16", "probe_cmp_bf16"),
        ("3.publish/idx_bf16", "probe_idx_bf16"),
        ("4.indexer/topk", "topk_indices"),
        ("5.attn/attended", "probe_attended"),
        ("6.oproj/att_project", "probe_att_project"),
        ("6.oproj/o_latent", "probe_o_latent"),
        ("6.oproj/output", "output"),
    ]
    print("\n======== STAGE BISECT (device vs golden intermediates) ========")
    first_bad = None
    for label, key in order:
        if key not in dump:
            print(f"{label:22s} MISSING")
            continue
        a, e = dump[key]
        # Focus live tokens / published rows for large workspaces.
        if key in ("probe_wqa_raw", "probe_qr", "probe_window_kv", "probe_latent"):
            a, e = a[:2], e[:2]
        elif key == "probe_q":
            a, e = a[:2], e[:2]
        elif key in ("probe_attended", "probe_att_project", "probe_o_latent"):
            a, e = a[:2], e[:2]
        elif key == "output":
            a, e = a[:2], e[:2]
        elif key == "topk_indices":
            a, e = a[:2], e[:2]
        elif key in ("probe_window_bf16", "probe_cmp_bf16", "probe_idx_bf16"):
            # Compare only first 2 physical rows (fixture publishes into 0/1).
            a, e = a[:2], e[:2]
        line = _stats(label, a, e)
        print(line)
        bad = "bad=" in line and not line.split("bad=")[1].startswith("0/")
        mismatch = "mismatch=" in line and not line.split("mismatch=")[1].startswith("0/")
        if (bad or mismatch) and first_bad is None:
            first_bad = label

    if "output" in dump and "output_qdq" in PROBE_REF:
        a = dump["output"][0][:2].float()
        e = PROBE_REF["output_qdq"][:2].float()
        print("\n======== vs FULL golden (MXFP qdq path) ========")
        print(_stats("output_vs_qdq", a, e))
    if "output" in dump and "probe_attended" in dump:
        output_from_device_attended = _project_output(
            dump["probe_attended"][0][:2],
            PROBE_REF["rope_cos"],
            PROBE_REF["rope_sin"],
            PROBE_REF["wo_a"],
            PROBE_REF["wo_b"],
            PROBE_REF["wo_b_scale"],
        )
        print("\n======== OPROJ isolation (device output vs Torch(device attended)) ========")
        print(_stats("output_from_dev_att", dump["output"][0][:2], output_from_device_attended))
    print("\n======== VERDICT ========")
    if first_bad is None:
        print("All probed stage outputs within atol=0.02 rtol=1/32 (or exact for topk).")
    else:
        print(f"First divergence: {first_bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
