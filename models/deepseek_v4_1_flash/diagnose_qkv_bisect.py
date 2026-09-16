#!/usr/bin/env python3
"""Bisect fused k_qkv: find the first hop whose snapshot misses golden."""

from __future__ import annotations

import json
import os
import sys

sys.argv = [sys.argv[0], "--tp", "1", *sys.argv[1:]]

import pypto.language as pl
import torch

from models.deepseek_v4_1_flash.decode_c2a_full import (
    ACT_COL_TILE,
    ACT_MAX,
    D,
    HEAD_DIM,
    K_SCALE_GROUPS,
    K_TILE,
    LOCAL_H,
    M_TILE,
    MX_GROUP,
    N_TILE_WIDE,
    Q_LORA,
    RING_HEAP_4G,
    ROPE_DIM,
    T_MAX,
    TOKEN_TILE,
    _apply_rope_tail_hd,
    _apply_rope_tail_kv,
    _mxfp8_weight,
    _mxfp8_wkv,
    _mxfp8_wq_a,
    _mxfp8_wq_b_fp32,
    _prepare_rope_interleaved,
    _rms_norm_rows,
    _rms_norm_rows_fp32,
)
from models.deepseek_v4_1_flash.golden import rms_norm, rope_interleave
from models.deepseek_v4_1_flash.quantization import mxfp8_linear
from models.deepseek_v4_1_flash.test_c2a_kernel_accuracy import ATOL, RTOL, _stats
from models.deepseek_v4_1_flash.test_c2a_kernels import (
    N_TOK,
    ScalarSpec,
    TensorSpec,
    _ones,
    _randn_bf16,
    _zeros,
    run,
)

Q_WIDTH = LOCAL_H * HEAD_DIM
K_GROUPS = Q_LORA // MX_GROUP
N_TILES = Q_WIDTH // N_TILE_WIDE


def _fmt(st: dict) -> str:
    return (
        f"bad={st['bad']}/{st['n']} ({st['pct']:.2f}%) "
        f"max={st['max']:.5g} mean={st['mean']:.5g} cos={st['cos']:.6f}"
    )


def _make_gather_from_act(width_c: int, hint: str):
    @pl.jit.inline
    def _gather(
        src: pl.Tensor[[T_MAX, ACT_MAX], pl.FP32],
        dst: pl.Tensor[[T_MAX, width_c], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        n_copy = (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE
        for blk in pl.spmd(n_copy, name_hint=hint):
            t0 = blk * TOKEN_TILE
            rows = pl.min(TOKEN_TILE, num_tokens - t0)
            if t0 < num_tokens:
                for c0 in pl.range(0, width_c, ACT_COL_TILE):
                    cols = pl.min(ACT_COL_TILE, width_c - c0)
                    dst[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE] = pl.set_validshape(
                        src[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE],
                        rows,
                        cols,
                    )

    return _gather


_gather_wqa = _make_gather_from_act(Q_LORA, "bisect_wqa")
_gather_qr_fp32 = _make_gather_from_act(Q_LORA, "bisect_qr_fp32")
_gather_q_fp32 = _make_gather_from_act(Q_WIDTH, "bisect_q_fp32")
_gather_q_full = _make_gather_from_act(Q_WIDTH, "split_full")


def _make_copy_cols(width_c: int, hint: str):
    @pl.jit.inline
    def _copy_cols(
        src: pl.Tensor[[T_MAX, width_c], pl.FP32],
        dst: pl.Tensor[[T_MAX, width_c], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        n_copy = (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE
        for blk in pl.spmd(n_copy, name_hint=hint):
            t0 = blk * TOKEN_TILE
            rows = pl.min(TOKEN_TILE, num_tokens - t0)
            if t0 < num_tokens:
                for c0 in pl.range(0, width_c, ACT_COL_TILE):
                    cols = pl.min(ACT_COL_TILE, width_c - c0)
                    dst[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE] = pl.set_validshape(
                        src[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE],
                        rows,
                        cols,
                    )

    return _copy_cols


_copy_q_lora_act = _make_copy_cols(Q_LORA, "split_act_out")


@pl.jit
def k_qkv_hops(
    x: pl.Tensor[[T_MAX, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, Q_WIDTH], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, Q_WIDTH], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_MAX, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_MAX, ROPE_DIM // 2], pl.FP32],
    wqa: pl.Out[pl.Tensor[[T_MAX, Q_LORA], pl.FP32]],
    qr_fp32: pl.Out[pl.Tensor[[T_MAX, Q_LORA], pl.FP32]],
    qr: pl.Out[pl.Tensor[[T_MAX, Q_LORA], pl.BF16]],
    q_fp32: pl.Out[pl.Tensor[[T_MAX, Q_WIDTH], pl.FP32]],
    q_cast: pl.Out[pl.Tensor[[T_MAX, Q_WIDTH], pl.BF16]],
    q_unflat: pl.Out[pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16]],
    q: pl.Out[pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16]],
    window_kv: pl.Out[pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Production fused order, with an Out snapshot after each major hop."""
    act_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.BF16)
    out_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    n_copy = (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE
    for blk in pl.spmd(n_copy, name_hint="bisect_x_pad"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            act_pad[t0 : t0 + TOKEN_TILE, 0:D] = pl.set_validshape(
                x[t0 : t0 + TOKEN_TILE, :], rows, D
            )

    _mxfp8_wq_a(act_pad, wq_a, wq_a_scale, out_pad, num_tokens)
    _gather_wqa(out_pad, wqa, num_tokens)

    qr_fp32_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    _rms_norm_rows_fp32(
        out_pad, q_norm_weight, qr_fp32_pad, num_tokens, pl.cast(Q_LORA, pl.INT32)
    )
    _gather_qr_fp32(qr_fp32_pad, qr_fp32, num_tokens)

    for blk in pl.spmd(n_copy, name_hint="bisect_qr"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            qr[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(
                pl.cast(
                    qr_fp32_pad[t0 : t0 + TOKEN_TILE, 0:Q_LORA],
                    target_type=pl.BF16,
                    mode="rint",
                ),
                rows,
                Q_LORA,
            )

    q_out_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    _mxfp8_wq_b_fp32(qr_fp32_pad, wq_b, wq_b_scale, q_out_pad, num_tokens)
    _gather_q_fp32(q_out_pad, q_fp32, num_tokens)

    kv_act_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.BF16)
    kv_out_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    for blk in pl.spmd(n_copy, name_hint="bisect_x_pad2"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            kv_act_pad[t0 : t0 + TOKEN_TILE, 0:D] = pl.set_validshape(
                x[t0 : t0 + TOKEN_TILE, :], rows, D
            )
    _mxfp8_wkv(kv_act_pad, wkv, wkv_scale, kv_out_pad, num_tokens)
    kv_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.BF16)
    _rms_norm_rows(kv_out_pad, kv_norm_weight, kv_pad, num_tokens, pl.cast(HEAD_DIM, pl.INT32))
    for blk in pl.spmd(n_copy, name_hint="bisect_kv"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            window_kv[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(
                kv_pad[t0 : t0 + TOKEN_TILE, 0:HEAD_DIM], rows, HEAD_DIM
            )

    for blk in pl.spmd(n_copy, name_hint="bisect_q_cast"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            for c0 in pl.range(0, Q_WIDTH, ACT_COL_TILE):
                cols = pl.min(ACT_COL_TILE, Q_WIDTH - c0)
                q_cast[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE] = pl.set_validshape(
                    pl.cast(
                        q_fp32[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE],
                        target_type=pl.BF16,
                        mode="rint",
                    ),
                    rows,
                    cols,
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="bisect_unflat"):
        for t in pl.range(num_tokens):
            for h in pl.range(LOCAL_H):
                q[t : t + 1, h : h + 1, :] = pl.reshape(
                    q_cast[t : t + 1, h * HEAD_DIM : (h + 1) * HEAD_DIM],
                    [1, 1, HEAD_DIM],
                )
                q_unflat[t : t + 1, h : h + 1, :] = q[t : t + 1, h : h + 1, :]

    cos_il = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.FP32)
    _prepare_rope_interleaved(
        rope_cos, rope_sin, cos_il, sin_signed, num_tokens, pl.cast(0, pl.INT32)
    )
    q_heads = pl.create_tensor([T_MAX * LOCAL_H, HEAD_DIM], dtype=pl.BF16)
    cos_heads = pl.create_tensor([T_MAX * LOCAL_H, ROPE_DIM], dtype=pl.FP32)
    sin_heads = pl.create_tensor([T_MAX * LOCAL_H, ROPE_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="bisect_rope_expand"):
        for t in pl.range(num_tokens):
            for h in pl.range(LOCAL_H):
                dst = t * LOCAL_H + h
                q_heads[dst : dst + 1, :] = pl.reshape(q[t : t + 1, h : h + 1, :], [1, HEAD_DIM])
                cos_heads[dst : dst + 1, :] = cos_il[t : t + 1, :]
                sin_heads[dst : dst + 1, :] = sin_signed[t : t + 1, :]
    _apply_rope_tail_hd(q_heads, cos_heads, sin_heads, pl.cast(num_tokens * LOCAL_H, pl.INT32))
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="bisect_rope_scatter"):
        for t in pl.range(num_tokens):
            for h in pl.range(LOCAL_H):
                dst = t * LOCAL_H + h
                q[t : t + 1, h : h + 1, :] = pl.reshape(q_heads[dst : dst + 1, :], [1, 1, HEAD_DIM])
    _apply_rope_tail_kv(window_kv, cos_il, sin_signed, num_tokens)
    return wqa, qr_fp32, qr, q_fp32, q_cast, q_unflat, q, window_kv


@pl.jit
def k_wqb_split(
    qr_in: pl.Tensor[[T_MAX, Q_LORA], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, Q_WIDTH], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, Q_WIDTH], pl.FP8E8M0, pl.MX_B_NN],
    act_copy: pl.Out[pl.Tensor[[T_MAX, Q_LORA], pl.FP32]],
    q_k0: pl.Out[pl.Tensor[[T_MAX, Q_WIDTH], pl.FP32]],
    q_full: pl.Out[pl.Tensor[[T_MAX, Q_WIDTH], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Split WQ_B: copy → first K-tile matmul_mx → full K pipeline."""
    act_fp32 = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    act_mx = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP8E4M3FN)
    scale_back = pl.create_tensor([1, T_MAX * K_GROUPS], dtype=pl.FP8E8M0)
    x_scale_mx = pl.tensor.view(scale_back, [T_MAX, K_GROUPS], layout=pl.MX_A_ZZ)
    n_copy = (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE
    for blk in pl.spmd(n_copy, name_hint="split_copy"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            for c0 in pl.range(0, Q_LORA, ACT_COL_TILE):
                cols = pl.min(ACT_COL_TILE, Q_LORA - c0)
                act_fp32[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE] = pl.set_validshape(
                    qr_in[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE],
                    rows,
                    cols,
                )
    _copy_q_lora_act(qr_in, act_copy, num_tokens)

    m_tiles = (num_tokens + M_TILE - 1) // M_TILE
    k_tiles = Q_LORA // K_TILE
    for blk in pl.spmd(m_tiles * k_tiles, name_hint="split_quant"):
        unit = pl.tile.get_block_idx()
        mt = unit // k_tiles
        kt = unit - mt * k_tiles
        t0 = mt * M_TILE
        k0 = kt * K_TILE
        if t0 < num_tokens:
            x_q, x_q_scale = pl.quant_mx(
                pl.load(act_fp32, [t0, k0], [M_TILE, K_TILE]), group_axis=1
            )
            act_mx = pl.store(x_q, [t0, k0], act_mx)
            scale_back = pl.store(
                pl.reshape(x_q_scale, [1, M_TILE * K_SCALE_GROUPS]),
                [0, t0 * K_GROUPS + kt * M_TILE * K_SCALE_GROUPS],
                scale_back,
            )

    # First K tile only: one matmul_mx, no acc loop.
    for blk in pl.spmd(N_TILES, name_hint="split_k0"):
        n0 = blk * N_TILE_WIDE
        t_k0 = pl.cast(0, pl.INT32)
        if t_k0 < num_tokens:
            xs0 = pl.load(act_mx, [t_k0, 0], [M_TILE, K_TILE])
            xs_scale0 = pl.load(x_scale_mx, [t_k0, 0], [M_TILE, K_SCALE_GROUPS])
            w0 = pl.load(wq_b, [0, n0], [K_TILE, N_TILE_WIDE])
            w0_scale = pl.load(wq_b_scale, [0, n0], [K_SCALE_GROUPS, N_TILE_WIDE])
            acc = pl.matmul_mx(xs0, xs_scale0, w0, w0_scale)
            q_k0 = pl.store(acc, [t_k0, n0], q_k0)

    q_full_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    _mxfp8_wq_b_fp32(act_fp32, wq_b, wq_b_scale, q_full_pad, num_tokens)
    _gather_q_full(q_full_pad, q_full, num_tokens)
    return act_copy, q_k0, q_full


def _run(fn, specs, golden_fn, names, device):
    captured = {}

    def cap(name):
        def _fn(actual, expected, **_kwargs):
            captured[name] = (
                actual.detach().cpu().clone(),
                expected.detach().cpu().clone(),
            )
            return True, ""

        _fn.__name__ = f"cap_{name}"
        return _fn

    run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        compare_fn={n: cap(n) for n in names},
        config=dict(platform="a5", device_id=device, ring_heap=RING_HEAP_4G),
    )
    return captured


def _host_partial_k0(qr_fp32, wq_b, wq_b_scale):
    return mxfp8_linear(
        qr_fp32[:, :K_TILE],
        wq_b[:K_TILE],
        wq_b_scale[: K_TILE // MX_GROUP],
        output_dtype=torch.float32,
    )


def main() -> int:
    device = int(os.environ.get("TASK_DEVICE", "2").split(",")[0])
    torch.manual_seed(0)
    wq_a, wq_a_s = _mxfp8_weight(D, Q_LORA)
    wq_b, wq_b_s = _mxfp8_weight(Q_LORA, Q_WIDTH)
    wkv, wkv_s = _mxfp8_weight(D, HEAD_DIM)
    stash = {}

    def golden_hops(tensors):
        n = int(tensors["num_tokens"])
        x = tensors["x"][:n]
        wqa = mxfp8_linear(x, tensors["wq_a"], tensors["wq_a_scale"], output_dtype=torch.float32)
        qr_fp32 = rms_norm(wqa, tensors["q_norm_weight"])
        q_fp32 = mxfp8_linear(
            qr_fp32, tensors["wq_b"], tensors["wq_b_scale"], output_dtype=torch.float32
        )
        q_bf16 = q_fp32.to(torch.bfloat16)
        q_heads = q_bf16.unflatten(-1, (LOCAL_H, HEAD_DIM))
        rd = tensors["rope_cos"].shape[-1] * 2
        q_rope = torch.cat(
            (
                q_heads[..., :-rd],
                rope_interleave(
                    q_heads[..., -rd:], tensors["rope_cos"][:n], tensors["rope_sin"][:n]
                ),
            ),
            dim=-1,
        )
        kv = mxfp8_linear(
            x, tensors["wkv"], tensors["wkv_scale"], output_dtype=torch.float32
        )
        kv_n = rms_norm(kv, tensors["kv_norm_weight"]).to(torch.bfloat16)
        kv_heads = kv_n.unsqueeze(1)
        kv_rope = torch.cat(
            (
                kv_heads[..., :-rd],
                rope_interleave(
                    kv_heads[..., -rd:], tensors["rope_cos"][:n], tensors["rope_sin"][:n]
                ),
            ),
            dim=-1,
        ).squeeze(1)
        for key in ("wqa", "qr_fp32", "qr", "q_fp32", "q_cast", "q_unflat", "q", "window_kv"):
            tensors[key].zero_()
        tensors["wqa"][:n] = wqa
        tensors["qr_fp32"][:n] = qr_fp32
        tensors["qr"][:n] = qr_fp32.to(torch.bfloat16)
        tensors["q_fp32"][:n] = q_fp32
        tensors["q_cast"][:n] = q_bf16
        tensors["q_unflat"][:n] = q_heads
        tensors["q"][:n] = q_rope
        tensors["window_kv"][:n] = kv_rope
        stash["qr_fp32"] = qr_fp32.clone()
        stash["q_fp32"] = q_fp32.clone()
        stash["wq_b"] = tensors["wq_b"].clone()
        stash["wq_b_scale"] = tensors["wq_b_scale"].clone()

    hop_specs = [
        TensorSpec("x", [T_MAX, D], torch.bfloat16, init_value=_randn_bf16(T_MAX, D)),
        TensorSpec("wq_a", [D, Q_LORA], torch.float8_e4m3fn, init_value=lambda: wq_a),
        TensorSpec("wq_a_scale", [D // 32, Q_LORA], wq_a_s.dtype, init_value=lambda: wq_a_s),
        TensorSpec("q_norm_weight", [Q_LORA], torch.bfloat16, init_value=_ones(Q_LORA, dtype=torch.bfloat16)),
        TensorSpec("wq_b", [Q_LORA, Q_WIDTH], torch.float8_e4m3fn, init_value=lambda: wq_b),
        TensorSpec("wq_b_scale", [Q_LORA // 32, Q_WIDTH], wq_b_s.dtype, init_value=lambda: wq_b_s),
        TensorSpec("wkv", [D, HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wkv),
        TensorSpec("wkv_scale", [D // 32, HEAD_DIM], wkv_s.dtype, init_value=lambda: wkv_s),
        TensorSpec("kv_norm_weight", [HEAD_DIM], torch.bfloat16, init_value=_ones(HEAD_DIM, dtype=torch.bfloat16)),
        TensorSpec("rope_cos", [T_MAX, ROPE_DIM // 2], torch.float32, init_value=_ones(T_MAX, ROPE_DIM // 2)),
        TensorSpec("rope_sin", [T_MAX, ROPE_DIM // 2], torch.float32, init_value=_zeros(T_MAX, ROPE_DIM // 2)),
        TensorSpec("wqa", [T_MAX, Q_LORA], torch.float32),
        TensorSpec("qr_fp32", [T_MAX, Q_LORA], torch.float32),
        TensorSpec("qr", [T_MAX, Q_LORA], torch.bfloat16),
        TensorSpec("q_fp32", [T_MAX, Q_WIDTH], torch.float32),
        TensorSpec("q_cast", [T_MAX, Q_WIDTH], torch.bfloat16),
        TensorSpec("q_unflat", [T_MAX, LOCAL_H, HEAD_DIM], torch.bfloat16),
        TensorSpec("q", [T_MAX, LOCAL_H, HEAD_DIM], torch.bfloat16),
        TensorSpec("window_kv", [T_MAX, HEAD_DIM], torch.bfloat16),
        ScalarSpec("num_tokens", torch.int32, N_TOK),
    ]
    hop_names = ["wqa", "qr_fp32", "qr", "q_fp32", "q_cast", "q_unflat", "q", "window_kv"]
    print("======== PHASE 1 fused hops ========")
    captured = _run(k_qkv_hops, hop_specs, golden_hops, hop_names, device)

    rows = {}
    first = None
    for name in hop_names:
        a, e = captured[name]
        a, e = a[:N_TOK], e[:N_TOK]
        st = _stats(a, e)
        rows[name] = st
        mark = ""
        if first is None and st["bad"] > 0:
            first = name
            mark = "  <-- FIRST FAIL"
        print(f"{name:12s} vs host  {_fmt(st)}{mark}")

    if first == "q_fp32" or (first in ("q_cast", "q_unflat", "q") and rows["qr_fp32"]["bad"] == 0):
        dev_qr = captured["qr_fp32"][0][:N_TOK].float()
        gold_qr = captured["qr_fp32"][1][:N_TOK].float()
        dev_q = captured["q_fp32"][0][:N_TOK].float()
        host_from_dev = mxfp8_linear(
            dev_qr, stash["wq_b"], stash["wq_b_scale"], output_dtype=torch.float32
        )
        print("-------- WQ_B isolation --------")
        print(f"{'host(WQ_B, dev QR) vs gold Q':28s} {_fmt(_stats(host_from_dev, stash['q_fp32']))}")
        print(f"{'dev Q vs host(WQ_B, dev QR)':28s} {_fmt(_stats(dev_q, host_from_dev))}")
        print(f"{'dev QR vs gold QR':28s} {_fmt(_stats(dev_qr, gold_qr))}")

    print(json.dumps({"first_fail": first, "hops": {k: {kk: v[kk] for kk in ("bad", "n", "pct", "max")} for k, v in rows.items()}}, indent=2))

    print("======== PHASE 2 WQ_B split (copy / first-K / full) ========")
    torch.manual_seed(0)
    qr_host = stash["qr_fp32"]
    qr_pad = torch.zeros(T_MAX, Q_LORA, dtype=torch.float32)
    qr_pad[:N_TOK] = qr_host

    def golden_split(tensors):
        n = int(tensors["num_tokens"])
        qr = tensors["qr_in"][:n]
        tensors["act_copy"].zero_()
        tensors["q_k0"].zero_()
        tensors["q_full"].zero_()
        tensors["act_copy"][:n] = qr
        tensors["q_k0"][:n] = _host_partial_k0(qr, tensors["wq_b"], tensors["wq_b_scale"])
        tensors["q_full"][:n] = mxfp8_linear(
            qr, tensors["wq_b"], tensors["wq_b_scale"], output_dtype=torch.float32
        )

    split_specs = [
        TensorSpec("qr_in", [T_MAX, Q_LORA], torch.float32, init_value=lambda: qr_pad.clone()),
        TensorSpec("wq_b", [Q_LORA, Q_WIDTH], torch.float8_e4m3fn, init_value=lambda: stash["wq_b"]),
        TensorSpec("wq_b_scale", [Q_LORA // 32, Q_WIDTH], stash["wq_b_scale"].dtype, init_value=lambda: stash["wq_b_scale"]),
        TensorSpec("act_copy", [T_MAX, Q_LORA], torch.float32),
        TensorSpec("q_k0", [T_MAX, Q_WIDTH], torch.float32),
        TensorSpec("q_full", [T_MAX, Q_WIDTH], torch.float32),
        ScalarSpec("num_tokens", torch.int32, N_TOK),
    ]
    split_names = ["act_copy", "q_k0", "q_full"]
    split_cap = _run(k_wqb_split, split_specs, golden_split, split_names, device)
    split_first = None
    for name in split_names:
        a, e = split_cap[name]
        st = _stats(a[:N_TOK], e[:N_TOK])
        mark = ""
        if split_first is None and st["bad"] > 0:
            split_first = name
            mark = "  <-- FIRST FAIL"
        print(f"{name:12s} vs host  {_fmt(st)}{mark}")
        for t in range(N_TOK):
            print(f"  token{t} {_fmt(_stats(a[t : t + 1], e[t : t + 1]))}")

    print(json.dumps({"phase1_first": first, "phase2_first": split_first}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
