#!/usr/bin/env python3
"""Snapshot fused Q writeback hops: WQ_B FP32 → BF16 cast → unflat → RoPE."""

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
    LOCAL_H,
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
from models.deepseek_v4_1_flash.test_c2a_kernel_accuracy import _stats
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


def _fmt(st: dict) -> str:
    return (
        f"bad={st['bad']}/{st['n']} ({st['pct']:.2f}%) "
        f"max={st['max']:.5g} mean={st['mean']:.5g} cos={st['cos']:.6f}"
    )


@pl.jit
def k_qkv_wb(
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
    qr: pl.Out[pl.Tensor[[T_MAX, Q_LORA], pl.BF16]],
    q_fp32: pl.Out[pl.Tensor[[T_MAX, Q_WIDTH], pl.FP32]],
    q_cast: pl.Out[pl.Tensor[[T_MAX, Q_WIDTH], pl.BF16]],
    q_unflat: pl.Out[pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16]],
    q: pl.Out[pl.Tensor[[T_MAX, LOCAL_H, HEAD_DIM], pl.BF16]],
    window_kv: pl.Out[pl.Tensor[[T_MAX, HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    act_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.BF16)
    out_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    n_copy = (num_tokens + TOKEN_TILE - 1) // TOKEN_TILE
    for blk in pl.spmd(n_copy, name_hint="wb_x_pad"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            act_pad[t0 : t0 + TOKEN_TILE, 0:D] = pl.set_validshape(
                x[t0 : t0 + TOKEN_TILE, :], rows, D
            )

    _mxfp8_wq_a(act_pad, wq_a, wq_a_scale, out_pad, num_tokens)
    qr_fp32_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    _rms_norm_rows_fp32(
        out_pad, q_norm_weight, qr_fp32_pad, num_tokens, pl.cast(Q_LORA, pl.INT32)
    )
    for blk in pl.spmd(n_copy, name_hint="wb_qr"):
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

    for blk in pl.spmd(n_copy, name_hint="wb_snap_fp32"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            for c0 in pl.range(0, Q_WIDTH, ACT_COL_TILE):
                cols = pl.min(ACT_COL_TILE, Q_WIDTH - c0)
                q_fp32[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE] = pl.set_validshape(
                    q_out_pad[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE],
                    rows,
                    cols,
                )

    q_flat = pl.create_tensor([T_MAX, Q_WIDTH], dtype=pl.BF16)
    for blk in pl.spmd(n_copy, name_hint="wb_q_cast"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            for c0 in pl.range(0, Q_WIDTH, ACT_COL_TILE):
                cols = pl.min(ACT_COL_TILE, Q_WIDTH - c0)
                q_flat[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE] = pl.set_validshape(
                    pl.cast(
                        q_out_pad[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE],
                        target_type=pl.BF16,
                        mode="rint",
                    ),
                    rows,
                    cols,
                )

    for blk in pl.spmd(n_copy, name_hint="wb_snap_cast"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            for c0 in pl.range(0, Q_WIDTH, ACT_COL_TILE):
                cols = pl.min(ACT_COL_TILE, Q_WIDTH - c0)
                q_cast[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE] = pl.set_validshape(
                    q_flat[t0 : t0 + TOKEN_TILE, c0 : c0 + ACT_COL_TILE],
                    rows,
                    cols,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="wb_q_unflat"):
        for t in pl.range(num_tokens):
            for h in pl.range(LOCAL_H):
                q[t : t + 1, h : h + 1, :] = pl.reshape(
                    q_flat[t : t + 1, h * HEAD_DIM : (h + 1) * HEAD_DIM],
                    [1, 1, HEAD_DIM],
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="wb_snap_unflat"):
        for t in pl.range(num_tokens):
            q_unflat[t : t + 1, :, :] = q[t : t + 1, :, :]

    kv_act_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.BF16)
    kv_out_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.FP32)
    for blk in pl.spmd(n_copy, name_hint="wb_x_pad2"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            kv_act_pad[t0 : t0 + TOKEN_TILE, 0:D] = pl.set_validshape(
                x[t0 : t0 + TOKEN_TILE, :], rows, D
            )
    _mxfp8_wkv(kv_act_pad, wkv, wkv_scale, kv_out_pad, num_tokens)
    kv_pad = pl.create_tensor([T_MAX, ACT_MAX], dtype=pl.BF16)
    _rms_norm_rows(kv_out_pad, kv_norm_weight, kv_pad, num_tokens, pl.cast(HEAD_DIM, pl.INT32))
    for blk in pl.spmd(n_copy, name_hint="wb_kv"):
        t0 = blk * TOKEN_TILE
        rows = pl.min(TOKEN_TILE, num_tokens - t0)
        if t0 < num_tokens:
            window_kv[t0 : t0 + TOKEN_TILE, :] = pl.set_validshape(
                kv_pad[t0 : t0 + TOKEN_TILE, 0:HEAD_DIM], rows, HEAD_DIM
            )

    cos_il = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.FP32)
    _prepare_rope_interleaved(rope_cos, rope_sin, cos_il, sin_signed, num_tokens, pl.cast(0, pl.INT32))
    q_heads = pl.create_tensor([T_MAX * LOCAL_H, HEAD_DIM], dtype=pl.BF16)
    cos_heads = pl.create_tensor([T_MAX * LOCAL_H, ROPE_DIM], dtype=pl.FP32)
    sin_heads = pl.create_tensor([T_MAX * LOCAL_H, ROPE_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="wb_rope_expand"):
        for t in pl.range(num_tokens):
            for h in pl.range(LOCAL_H):
                dst = t * LOCAL_H + h
                q_heads[dst : dst + 1, :] = pl.reshape(q[t : t + 1, h : h + 1, :], [1, HEAD_DIM])
                cos_heads[dst : dst + 1, :] = cos_il[t : t + 1, :]
                sin_heads[dst : dst + 1, :] = sin_signed[t : t + 1, :]
    _apply_rope_tail_hd(q_heads, cos_heads, sin_heads, pl.cast(num_tokens * LOCAL_H, pl.INT32))
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="wb_rope_scatter"):
        for t in pl.range(num_tokens):
            for h in pl.range(LOCAL_H):
                dst = t * LOCAL_H + h
                q[t : t + 1, h : h + 1, :] = pl.reshape(q_heads[dst : dst + 1, :], [1, 1, HEAD_DIM])
    _apply_rope_tail_kv(window_kv, cos_il, sin_signed, num_tokens)
    return qr, q_fp32, q_cast, q_unflat, q, window_kv


def main() -> int:
    device = int(os.environ.get("TASK_DEVICE", "2").split(",")[0])
    torch.manual_seed(0)
    wq_a, wq_a_s = _mxfp8_weight(D, Q_LORA)
    wq_b, wq_b_s = _mxfp8_weight(Q_LORA, Q_WIDTH)
    wkv, wkv_s = _mxfp8_weight(D, HEAD_DIM)
    stash = {}
    captured = {}

    def golden(tensors):
        n = int(tensors["num_tokens"])
        x = tensors["x"][:n]
        qr_fp32 = rms_norm(
            mxfp8_linear(x, tensors["wq_a"], tensors["wq_a_scale"], output_dtype=torch.float32),
            tensors["q_norm_weight"],
        )
        q_fp32 = mxfp8_linear(
            qr_fp32, tensors["wq_b"], tensors["wq_b_scale"], output_dtype=torch.float32
        )
        q_bf16 = q_fp32.to(torch.bfloat16)
        q_heads = q_bf16.unflatten(-1, (LOCAL_H, HEAD_DIM))
        rd = tensors["rope_cos"].shape[-1] * 2
        q_rope = torch.cat(
            (
                q_heads[..., :-rd],
                rope_interleave(q_heads[..., -rd:], tensors["rope_cos"][:n], tensors["rope_sin"][:n]),
            ),
            dim=-1,
        )
        tensors["qr"].zero_()
        tensors["q_fp32"].zero_()
        tensors["q_cast"].zero_()
        tensors["q_unflat"].zero_()
        tensors["q"].zero_()
        tensors["qr"][:n] = qr_fp32.to(torch.bfloat16)
        tensors["q_fp32"][:n] = q_fp32
        tensors["q_cast"][:n] = q_bf16
        tensors["q_unflat"][:n] = q_heads
        tensors["q"][:n] = q_rope
        stash["gold"] = {k: tensors[k][:n].clone() for k in ("qr", "q_fp32", "q_cast", "q_unflat", "q")}

    def cap(name):
        def _fn(actual, expected, **_kwargs):
            captured[name] = (actual.detach().cpu().clone(), expected.detach().cpu().clone())
            return True, ""

        _fn.__name__ = f"cap_{name}"
        return _fn

    specs = [
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
        TensorSpec("qr", [T_MAX, Q_LORA], torch.bfloat16),
        TensorSpec("q_fp32", [T_MAX, Q_WIDTH], torch.float32),
        TensorSpec("q_cast", [T_MAX, Q_WIDTH], torch.bfloat16),
        TensorSpec("q_unflat", [T_MAX, LOCAL_H, HEAD_DIM], torch.bfloat16),
        TensorSpec("q", [T_MAX, LOCAL_H, HEAD_DIM], torch.bfloat16),
        TensorSpec("window_kv", [T_MAX, HEAD_DIM], torch.bfloat16),
        ScalarSpec("num_tokens", torch.int32, N_TOK),
    ]
    names = ["qr", "q_fp32", "q_cast", "q_unflat", "q"]
    run(
        fn=k_qkv_wb,
        specs=specs,
        golden_fn=golden,
        compare_fn={n: cap(n) for n in names} | {"window_kv": cap("window_kv")},
        config=dict(platform="a5", device_id=device, ring_heap=RING_HEAP_4G),
    )

    print("======== FUSED Q WRITEBACK HOPS ========")
    rows = {}
    for name in names:
        a, e = captured[name]
        a, e = a[:N_TOK], e[:N_TOK]
        st = _stats(a, e)
        rows[f"{name} vs host"] = st
        print(f"{name:12s} vs host  {_fmt(st)}")

    # hops relative to the previous device snapshot
    q_fp32_a = captured["q_fp32"][0][:N_TOK].float()
    q_cast_a = captured["q_cast"][0][:N_TOK].float()
    q_unflat_a = captured["q_unflat"][0][:N_TOK].float().reshape(N_TOK, Q_WIDTH)
    q_a = captured["q"][0][:N_TOK].float().reshape(N_TOK, Q_WIDTH)
    hops = {
        "cast vs rint(fp32 snap)": _stats(q_cast_a, q_fp32_a.to(torch.bfloat16)),
        "unflat vs cast": _stats(q_unflat_a, q_cast_a),
        "rope vs unflat": _stats(q_a, q_unflat_a),
    }
    print("-------- device hops --------")
    for name, st in hops.items():
        rows[name] = st
        print(f"{name:28s} {_fmt(st)}")

    print(json.dumps({k: {kk: v[kk] for kk in ("bad", "n", "pct", "max", "cos")} for k, v in {**rows}.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
