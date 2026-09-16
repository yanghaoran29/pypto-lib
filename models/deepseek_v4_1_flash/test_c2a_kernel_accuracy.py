# Copyright (c) PyPTO Contributors.
"""Per-kernel numeric accuracy vs Torch golden (single-card TP=1)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

if not any(arg == "--tp" or arg.startswith("--tp=") for arg in sys.argv):
    sys.argv = [sys.argv[0], "--tp", "1", *sys.argv[1:]]

import pypto.language as pl
import torch

from models.deepseek_v4_1_flash.attention_common import _project_output
from models.deepseek_v4_1_flash.decode_c2a_full import (
    ACT_MAX,
    D,
    HEAD_DIM,
    INDEX_DIM,
    INDEX_H,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    RING_HEAP_4G,
    ROPE_DIM,
    T_MAX,
    _mxfp8_wo_b_compact,
    _mxfp8_wq_b_fp32,
    _mxfp8_weight,
    _rms_norm_rows_fp32,
)
from models.deepseek_v4_1_flash.golden import compressor_ratio2_paged, paged_sparse_attention, qkv_proj_rope, rms_norm
from models.deepseek_v4_1_flash.quantization import mxfp8_linear
from models.deepseek_v4_1_flash.test_c2a_kernels import (
    N_TOK,
    ScalarSpec,
    TensorSpec,
    _build_cases,
    k_attn,
    k_bf16_linear,
    k_compressor,
    k_fp32_linear,
    k_mx_idx,
    k_mx_wkv,
    k_mx_wo,
    k_mx_wq_a,
    k_mx_wq_b,
    k_oproj,
    k_qkv,
    k_rms,
    k_rope_hd,
    k_rope_idx,
    k_rope_kv,
    k_rope_prep,
    run,
)


ATOL = 0.02
RTOL = 1.0 / 32
OUT_PATH = Path("/tmp/c2a_kernel_accuracy.json")


@pl.jit
def k_rms_fp32(
    x: pl.Tensor[[T_MAX, ACT_MAX], pl.FP32],
    weight: pl.Tensor[[Q_LORA], pl.BF16],
    y: pl.Out[pl.Tensor[[T_MAX, ACT_MAX], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    _rms_norm_rows_fp32(x, weight, y, num_tokens, pl.cast(Q_LORA, pl.INT32))
    return y


@pl.jit
def k_mx_wq_b_fp32(
    act: pl.Tensor[[T_MAX, ACT_MAX], pl.FP32],
    weight: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    weight_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[T_MAX, ACT_MAX], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    _mxfp8_wq_b_fp32(act, weight, weight_scale, out, num_tokens)
    return out


@pl.jit
def k_mx_wo_compact(
    act: pl.Tensor[[T_MAX, LOCAL_O_WIDTH], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    weight_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[T_MAX, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    _mxfp8_wo_b_compact(act, weight, weight_scale, out, num_tokens)
    return out


def _stats(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a = actual.detach().cpu().float().reshape(-1)
    e = expected.detach().cpu().float().reshape(-1)
    if a.numel() != e.numel():
        return {
            "ok": False,
            "bad": -1,
            "n": 0,
            "pct": -1.0,
            "max": float("nan"),
            "mean": float("nan"),
            "cos": float("nan"),
            "note": f"shape {tuple(actual.shape)} vs {tuple(expected.shape)}",
        }
    if actual.dtype in (torch.int32, torch.int64):
        neq = int((actual.cpu().reshape(-1).to(torch.int64) != expected.cpu().reshape(-1).to(torch.int64)).sum())
        return {
            "ok": neq == 0,
            "bad": neq,
            "n": int(a.numel()),
            "pct": 100.0 * neq / max(a.numel(), 1),
            "max": float("nan"),
            "mean": float("nan"),
            "cos": 1.0 if neq == 0 else 0.0,
        }
    diff = (a - e).abs()
    bad = int((diff > (ATOL + RTOL * e.abs())).sum())
    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), e.unsqueeze(0), dim=1))
    return {
        "ok": bad == 0,
        "bad": bad,
        "n": int(a.numel()),
        "pct": 100.0 * bad / max(a.numel(), 1),
        "max": float(diff.max()) if a.numel() else 0.0,
        "mean": float(diff.mean()) if a.numel() else 0.0,
        "cos": cos,
    }


def _slice(t: torch.Tensor, rows: int, cols: int | None = None) -> torch.Tensor:
    t = t[:rows]
    if cols is None:
        return t
    return t[..., :cols]


def _fmt(name: str, st: dict) -> str:
    if st.get("note"):
        return f"{name:22s} {st['note']}"
    return (
        f"{name:22s} bad={st['bad']}/{st['n']} ({st['pct']:.2f}%) "
        f"max={st['max']:.5g} mean={st['mean']:.5g} cos={st['cos']:.6f}"
    )


def _mx_golden(n_dim: int, act_k: int | None = None):
    def golden(tensors):
        n = int(tensors["num_tokens"])
        k = tensors["weight"].shape[0] if act_k is None else act_k
        out = mxfp8_linear(
            tensors["act"][:n, :k],
            tensors["weight"],
            tensors["weight_scale"],
            output_dtype=torch.float32,
        )
        tensors["out"].zero_()
        tensors["out"][:n, :n_dim] = out

    return golden


def _mx_fp32_specs():
    weight, scale = _mxfp8_weight(Q_LORA, LOCAL_H * HEAD_DIM)
    return [
        TensorSpec("act", [T_MAX, ACT_MAX], torch.float32, init_value=torch.randn),
        TensorSpec("weight", [Q_LORA, LOCAL_H * HEAD_DIM], torch.float8_e4m3fn, init_value=lambda w=weight: w),
        TensorSpec("weight_scale", [Q_LORA // 32, LOCAL_H * HEAD_DIM], scale.dtype, init_value=lambda s=scale: s),
        TensorSpec("out", [T_MAX, ACT_MAX], torch.float32),
        ScalarSpec("num_tokens", torch.int32, N_TOK),
    ]


def _wo_compact_specs():
    weight, scale = _mxfp8_weight(LOCAL_O_WIDTH, D)
    return [
        TensorSpec(
            "act",
            [T_MAX, LOCAL_O_WIDTH],
            torch.bfloat16,
            init_value=lambda: torch.randn(T_MAX, LOCAL_O_WIDTH, dtype=torch.bfloat16),
        ),
        TensorSpec("weight", [LOCAL_O_WIDTH, D], torch.float8_e4m3fn, init_value=lambda w=weight: w),
        TensorSpec("weight_scale", [LOCAL_O_WIDTH // 32, D], scale.dtype, init_value=lambda s=scale: s),
        TensorSpec("out", [T_MAX, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, N_TOK),
    ]


def _rms_fp32_specs():
    return [
        TensorSpec("x", [T_MAX, ACT_MAX], torch.float32, init_value=torch.randn),
        TensorSpec(
            "weight",
            [Q_LORA],
            torch.bfloat16,
            init_value=lambda: torch.ones(Q_LORA, dtype=torch.bfloat16),
        ),
        TensorSpec("y", [T_MAX, ACT_MAX], torch.float32),
        ScalarSpec("num_tokens", torch.int32, N_TOK),
    ]


def _goldens():
    def rms(tensors):
        n = int(tensors["num_tokens"])
        y = rms_norm(tensors["x"][:n, :Q_LORA], tensors["weight"]).to(torch.bfloat16)
        tensors["y"].zero_()
        tensors["y"][:n, :Q_LORA] = y

    def rms_fp32(tensors):
        n = int(tensors["num_tokens"])
        y = rms_norm(tensors["x"][:n, :Q_LORA], tensors["weight"])
        tensors["y"].zero_()
        tensors["y"][:n, :Q_LORA] = y.float()

    def rope_prep(tensors):
        n = int(tensors["num_tokens"])
        cos_h = tensors["rope_cos"][:n]
        sin_h = tensors["rope_sin"][:n]
        tensors["cos_il"].zero_()
        tensors["sin_signed"].zero_()
        tensors["cos_il"][:n, 0::2] = cos_h
        tensors["cos_il"][:n, 1::2] = cos_h
        tensors["sin_signed"][:n, 0::2] = -sin_h
        tensors["sin_signed"][:n, 1::2] = sin_h

    def rope_apply(key_n: str):
        def golden(tensors):
            n = int(tensors["num_tokens"])
            values = tensors["values"].clone()
            width = values.shape[-1]
            rope_off = width - ROPE_DIM
            chunk = values[:n, rope_off : rope_off + ROPE_DIM].float()
            even = chunk[..., 0::2]
            odd = chunk[..., 1::2]
            swapped = torch.stack((odd, even), dim=-1).flatten(-2)
            rot = chunk * tensors["cos_il"][:n] + swapped * tensors["sin_signed"][:n]
            values[:n, rope_off : rope_off + ROPE_DIM] = rot.to(values.dtype)
            tensors["values"].copy_(values)

        golden.__name__ = f"golden_{key_n}"
        return golden

    def qkv(tensors):
        n = int(tensors["num_tokens"])
        query, window_kv, qr = qkv_proj_rope(
            tensors["x"][:n],
            tensors["wq_a"],
            tensors["wq_a_scale"],
            tensors["q_norm_weight"],
            tensors["wq_b"],
            tensors["wq_b_scale"],
            tensors["wkv"],
            tensors["wkv_scale"],
            tensors["kv_norm_weight"],
            tensors["rope_cos"][:n],
            tensors["rope_sin"][:n],
        )
        tensors["qr"].zero_()
        tensors["q"].zero_()
        tensors["window_kv"].zero_()
        tensors["qr"][:n] = qr
        tensors["q"][:n] = query
        tensors["window_kv"][:n] = window_kv

    def compressor(tensors):
        n = int(tensors["num_tokens"])
        state = tensors["compressor_state"].clone()
        latent, publish = compressor_ratio2_paged(
            tensors["x"][:n],
            tensors["position_ids"][:n],
            tensors["compressor_state_rows"][:n],
            state,
            tensors["compressor_wkv"],
            tensors["compressor_wgate"],
            tensors["compressor_norm_weight"],
        )
        tensors["compressor_state"].copy_(state)
        tensors["latent"].zero_()
        tensors["publish_mask"].zero_()
        tensors["latent"][:n] = latent
        tensors["publish_mask"][:n] = publish.to(torch.int32)

    def oproj(tensors):
        n = int(tensors["num_tokens"])
        out = _project_output(
            tensors["attended"][:n],
            tensors["rope_cos"][:n],
            tensors["rope_sin"][:n],
            tensors["wo_a"],
            tensors["wo_b"],
            tensors["wo_b_scale"],
        )
        tensors["output_partial"].zero_()
        tensors["output_partial"][:n] = out.float()

    def attn(tensors):
        n = int(tensors["num_tokens"])
        win = tensors["window_bf16"].view(1, -1, 1, HEAD_DIM)
        cmp = tensors["cmp_bf16"].view(1, -1, 1, HEAD_DIM)
        out = paged_sparse_attention(
            tensors["q"][:n],
            win,
            tensors["window_indices"][:n],
            cmp,
            tensors["topk_indices"][:n],
            tensors["attn_sink"],
        )
        tensors["attended"].zero_()
        tensors["attended"][:n] = out

    def bf16(tensors):
        n = int(tensors["num_tokens"])
        tensors["out"].zero_()
        tensors["out"][:n] = torch.matmul(
            tensors["act"][:n].float(), tensors["weight"].float()
        )

    def fp32(tensors):
        n = int(tensors["num_tokens"])
        tensors["out"].zero_()
        tensors["out"][:n] = torch.matmul(
            tensors["act"][:n].float(), tensors["weight"].float()
        )

    return {
        "rms": rms,
        "rms_fp32": rms_fp32,
        "rope_prep": rope_prep,
        "rope_kv": rope_apply("kv"),
        "rope_hd": rope_apply("hd"),
        "rope_idx": rope_apply("idx"),
        "mx_wq_a": _mx_golden(Q_LORA),
        "mx_wkv": _mx_golden(HEAD_DIM),
        "mx_idx": _mx_golden(INDEX_H * INDEX_DIM),
        "mx_wq_b": _mx_golden(LOCAL_H * HEAD_DIM),
        "mx_wq_b_fp32": _mx_golden(LOCAL_H * HEAD_DIM),
        "mx_wo": _mx_golden(D, act_k=LOCAL_O_WIDTH),
        "mx_wo_compact": _mx_golden(D),
        "bf16_linear": bf16,
        "fp32_linear": fp32,
        "qkv": qkv,
        "compressor": compressor,
        "oproj": oproj,
        "attn": attn,
    }


def _compare_maps():
    return {
        "rms": {"y": (N_TOK, Q_LORA)},
        "rms_fp32": {"y": (N_TOK, Q_LORA)},
        "rope_prep": {"cos_il": (N_TOK, None), "sin_signed": (N_TOK, None)},
        "rope_kv": {"values": (N_TOK, None)},
        "rope_hd": {"values": (min(N_TOK * LOCAL_H, 32), None)},
        "rope_idx": {"values": (N_TOK, None)},
        "mx_wq_a": {"out": (N_TOK, Q_LORA)},
        "mx_wkv": {"out": (N_TOK, HEAD_DIM)},
        "mx_idx": {"out": (N_TOK, INDEX_H * INDEX_DIM)},
        "mx_wq_b": {"out": (N_TOK, LOCAL_H * HEAD_DIM)},
        "mx_wq_b_fp32": {"out": (N_TOK, LOCAL_H * HEAD_DIM)},
        "mx_wo": {"out": (N_TOK, D)},
        "mx_wo_compact": {"out": (N_TOK, D)},
        "bf16_linear": {"out": (N_TOK, None)},
        "fp32_linear": {"out": (N_TOK, None)},
        "qkv": {"qr": (N_TOK, None), "q": (N_TOK, None), "window_kv": (N_TOK, None)},
        "compressor": {"latent": (N_TOK, None), "publish_mask": (N_TOK, None)},
        "oproj": {"output_partial": (N_TOK, None)},
        "attn": {"attended": (N_TOK, None)},
    }


def _cases():
    built = _build_cases()
    extra = {
        "rms_fp32": (k_rms_fp32, _rms_fp32_specs()),
        "mx_wq_b_fp32": (k_mx_wq_b_fp32, _mx_fp32_specs()),
        "mx_wo_compact": (k_mx_wo_compact, _wo_compact_specs()),
    }
    names = [
        "rms",
        "rms_fp32",
        "rope_prep",
        "rope_kv",
        "rope_hd",
        "rope_idx",
        "mx_wq_a",
        "mx_wkv",
        "mx_idx",
        "mx_wq_b",
        "mx_wq_b_fp32",
        "mx_wo",
        "mx_wo_compact",
        "bf16_linear",
        "fp32_linear",
        "qkv",
        "compressor",
        "attn",
        "oproj",
    ]
    table = {}
    for name in names:
        if name in extra:
            table[name] = extra[name]
        else:
            table[name] = built[name]
    return names, table


def run_one(name: str, device: int) -> dict:
    torch.manual_seed(0)
    names, table = _cases()
    if name not in table:
        raise SystemExit(f"unknown kernel {name}; known={names}")
    fn, specs = table[name]
    goldens = _goldens()
    slices = _compare_maps()[name]
    dump: dict[str, dict] = {}

    def capture(key: str, rows: int, cols: int | None):
        def _fn(actual, expected, **_kwargs):
            st = _stats(_slice(actual, rows, cols), _slice(expected, rows, cols))
            dump[key] = st
            print("  " + _fmt(f"{name}/{key}", st), flush=True)
            return True, ""

        _fn.__name__ = f"cap_{name}_{key}"
        return _fn

    compare_fn = {key: capture(key, rows, cols) for key, (rows, cols) in slices.items()}
    print(f"\n======== ACC {name} device={device} ========", flush=True)
    try:
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=goldens[name],
            compare_fn=compare_fn,
            config=dict(platform="a5", device_id=device, ring_heap=RING_HEAP_4G),
        )
    except Exception as exc:  # noqa: BLE001
        detail = traceback.format_exc()
        print(detail, flush=True)
        return {"kernel": name, "run": "FAIL", "error": str(exc), "outputs": dump}
    return {
        "kernel": name,
        "run": "PASS" if result.passed else "FAIL",
        "error": result.error or "",
        "outputs": dump,
    }


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=int, default=int(os.environ.get("TASK_DEVICE", "2").split(",")[0]))
    parser.add_argument("--kernel", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(OUT_PATH))
    args, _ = parser.parse_known_args()
    names, _table = _cases()
    if args.kernel:
        row = run_one(args.kernel, args.device)
        _write(Path(args.out), row)
        print(json.dumps(row, indent=2), flush=True)
        return 0 if row["run"] == "PASS" else 1

    rows = []
    script = str(Path(__file__).resolve())
    for name in names:
        one_out = f"/tmp/c2a_acc_{name}.json"
        cmd = [
            sys.executable,
            script,
            "--tp",
            "1",
            "-d",
            str(args.device),
            "--kernel",
            name,
            "--out",
            one_out,
        ]
        print(f"\n[ACC] subprocess {name}", flush=True)
        proc = subprocess.run(cmd, check=False)
        try:
            row = json.loads(Path(one_out).read_text(encoding="utf-8"))
        except OSError:
            row = {"kernel": name, "run": "FAIL", "error": f"exit={proc.returncode}", "outputs": {}}
        rows.append(row)

    print("\n======== KERNEL ACCURACY SUMMARY (atol=0.02 rtol=1/32) ========", flush=True)
    for row in rows:
        outs = row.get("outputs") or {}
        if not outs:
            print(f"{row['kernel']:22s} RUN={row['run']:4s} {row.get('error', '')[:120]}", flush=True)
            continue
        for key, st in outs.items():
            tag = "OK " if st.get("ok") else "BAD"
            print(f"{tag} {_fmt(row['kernel'] + '/' + key, st)}", flush=True)
    _write(Path(args.out), rows)
    print(f"[ACC] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
