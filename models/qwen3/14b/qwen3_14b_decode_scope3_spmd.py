# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B decode Scope 3 (SPMD variant) — output projection + residual + post RMSNorm + MLP + residual.

Ports the SPMD layout from ``models/qwen3/32b/qwen3_32b_decode.py``:

- Out-proj + residual: ``pl.parallel(step=2)`` outer, two inner blocks, with
  ``pl.pipeline`` over the K dimension.
- Post RMSNorm: single ``pl.at(CORE_GROUP)`` block with two pipelined passes.
- Gate / Up / SiLU: outer ``pl.parallel`` over MLP_SPMD_INNER groups, inner
  ``pl.spmd(MLP_SPMD_INNER)`` for each of gate, up, silu_fuse stages.
- Down-proj + residual writeback: ``pl.parallel(step=2)`` outer, ``pl.pipeline``
  over the K (intermediate) dimension.
"""

import pypto.language as pl

BATCH = 16
HIDDEN = 5120
INTERMEDIATE = 17408

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

K_CHUNK = 128
Q_OUT_CHUNK = 256
OUT_PROJ_K_CHUNK = 128
MLP_OUT_CHUNK = 256
DOWN_N_CHUNK = 256
DOWN_K_CHUNK = 128

MLP_SPMD_INNER = 2
MLP_GROUP_CHUNK = MLP_SPMD_INNER * MLP_OUT_CHUNK


def build_qwen3_scope3_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    hidden_blocks = hidden_size // K_CHUNK
    out_proj_blocks = hidden_size // Q_OUT_CHUNK
    mlp_out_blocks = intermediate_size // MLP_OUT_CHUNK
    down_proj_blocks = hidden_size // DOWN_N_CHUNK

    assert mlp_out_blocks % MLP_SPMD_INNER == 0
    assert out_proj_blocks % 2 == 0
    assert down_proj_blocks % 2 == 0

    @pl.program
    class Qwen3Scope3:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_scope3(
            self,
            attn_out: pl.Tensor[[batch, hidden_size], pl.BF16],
            hidden_states: pl.Tensor[[batch, hidden_size], pl.BF16],
            wo: pl.Tensor[[hidden_size, hidden_size], pl.BF16],
            post_rms_weight: pl.Tensor[[1, hidden_size], pl.FP32],
            w_gate: pl.Tensor[[hidden_size, intermediate_size], pl.BF16],
            w_up: pl.Tensor[[hidden_size, intermediate_size], pl.BF16],
            w_down: pl.Tensor[[intermediate_size, hidden_size], pl.BF16],
            out: pl.Out[pl.Tensor[[batch, hidden_size], pl.BF16]],
        ) -> pl.Tensor[[batch, hidden_size], pl.BF16]:
            resid1_tile = pl.create_tensor([batch, hidden_size], dtype=pl.FP32)

            # Stage 1 & 2: Output projection + residual addition with hidden_states.
            for ob in pl.parallel(0, out_proj_blocks, 2):
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)], name_hint="out_proj_residual"):
                    for oi in pl.range(ob, ob + 2):
                        o0 = oi * Q_OUT_CHUNK
                        hidden_chunk = hidden_states[:, o0 : o0 + Q_OUT_CHUNK]
                        o_acc = pl.create_tensor([batch, Q_OUT_CHUNK], dtype=pl.FP32)
                        for kb in pl.pipeline(0, hidden_size // OUT_PROJ_K_CHUNK, stage=2):
                            k0 = kb * OUT_PROJ_K_CHUNK
                            a_chunk = attn_out[:, k0 : k0 + OUT_PROJ_K_CHUNK]
                            w_chunk = wo[k0 : k0 + OUT_PROJ_K_CHUNK, o0 : o0 + Q_OUT_CHUNK]
                            if k0 == 0:
                                o_acc = pl.matmul(a_chunk, w_chunk, out_dtype=pl.FP32)
                            else:
                                o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)
                        resid = pl.cast(hidden_chunk, target_type=pl.FP32)
                        resid_sum = pl.add(o_acc, resid)
                        resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

            # Stage 3: Post-attention RMSNorm.
            post_norm_tile = pl.create_tensor([batch, hidden_size], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
                sq_sum = pl.full([1, batch], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(hidden_blocks, stage=2):
                    k0 = kb * K_CHUNK
                    resid_chunk = resid1_tile[:, k0 : k0 + K_CHUNK]
                    sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, batch]))
                inv_rms_s3 = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))
                inv_rms_s3_col = pl.reshape(inv_rms_s3, [batch, 1])
                for kb in pl.pipeline(hidden_blocks, stage=2):
                    k0 = kb * K_CHUNK
                    resid_chunk = resid1_tile[:, k0 : k0 + K_CHUNK]
                    post_gamma = post_rms_weight[:, k0 : k0 + K_CHUNK]
                    post_normed = pl.col_expand_mul(pl.row_expand_mul(resid_chunk, inv_rms_s3_col), post_gamma)
                    post_norm_tile = pl.assemble(post_norm_tile, pl.cast(post_normed, target_type=pl.BF16), [0, k0])

            # Stage 4~6: outer parallel + inner SPMD groups, cache per group only.
            mlp_tile = pl.create_tensor([batch, intermediate_size], dtype=pl.BF16)
            for ob_base in pl.parallel(0, mlp_out_blocks, MLP_SPMD_INNER):
                gate_group = pl.create_tensor([batch, MLP_GROUP_CHUNK], dtype=pl.FP32)
                up_group = pl.create_tensor([batch, MLP_GROUP_CHUNK], dtype=pl.FP32)

                # Stage 4: gate projection.
                for ob in pl.spmd(MLP_SPMD_INNER, name_hint="gate_proj_spmd"):
                    o0 = (ob_base + ob) * MLP_OUT_CHUNK
                    g0 = ob * MLP_OUT_CHUNK
                    post_chunk_0 = pl.slice(post_norm_tile, [batch, K_CHUNK], [0, 0])
                    post_chunk_1 = pl.slice(post_norm_tile, [batch, K_CHUNK], [0, K_CHUNK])
                    wg_0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                    gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)

                    wg_1 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [K_CHUNK, o0])
                    gate_acc = pl.matmul_acc(gate_acc, post_chunk_1, wg_1)

                    for kb in pl.pipeline(2, hidden_blocks, stage=2):
                        k0 = kb * K_CHUNK
                        post_chunk = pl.slice(post_norm_tile, [batch, K_CHUNK], [0, k0])
                        wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                        gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
                    gate_group = pl.assemble(gate_group, gate_acc, [0, g0])

                # Stage 5: up projection.
                for ob in pl.spmd(MLP_SPMD_INNER, name_hint="up_proj_spmd"):
                    o0 = (ob_base + ob) * MLP_OUT_CHUNK
                    g0 = ob * MLP_OUT_CHUNK
                    post_chunk_0 = pl.slice(post_norm_tile, [batch, K_CHUNK], [0, 0])
                    post_chunk_1 = pl.slice(post_norm_tile, [batch, K_CHUNK], [0, K_CHUNK])
                    wu_0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                    up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)

                    wu_1 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [K_CHUNK, o0])
                    up_acc = pl.matmul_acc(up_acc, post_chunk_1, wu_1)

                    for kb in pl.pipeline(2, hidden_blocks, stage=2):
                        k0 = kb * K_CHUNK
                        post_chunk = pl.slice(post_norm_tile, [batch, K_CHUNK], [0, k0])
                        wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                        up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
                    up_group = pl.assemble(up_group, up_acc, [0, g0])

                # Stage 6: SiLU + gate/up fuse.
                for ob in pl.spmd(MLP_SPMD_INNER, name_hint="silu_spmd"):
                    o0 = (ob_base + ob) * MLP_OUT_CHUNK
                    g0 = ob * MLP_OUT_CHUNK
                    gate_acc = pl.slice(gate_group, [batch, MLP_OUT_CHUNK], [0, g0])
                    up_acc = pl.slice(up_group, [batch, MLP_OUT_CHUNK], [0, g0])
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                    mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                    mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                    mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, o0])

            # Stage 7 & 8: Down projection + final residual writeback.
            for db in pl.parallel(0, down_proj_blocks, 2):
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)], name_hint="down_proj_residual"):
                    for di in pl.range(db, db + 2):
                        d0 = di * DOWN_N_CHUNK
                        resid1_tile_chunk = resid1_tile[:, d0 : d0 + DOWN_N_CHUNK]
                        down_acc = pl.create_tensor([batch, DOWN_N_CHUNK], dtype=pl.FP32)
                        for ob in pl.pipeline(0, intermediate_size // DOWN_K_CHUNK, stage=2):
                            o0 = ob * DOWN_K_CHUNK
                            down_mlp_chunk = mlp_tile[:, o0 : o0 + DOWN_K_CHUNK]
                            w_down_chunk = w_down[o0 : o0 + DOWN_K_CHUNK, d0 : d0 + DOWN_N_CHUNK]
                            if o0 == 0:
                                down_acc = pl.matmul(down_mlp_chunk, w_down_chunk, out_dtype=pl.FP32)
                            else:
                                down_acc = pl.matmul_acc(down_acc, down_mlp_chunk, w_down_chunk)
                        out_chunk = pl.add(down_acc, resid1_tile_chunk)
                        out = pl.assemble(out, pl.cast(out_chunk, target_type=pl.BF16), [0, d0])

            return out

    return Qwen3Scope3


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    import torch
    from golden import TensorSpec

    def init_attn_out():
        return torch.rand(batch, hidden_size) - 0.5

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_wo():
        return (torch.rand(hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_post_rms_weight():
        return torch.ones(1, hidden_size)

    def init_w_gate():
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        return (torch.rand(intermediate_size, hidden_size) - 0.5) / intermediate_size ** 0.5

    return [
        TensorSpec("attn_out", [batch, hidden_size], torch.bfloat16, init_value=init_attn_out),
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("wo", [hidden_size, hidden_size], torch.bfloat16, init_value=init_wo),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32, init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden_size, intermediate_size], torch.bfloat16, init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, intermediate_size], torch.bfloat16, init_value=init_w_up),
        TensorSpec("w_down", [intermediate_size, hidden_size], torch.bfloat16, init_value=init_w_down),
        TensorSpec("out", [batch, hidden_size], torch.bfloat16, is_output=True),
    ]


def golden_qwen3_scope3(tensors):
    import torch

    attn_out = tensors["attn_out"]
    hidden_states = tensors["hidden_states"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()
    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + EPS)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()
    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())
    tensors["out"][:] = (down + resid1).bfloat16()


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_qwen3_scope3_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_qwen3_scope3,
        config=RunConfig(
            rtol=3e-3,
            atol=3e-3,
            compile=dict(dump_passes=True),
            runtime=dict(platform=args.platform, device_id=args.device, runtime_profiling=args.runtime_profiling),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
