# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B decode Scope 1 — input RMSNorm + Q/K/V projection.

SPMD variant ported from ``models/qwen3/32b/qwen3_32b_decode.py``: outer
``pl.spmd`` over output tiles with inner ``pl.pipeline`` over the K dimension.
"""

import pypto.language as pl

BATCH = 16
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 5120
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

RMSNORM_K_CHUNK = 512
Q_OUT_CHUNK = 256
Q_PROJ_K_CHUNK = 128
KV_OUT_CHUNK = 256
KV_PROJ_K_CHUNK = 128


def build_qwen3_scope1_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    hidden = hidden_size
    kv_hidden = num_kv_heads * head_dim
    q_out_blocks = hidden // Q_OUT_CHUNK
    kv_out_blocks = kv_hidden // KV_OUT_CHUNK

    @pl.program
    class Qwen3Scope1:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_scope1(
            self,
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            wq: pl.Tensor[[hidden, hidden], pl.BF16],
            wk: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            q_proj: pl.Out[pl.Tensor[[batch, hidden], pl.FP32]],
            k_proj: pl.Out[pl.Tensor[[batch, kv_hidden], pl.FP32]],
            v_proj: pl.Out[pl.Tensor[[batch, kv_hidden], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[batch, hidden], pl.FP32],
            pl.Tensor[[batch, kv_hidden], pl.FP32],
            pl.Tensor[[batch, kv_hidden], pl.FP32],
        ]:
            normed_states = pl.create_tensor([batch, hidden], dtype=pl.BF16)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                partial_sq = pl.full([1, batch], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(hidden // RMSNORM_K_CHUNK, stage=4):
                    k0 = kb * RMSNORM_K_CHUNK
                    x_chunk = pl.cast(hidden_states[:, k0 : k0 + RMSNORM_K_CHUNK], target_type=pl.FP32)
                    partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, batch]))
                variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [batch, 1])
                inv_rms = pl.recip(pl.sqrt(variance))
                for kb in pl.pipeline(hidden // RMSNORM_K_CHUNK, stage=4):
                    k0 = kb * RMSNORM_K_CHUNK
                    x_chunk = pl.cast(hidden_states[:, k0 : k0 + RMSNORM_K_CHUNK], target_type=pl.FP32)
                    gamma = input_rms_weight[:, k0 : k0 + RMSNORM_K_CHUNK]
                    normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                    normed_states = pl.assemble(normed_states, pl.cast(normed, target_type=pl.BF16), [0, k0])

            # Q projection.
            for qi in pl.spmd(q_out_blocks, name_hint="q_proj"):
                q0 = qi * Q_OUT_CHUNK
                q_acc = pl.create_tensor([batch, Q_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.pipeline(0, hidden // Q_PROJ_K_CHUNK, stage=2):
                    k0 = kb * Q_PROJ_K_CHUNK
                    tile_a_i = normed_states[:, k0 : k0 + Q_PROJ_K_CHUNK]
                    tile_b_i = wq[k0 : k0 + Q_PROJ_K_CHUNK, q0 : q0 + Q_OUT_CHUNK]
                    if k0 == 0:
                        q_acc = pl.matmul(tile_a_i, tile_b_i, out_dtype=pl.FP32)
                    else:
                        q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_b_i)
                q_proj = pl.assemble(q_proj, q_acc, [0, q0])

            # K/V projection.
            for kvi in pl.spmd(kv_out_blocks, name_hint="kv_proj"):
                kv0 = kvi * KV_OUT_CHUNK
                k_acc = pl.create_tensor([batch, KV_OUT_CHUNK], dtype=pl.FP32)
                v_acc = pl.create_tensor([batch, KV_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.pipeline(0, hidden // KV_PROJ_K_CHUNK, stage=2):
                    k0 = kb * KV_PROJ_K_CHUNK
                    tile_a_i = normed_states[:, k0 : k0 + KV_PROJ_K_CHUNK]
                    tile_wk_i = wk[k0 : k0 + KV_PROJ_K_CHUNK, kv0 : kv0 + KV_OUT_CHUNK]
                    tile_wv_i = wv[k0 : k0 + KV_PROJ_K_CHUNK, kv0 : kv0 + KV_OUT_CHUNK]
                    if k0 == 0:
                        k_acc = pl.matmul(tile_a_i, tile_wk_i, out_dtype=pl.FP32)
                        v_acc = pl.matmul(tile_a_i, tile_wv_i, out_dtype=pl.FP32)
                    else:
                        k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                        v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                k_proj = pl.assemble(k_proj, k_acc, [0, kv0])
                v_proj = pl.assemble(v_proj, v_acc, [0, kv0])

            return q_proj, k_proj, v_proj

    return Qwen3Scope1


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    import torch
    from golden import TensorSpec

    kv_hidden = num_kv_heads * head_dim

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_rms_weight():
        return torch.rand(1, hidden_size) - 0.5

    def init_wq():
        return (torch.rand(hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_wk():
        return (torch.rand(hidden_size, kv_hidden) - 0.5) / hidden_size ** 0.5

    def init_wv():
        return (torch.rand(hidden_size, kv_hidden) - 0.5) / hidden_size ** 0.5

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32, init_value=init_rms_weight),
        TensorSpec("wq", [hidden_size, hidden_size], torch.bfloat16, init_value=init_wq),
        TensorSpec("wk", [hidden_size, kv_hidden], torch.bfloat16, init_value=init_wk),
        TensorSpec("wv", [hidden_size, kv_hidden], torch.bfloat16, init_value=init_wv),
        TensorSpec("q_proj", [batch, hidden_size], torch.float32, is_output=True),
        TensorSpec("k_proj", [batch, kv_hidden], torch.float32, is_output=True),
        TensorSpec("v_proj", [batch, kv_hidden], torch.float32, is_output=True),
    ]


def golden_qwen3_scope1(tensors):
    import torch

    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]

    x = hidden_states.float()
    sq_sum = (x ** 2).sum(dim=-1, keepdim=True)
    variance = sq_sum / x.shape[-1] + EPS
    rms = torch.sqrt(variance)
    normed = (x / rms * input_rms_weight.float()).bfloat16()

    tensors["q_proj"][:] = (normed.float() @ wq.float()).float()
    tensors["k_proj"][:] = (normed.float() @ wk.float()).float()
    tensors["v_proj"][:] = (normed.float() @ wv.float()).float()


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_qwen3_scope1_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_qwen3_scope1,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(platform=args.platform, device_id=args.device, runtime_profiling=args.runtime_profiling),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
