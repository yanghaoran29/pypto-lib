# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Minimal repro of the qwen3 SPMD qk_matmul + dynamic ``all_q_padded`` pattern.

Distilled from ``models/qwen3/14b/qwen3_14b_decode_spmd.py`` (the qk_matmul
``pl.spmd`` block at lines 391-420). Two simplifications vs. the production
model:

  * ``all_q_padded`` is **created** via ``pl.create_tensor`` with the same
    ``[batch_padded * total_q_groups * Q_HEAD_PAD, head_dim]`` shape (the
    ``Var * Const`` shape that previously tripped the kernel-wrapper /
    ptoas-signature mismatch); it is filled with zeros, no rope_kv_cache.
  * The paired ``gi0``/``gi1`` matmuls in the original ``pl.spmd`` body
    are collapsed to a single matmul -- one ``q_padded`` per spmd block --
    so the example exercises the exact same SPMD outline boundary but
    without the duplicated code.

End-to-end "does it compile" check: ``python spmd_qk_matmul_dyn.py``
calls ``ir.compile(...)``. On the broken main-branch codegen the kernel
wrapper and ptoas-generated function signatures disagree by one int32
(the missing ``batch_padded`` slot) and clang errors with
"no matching function for call to 'qk_matmul_kernel'". On the
usage-driven dyn-dim fix both sides agree and compilation finishes clean.
"""
# pyright: reportUndefinedVariable=false

import pypto.language as pl

USER_BATCH_DYN = pl.dynamic("USER_BATCH_DYN")
KV_CACHE_ROWS_DYN = pl.dynamic("KV_CACHE_ROWS_DYN")

BATCH_TILE = 16
HEAD_DIM = 128
NUM_KV_HEADS = 8
Q_HEAD_PAD = 16
BLOCK_SIZE = 256
TOTAL_Q_GROUPS = NUM_KV_HEADS  # q_per_kv == Q_HEAD_BATCH, so q_groups == 1
SPMD_GROUPS = TOTAL_Q_GROUPS // 2


def build_spmd_qk_matmul_program(head_dim: int = HEAD_DIM):
    total_q_groups = TOTAL_Q_GROUPS

    @pl.program
    class SpmdQkMatmulDyn:
        @pl.function(type=pl.FunctionType.Opaque)
        def qk_matmul_entry(
            self,
            hidden_states: pl.Tensor[[USER_BATCH_DYN, head_dim], pl.BF16],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, head_dim], pl.BF16],
            out: pl.Out[pl.Tensor[[USER_BATCH_DYN, total_q_groups * Q_HEAD_PAD * BLOCK_SIZE], pl.FP32]],
        ) -> pl.Tensor[[USER_BATCH_DYN, total_q_groups * Q_HEAD_PAD * BLOCK_SIZE], pl.FP32]:
            # ``hidden_states`` is only kept around to source the dynamic
            # ``user_batch`` -- it is never read in the body. The bug
            # pattern we're reproducing lives entirely in the Mul-shape
            # ``pl.create_tensor`` below and the SPMD slice off it.
            user_batch = pl.tensor.dim(hidden_states, 0)
            batch_padded = ((user_batch + BATCH_TILE - 1) // BATCH_TILE) * BATCH_TILE

            all_q_padded = pl.create_tensor(
                [batch_padded * total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16,
            )

            for b in pl.parallel(user_batch):
                all_raw_scores = pl.create_tensor(
                    [total_q_groups * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.FP32,
                )
                for gi in pl.spmd(SPMD_GROUPS, name_hint="qk_matmul"):
                    q_padded_row = b * total_q_groups * Q_HEAD_PAD + gi * Q_HEAD_PAD
                    q_padded = pl.slice(all_q_padded, [Q_HEAD_PAD, head_dim], [q_padded_row, 0])
                    cache_row = gi * BLOCK_SIZE
                    k_tile = pl.slice(k_cache, [BLOCK_SIZE, head_dim], [cache_row, 0])
                    raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                    all_raw_scores = pl.assemble(
                        all_raw_scores,
                        raw_scores,
                        [gi * Q_HEAD_PAD, 0],
                    )

                flat = pl.reshape(
                    all_raw_scores,
                    [1, total_q_groups * Q_HEAD_PAD * BLOCK_SIZE],
                )
                out = pl.assemble(out, flat, [b, 0])
            return out

    return SpmdQkMatmulDyn


if __name__ == "__main__":
    import argparse

    from pypto import ir

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    args = parser.parse_args()

    program = build_spmd_qk_matmul_program()
    compiled = ir.compile(program, platform=args.platform, dump_passes=True)
    print(f"[OK] compiled successfully -> {compiled.output_dir}")
