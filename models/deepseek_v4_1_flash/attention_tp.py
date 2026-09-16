# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure tensor-parallel output reduction shared by every attention mode."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash import config as C

# Module-level mirrors — kernel bodies cannot do ATTR access.
D = C.D
DECODE_MAX_TOKENS = C.DECODE_MAX_TOKENS
PREFILL_MAX_TOKENS = C.PREFILL_MAX_TOKENS
TP_SIZE = C.TP_SIZE
T_DYN = C.T_DYN


# Row / column tiles for local window publish / remote accumulate.
_TP_ROW_TILE = 8
_TP_COL_TILE = 512
assert DECODE_MAX_TOKENS % _TP_ROW_TILE == 0
assert PREFILL_MAX_TOKENS % _TP_ROW_TILE == 0
assert D % _TP_COL_TILE == 0
assert D % 64 == 0


def golden_tp_output_all_reduce(output_partials: torch.Tensor) -> torch.Tensor:
    """Sum the row-parallel output projection from every TP rank."""
    return output_partials.float().sum(dim=0).to(output_partials.dtype)


@pl.jit.inline(auto_scope=False)
def _tp_output_all_reduce(
    output_partial: pl.Tensor[[T_DYN, D], pl.FP32],
    output_window: pld.DistributedTensor,
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    max_tokens: pl.Scalar[pl.INT32],
):
    """Publish local FP32 partials, barrier, then sum TP windows into BF16 output.

    Each rank owns a ``[max_tokens, D]`` window slice. The local contribution is
    staged at row offset ``0`` (one batch of continuous-batch tokens). Peers are
    reached via ``group_base + tp``; arrival credits use ``attention_epoch``.
    TP=1 leaves the peer loops empty and reduces to a local cast.
    """
    n_pub = (num_tokens + _TP_ROW_TILE - 1) // _TP_ROW_TILE
    n_cols = D // _TP_COL_TILE
    with pl.spmd(n_pub * n_cols, name_hint="tp_out_publish") as pub_tid:
        pb = pl.tile.get_block_idx()
        rt = pb // n_cols
        ct = pb - rt * n_cols
        t0 = rt * _TP_ROW_TILE
        d0 = ct * _TP_COL_TILE
        rows = pl.min(_TP_ROW_TILE, num_tokens - t0)
        if t0 < num_tokens:
            partial = pl.load(output_partial, [t0, d0], [_TP_ROW_TILE, _TP_COL_TILE])
            output_window = pl.store(partial, [t0, d0], output_window)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_out_notify", deps=[pub_tid]) as nfy_tid:
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=output_arrived,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_out_wait", deps=[nfy_tid]) as wait_tid:
        expected = pl.cast(attention_epoch, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(
                    signal=output_arrived,
                    offsets=[src_tp, 0],
                    expected=expected,
                    cmp=pld.WaitCmp.Ge,
                )

    n_red = (num_tokens + _TP_ROW_TILE - 1) // _TP_ROW_TILE
    with pl.spmd(n_red * n_cols, name_hint="tp_out_reduce", deps=[wait_tid]):
        rb = pl.tile.get_block_idx()
        rt = rb // n_cols
        ct = rb - rt * n_cols
        r0 = rt * _TP_ROW_TILE
        d0 = ct * _TP_COL_TILE
        if r0 < num_tokens:
            for dt in pl.range(_TP_ROW_TILE):
                t = r0 + dt
                if t < num_tokens:
                    acc = pl.load(output_window, [t, d0], [1, _TP_COL_TILE])
                    for peer_tp in pl.range(TP_SIZE):
                        if peer_tp != tp_rank:
                            recv = pld.tile.remote_load(
                                output_window,
                                peer=group_base + peer_tp,
                                offsets=[t, d0],
                                shape=[1, _TP_COL_TILE],
                            )
                            acc = pl.add(acc, recv)
                    out_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
                    output = pl.store(out_bf16, [t, d0], output)

    _ = max_tokens


@pl.jit.inline(auto_scope=False)
def prefill_tp_output_all_reduce(
    output_partial: pl.Tensor[[T_DYN, D], pl.FP32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    max_tokens = pl.cast(PREFILL_MAX_TOKENS, pl.INT32)
    _tp_output_all_reduce(
        output_partial,
        output_window,
        output_arrived,
        output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
        max_tokens,
    )


@pl.jit.inline(auto_scope=False)
def decode_tp_output_all_reduce(
    output_partial: pl.Tensor[[T_DYN, D], pl.FP32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    max_tokens = pl.cast(DECODE_MAX_TOKENS, pl.INT32)
    _tp_output_all_reduce(
        output_partial,
        output_window,
        output_arrived,
        output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
        max_tokens,
    )


__all__ = ["decode_tp_output_all_reduce", "golden_tp_output_all_reduce", "prefill_tp_output_all_reduce"]


if __name__ == "__main__":
    torch.manual_seed(17)
    partials = torch.randn(4, 7, 5)
    reduced = golden_tp_output_all_reduce(partials)
    torch.testing.assert_close(reduced, partials.float().sum(dim=0).to(partials.dtype))
    print(f"[GOLDEN] PASS attention pure-TP all-reduce output={tuple(reduced.shape)}")
