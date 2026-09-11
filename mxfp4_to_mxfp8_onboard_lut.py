# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 trial: cast packed MXFP4 to MXFP8 and transpose it on device."""

from __future__ import annotations

import os

import pypto.language as pl
import torch
from pypto.runtime import RunConfig


K = 7168
N = 3072
PACKED_K = K // 2
N_TILE = 256
K_TILE = 64
TASK_N_TILE = 1024
TASK_K_TILE = 512

# DeepSeek-V4-Pro's E2M1 nibble to E4M3FN bit-pattern mapping.
NIBBLE_CODES = [
    0x00,
    0x30,
    0x38,
    0x3C,
    0x40,
    0x44,
    0x48,
    0x4C,
    0x80,
    0xB0,
    0xB8,
    0xBC,
    0xC0,
    0xC4,
    0xC8,
    0xCC,
]

@pl.jit
def mxfp4_to_mxfp8_onboard_cast(
    packed_nk: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Cast one W1/W3 expert from FP4 ``[N, K]`` into FP8 ``[K, N]``."""
    with pl.spmd((K // TASK_K_TILE) * (N // TASK_N_TILE)):
        block_idx = pl.tile.get_block_idx()
        k_base = (block_idx // (N // TASK_N_TILE)) * TASK_K_TILE
        n_base = (block_idx % (N // TASK_N_TILE)) * TASK_N_TILE
        for ni in pl.range(TASK_N_TILE // N_TILE):
            n0 = n_base + ni * N_TILE
            for ki in pl.range(TASK_K_TILE // K_TILE):
                k0 = k_base + ki * K_TILE
                packed_fp4 = pl.load(
                    packed_nk,
                    [n0, k0],
                    [N_TILE, K_TILE],
                    target_memory=pl.Mem.Vec,
                )
                values_bf16 = pl.cast(packed_fp4, target_type=pl.BF16)
                values_kn = pl.transpose(values_bf16, axis1=0, axis2=1)
                values_fp32 = pl.cast(values_kn, target_type=pl.FP32)
                values_fp8 = pl.cast(values_fp32, target_type=pl.FP8E4M3FN)
                pl.store(values_fp8, [k0, n0], out)
    return out


def build_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Build checkpoint-shaped MXFP4 bytes and the Cube-layout code golden."""
    indices_nk = torch.arange(N * K, dtype=torch.uint8).reshape(N, K) % 16
    low = indices_nk[:, 0::2]
    high = indices_nk[:, 1::2]
    packed_nk = (low | (high << 4)).contiguous()
    packed_fp4_nk = packed_nk.view(torch.float4_e2m1fn_x2)
    code_lut = torch.tensor(NIBBLE_CODES, dtype=torch.uint8).view(torch.float8_e4m3fn)
    golden = code_lut[indices_nk.to(torch.int64)].transpose(0, 1).contiguous()
    return packed_fp4_nk, golden


def device_id() -> int:
    """Use the scheduler-assigned device when task-submit provides one."""
    for name in ("TASK_DEVICE", "ACL_DEVICE_ID", "ASCEND_DEVICE_ID", "NPU_LOCKED_DEVICE"):
        value = os.environ.get(name)
        if value:
            return int(value.split(",")[0])
    return 0


def main() -> None:
    packed_nk, golden = build_inputs()
    out = torch.zeros((K, N), dtype=torch.float8_e4m3fn)
    platform = os.environ.get("PYPTO_PLATFORM", "a5")
    mxfp4_to_mxfp8_onboard_cast(
        packed_nk,
        out,
        config=RunConfig(platform=platform, device_id=device_id()),
    )
    mismatch = int((out.view(torch.uint8) != golden.view(torch.uint8)).sum())
    print(f"mismatches: {mismatch} / {K * N}")
    if mismatch:
        first = (out != golden).nonzero(as_tuple=False)[0]
        row = int(first[0])
        col = int(first[1])
        raise RuntimeError(
            f"first mismatch at [{row}, {col}]: got=0x{int(out.view(torch.uint8)[row, col]):02x}, "
            f"expected=0x{int(golden.view(torch.uint8)[row, col]):02x}"
        )
    print("OK: packed MXFP4 was cast and transposed on device into [K, N]")


if __name__ == "__main__":
    main()
