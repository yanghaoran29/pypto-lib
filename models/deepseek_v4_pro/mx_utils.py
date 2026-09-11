# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MXFP4/MXFP8 data preparation for DeepSeek-V4-Pro MoE experts.

The routed-expert experiment keeps W1/W3/W2 weights packed in a board-ready,
tile-major layout. Each matmul expands only its current Cube tile on device
through a packed-byte lookup table. E8M0 scales remain unchanged, and
activations are quantized on device with ``pl.quant_mx``.
"""

from __future__ import annotations

import math

MX_GROUP = 32
SCALE_BLOCK_SIZE = 16
SCALE_C0_SIZE = 2
FP8_E4M3_MAX = 448.0
FP4_MAX = 6.0
TINY = 1e-20

# Precomputed FP4 nibble -> FP8 E4M3 codes (Issue #238 MXFP4 magnitude table).
NIBBLE_LUT = [
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


def build_mxfp4_pair_lut():
    """Build a packed-byte LUT whose INT16 entries contain two FP8 payloads."""
    import torch

    packed = torch.arange(256, dtype=torch.int64)
    fp8_codes = torch.tensor(NIBBLE_LUT, dtype=torch.int64)
    low_codes = fp8_codes[packed & 0x0F]
    high_codes = fp8_codes[packed >> 4]
    pairs_u16 = low_codes | (high_codes << 8)
    pairs_i16 = torch.where(pairs_u16 < 0x8000, pairs_u16, pairs_u16 - 0x10000)
    return pairs_i16.to(torch.int16).reshape(1, 256).repeat(2, 1).contiguous()


def _pack_mxfp4_nibbles_kn_tiles(nibbles_kn, k_tile, n_tile, split_mode):
    """Pack one ``[K, N]`` nibble grid into tile-major AIV lane rows."""
    k, n = nibbles_kn.shape
    if k % k_tile != 0 or n % n_tile != 0:
        raise ValueError("MXFP4 tile packing requires K and N to divide their tiles")
    if k_tile % 2 != 0 or n_tile % 2 != 0:
        raise ValueError("MXFP4 tile packing requires even K and N tiles")

    k_blocks = k // k_tile
    n_blocks = n // n_tile
    blocked = nibbles_kn.reshape(k_blocks, k_tile, n_blocks, n_tile)
    blocked = blocked.permute(2, 0, 1, 3)
    if split_mode == "up_down":
        lanes = blocked.reshape(n_blocks, k_blocks, 2, k_tile // 2, n_tile)
    elif split_mode == "left_right":
        if n_tile % 4 != 0:
            raise ValueError("left-right MXFP4 packing requires N tile divisible by four")
        lanes = blocked.reshape(n_blocks, k_blocks, k_tile, 2, n_tile // 2)
        lanes = lanes.permute(0, 1, 3, 2, 4)
    else:
        raise ValueError(f"unsupported MXFP4 split mode: {split_mode!r}")

    low = lanes[..., 0::2] & 0x0F
    high = lanes[..., 1::2] & 0x0F
    packed = low | (high << 4)
    lane_bytes = k_tile * n_tile // 4
    return packed.contiguous().reshape(n_blocks * k_blocks * 2, lane_bytes)


def _mxfp8_grid_to_mxfp4_nibbles(weight):
    """Map an FP8 grid restricted to the MXFP4 value set back to nibble codes."""
    import torch

    codes = weight.contiguous().view(torch.uint8)
    inverse = torch.zeros(256, dtype=torch.uint8)
    valid = torch.zeros(256, dtype=torch.bool)
    lut_codes = torch.tensor(NIBBLE_LUT, dtype=torch.uint8)
    lut_indices = lut_codes.to(torch.int64)
    inverse[lut_indices] = torch.arange(16, dtype=torch.uint8)
    valid[lut_indices] = True
    code_indices = codes.to(torch.int64)
    if not bool(valid[code_indices].all()):
        raise ValueError("MXFP8 fixture contains a code outside the MXFP4 LUT")
    return inverse[code_indices]


def pack_mxfp4_weight_tiles(weight, k_tile, n_tile, split_mode="up_down"):
    """Pack ``[..., K, N]`` MXFP4-grid FP8 weights into tile-major lane rows."""
    import torch

    *leading, k, n = weight.shape
    lane_bytes = k_tile * n_tile // 4
    tile_rows = (k // k_tile) * (n // n_tile) * 2
    weights = weight.reshape(-1, k, n)
    packed = torch.empty(weights.shape[0], tile_rows, lane_bytes, dtype=torch.uint8)
    for batch in range(weights.shape[0]):
        nibbles_kn = _mxfp8_grid_to_mxfp4_nibbles(weights[batch])
        packed[batch] = _pack_mxfp4_nibbles_kn_tiles(
            nibbles_kn,
            k_tile,
            n_tile,
            split_mode,
        )
    return packed.reshape(*leading, tile_rows, lane_bytes)


def repack_mxfp4_checkpoint_to_tiles(
    weight_packed,
    k_tile,
    n_tile,
    split_mode="up_down",
):
    """Repack ``[..., N, K/2]`` checkpoint bytes into tile-major lane rows."""
    import torch

    packed_nk = weight_packed.contiguous().view(torch.uint8)
    *leading, n, half_k = packed_nk.shape
    k = half_k * 2
    lane_bytes = k_tile * n_tile // 4
    tile_rows = (k // k_tile) * (n // n_tile) * 2
    checkpoint = packed_nk.reshape(-1, n, half_k)
    packed = torch.empty(checkpoint.shape[0], tile_rows, lane_bytes, dtype=torch.uint8)
    for batch in range(checkpoint.shape[0]):
        low_k = checkpoint[batch] & 0x0F
        high_k = checkpoint[batch] >> 4
        nibbles_nk = torch.stack((low_k, high_k), dim=-1).reshape(n, k)
        nibbles_kn = nibbles_nk.transpose(0, 1).contiguous()
        packed[batch] = _pack_mxfp4_nibbles_kn_tiles(
            nibbles_kn,
            k_tile,
            n_tile,
            split_mode,
        )
    return packed.reshape(*leading, tile_rows, lane_bytes)


def repack_mxfp4_kn_pairs_to_tiles(
    weight_packed,
    k_tile,
    n_tile,
    split_mode="up_down",
):
    """Repack ``[..., K, N/2]`` pair bytes into tile-major lane rows."""
    import torch

    packed_kn = weight_packed.contiguous().view(torch.uint8)
    *leading, k, half_n = packed_kn.shape
    n = half_n * 2
    lane_bytes = k_tile * n_tile // 4
    tile_rows = (k // k_tile) * (n // n_tile) * 2
    pairs = packed_kn.reshape(-1, k, half_n)
    packed = torch.empty(pairs.shape[0], tile_rows, lane_bytes, dtype=torch.uint8)
    for batch in range(pairs.shape[0]):
        low_n = pairs[batch] & 0x0F
        high_n = pairs[batch] >> 4
        nibbles_kn = torch.stack((low_n, high_n), dim=-1).reshape(k, n)
        packed[batch] = _pack_mxfp4_nibbles_kn_tiles(
            nibbles_kn,
            k_tile,
            n_tile,
            split_mode,
        )
    return packed.reshape(*leading, tile_rows, lane_bytes)


def unpack_mxfp4_weight_tiles(
    packed,
    k,
    n,
    k_tile,
    n_tile,
    split_mode="up_down",
):
    """Expand tile-major lane rows into FP8 ``[..., K, N]`` weights."""
    import torch

    *leading, tile_rows, lane_bytes = packed.shape
    k_blocks = k // k_tile
    n_blocks = n // n_tile
    expected_rows = n_blocks * k_blocks * 2
    expected_bytes = k_tile * n_tile // 4
    if tile_rows != expected_rows or lane_bytes != expected_bytes:
        raise ValueError("MXFP4 tile-major payload does not match the requested matrix")

    payloads = packed.contiguous().view(torch.uint8).reshape(-1, tile_rows, lane_bytes)
    output = torch.empty(payloads.shape[0], k, n, dtype=torch.float8_e4m3fn)
    for batch in range(payloads.shape[0]):
        if split_mode == "up_down":
            lane_shape = (n_blocks, k_blocks, 2, k_tile // 2, n_tile // 2)
            lane_bytes_view = payloads[batch].reshape(lane_shape)
            low = lane_bytes_view & 0x0F
            high = lane_bytes_view >> 4
            lanes = torch.stack((low, high), dim=-1)
            blocked = lanes.reshape(n_blocks, k_blocks, k_tile, n_tile)
        elif split_mode == "left_right":
            lane_shape = (n_blocks, k_blocks, 2, k_tile, n_tile // 4)
            lane_bytes_view = payloads[batch].reshape(lane_shape)
            low = lane_bytes_view & 0x0F
            high = lane_bytes_view >> 4
            lanes = torch.stack((low, high), dim=-1)
            lanes = lanes.reshape(n_blocks, k_blocks, 2, k_tile, n_tile // 2)
            blocked = lanes.permute(0, 1, 3, 2, 4).reshape(
                n_blocks,
                k_blocks,
                k_tile,
                n_tile,
            )
        else:
            raise ValueError(f"unsupported MXFP4 split mode: {split_mode!r}")
        nibbles_kn = blocked.permute(1, 2, 0, 3).reshape(k, n)
        output[batch] = nibble_indices_to_fp8(nibbles_kn)
    return output.reshape(*leading, k, n)


def repack_mxfp4_checkpoint_to_kn_pairs(weight_packed):
    """Repack ``[..., N, K/2]`` checkpoint bytes as ``[..., K, N/2]`` bytes."""
    import torch

    packed_nk = weight_packed.contiguous().view(torch.uint8)
    *leading, n, half_k = packed_nk.shape
    if n % 2 != 0:
        raise ValueError("MXFP4 pair packing requires an even N dimension")

    low_k = packed_nk & 0x0F
    high_k = packed_nk >> 4
    nibbles_nk = torch.stack((low_k, high_k), dim=-1).reshape(
        *leading,
        n,
        half_k * 2,
    )
    nibbles_kn = nibbles_nk.transpose(-2, -1).contiguous()
    return (
        nibbles_kn[..., 0::2]
        | (nibbles_kn[..., 1::2] << 4)
    ).contiguous()


def pack_mxfp4_weight_kn_pairs(weight):
    """Pack LUT-representable FP8 ``[..., K, N]`` weights along N as MXFP4 bytes."""
    import torch

    codes = weight.contiguous().view(torch.uint8)
    if codes.shape[-1] % 2 != 0:
        raise ValueError("MXFP4 pair packing requires an even N dimension")

    nibbles = _mxfp8_grid_to_mxfp4_nibbles(weight)
    return (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).contiguous()


def unpack_mxfp4_weight_kn_pairs(packed):
    """Expand packed ``[..., K, N/2]`` MXFP4 bytes into FP8 ``[..., K, N]``."""
    import torch

    packed_u8 = packed.contiguous().view(torch.uint8)
    low = packed_u8 & 0x0F
    high = packed_u8 >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(
        *packed_u8.shape[:-1], packed_u8.shape[-1] * 2
    )
    return nibble_indices_to_fp8(nibbles)


def pack_a_scale(scale_codes):
    """Pack logical A scales ``[M, K/32]`` into the MX_A_ZZ physical layout."""
    m, k_groups = scale_codes.shape
    assert m % SCALE_BLOCK_SIZE == 0
    assert k_groups % SCALE_C0_SIZE == 0
    return (
        scale_codes.reshape(
            m // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def unpack_a_scale(packed_codes):
    """Restore MX_A_ZZ physical scale bytes to logical ``[M, K/32]``."""
    m, k_groups = packed_codes.shape
    return (
        packed_codes.reshape(
            m // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def pack_b_scale(scale_codes):
    """Pack logical B scales ``[K/32, N]`` into the MX_B_NN physical layout."""
    k_groups, n = scale_codes.shape
    assert k_groups % SCALE_C0_SIZE == 0
    assert n % SCALE_BLOCK_SIZE == 0
    return (
        scale_codes.reshape(
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
            n // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
        )
        .permute(2, 0, 3, 1)
        .contiguous()
        .reshape(k_groups, n)
    )


def pack_b_scale_batched(scale_codes):
    """Pack ``[..., K/32, N]`` E8M0 codes as independent MX_B_NN matrices."""
    *lead, k_groups, n = scale_codes.shape
    assert k_groups % SCALE_C0_SIZE == 0
    assert n % SCALE_BLOCK_SIZE == 0
    lead_axes = list(range(len(lead)))
    base = len(lead)
    return (
        scale_codes.reshape(
            *lead,
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
            n // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
        )
        .permute(*lead_axes, base + 2, base, base + 3, base + 1)
        .contiguous()
        .reshape(*lead, k_groups, n)
    )


def unpack_b_scale(packed_codes):
    """Restore MX_B_NN physical scale bytes to logical ``[K/32, N]``."""
    k_groups, n = packed_codes.shape
    return (
        packed_codes.reshape(
            n // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(1, 3, 0, 2)
        .contiguous()
        .reshape(k_groups, n)
    )


def unpack_b_scale_batched(packed_codes):
    """Restore independently packed ``[..., K/32, N]`` MX_B_NN matrices."""
    *lead, k_groups, n = packed_codes.shape
    lead_axes = list(range(len(lead)))
    base = len(lead)
    return (
        packed_codes.reshape(
            *lead,
            n // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(*lead_axes, base + 1, base + 3, base, base + 2)
        .contiguous()
        .reshape(*lead, k_groups, n)
    )


def _e8m0_codes_from_amax(amax, fp_max: float):
    """Ascend OCP shared-exponent E8M0 codes for each group maximum."""
    import torch

    format_emax = int(math.floor(math.log2(fp_max)))
    _, exponent = torch.frexp(amax)
    codes = exponent.to(torch.int32) - 1 - format_emax + 127
    codes = codes.clamp(0, 255)
    codes = torch.where(amax == 0, torch.zeros_like(codes), codes)
    return codes.to(torch.uint8)


def e8m0_codes_to_fp32(codes):
    """Decode logical E8M0 uint8 codes to FP32 powers of two."""
    import torch

    return torch.exp2(codes.to(torch.float32) - 127.0)


def host_quant_mxfp8(x_bf16_or_fp32, *, pack_zz: bool = False, return_e8m0: bool = False):
    """Per-row group-32 MXFP8 quant along the last dim.

    Returns ``(data_fp8, scale)`` with logical shapes ``[..., K]`` /
    ``[..., K/32]``. By default ``scale`` is **decoded FP32** for the kernel
    ABI. Pass ``return_e8m0=True`` for packed/logical E8M0 codes (docs /
    fixtures). ``pack_zz`` only applies when returning E8M0 and the leading
    row dim is a multiple of 16 with even K/32.
    """
    import torch

    x = x_bf16_or_fp32.float()
    *lead, k = x.shape
    assert k % MX_GROUP == 0
    groups = k // MX_GROUP
    xg = x.reshape(*lead, groups, MX_GROUP)
    amax = xg.abs().amax(dim=-1)
    codes = _e8m0_codes_from_amax(amax, FP8_E4M3_MAX)
    scale_f = e8m0_codes_to_fp32(codes)
    q = (xg / scale_f.unsqueeze(-1)).to(torch.float8_e4m3fn)
    data = q.reshape(*lead, k)
    if not return_e8m0:
        return data, scale_f.contiguous()

    scale = codes
    if pack_zz and len(lead) == 1:
        m = lead[0]
        if m % SCALE_BLOCK_SIZE == 0 and groups % SCALE_C0_SIZE == 0:
            scale = pack_a_scale(scale.reshape(m, groups)).reshape(m, groups)
    scale_e8m0 = scale.contiguous().view(torch.float8_e8m0fnu)
    return data, scale_e8m0


def gen_mxfp8_weight_kn(out: int, inn: int, dequant_std: float, *, chan_cv: float = 0.5, seed: int = 0):
    """Simulate an MXFP8 weight grid in Cube ``[K, N] = [inn, out]`` layout.

    Returns FP8 data ``[inn, out]`` and **decoded FP32** logical scales
    ``[inn/32, out]`` (kernel ABI). Does **not** requantize to INT8.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    weight_base = torch.randn(out, inn, generator=g)
    channel_noise = torch.randn(out, 1, generator=g)
    channel_gain = torch.exp(chan_cv * channel_noise)
    w = weight_base * channel_gain  # [out, inn]
    assert inn % MX_GROUP == 0
    wg = w.reshape(out, inn // MX_GROUP, MX_GROUP)
    amax = wg.abs().amax(dim=-1)
    codes_on = _e8m0_codes_from_amax(amax, FP8_E4M3_MAX)  # [out, inn/32]
    scale_f = e8m0_codes_to_fp32(codes_on)
    q = (wg / scale_f.unsqueeze(-1)).to(torch.float8_e4m3fn)
    data_on = q.reshape(out, inn)
    data_kn = data_on.transpose(0, 1).contiguous()  # [inn, out]
    codes_kn = codes_on.transpose(0, 1).contiguous()
    decoded = data_kn.float() * e8m0_codes_to_fp32(codes_kn).repeat_interleave(MX_GROUP, dim=0)
    cur_std = decoded.std().clamp_min(TINY)
    gain = dequant_std / cur_std
    exp_shift = int(round(math.log2(float(gain))))
    codes_kn = (codes_kn.to(torch.int32) + exp_shift).clamp(0, 255).to(torch.uint8)
    scale_fp32 = e8m0_codes_to_fp32(codes_kn).contiguous()
    return data_kn.to(torch.float8_e4m3fn), scale_fp32


def gen_mxfp8_weight_kn_device(out, inn, dequant_std, *, chan_cv=0.5, seed=0):
    """Device payloads for Cube ``[K, N] = [inn, out]`` MX matmul rhs.

    Returns FP8 data ``[inn, out]`` and MX_B_NN-packed E8M0 scale
    ``[inn // 32, out]``.
    """
    import torch

    data_kn, scale_fp32 = gen_mxfp8_weight_kn(out, inn, dequant_std, chan_cv=chan_cv, seed=seed)
    codes = (
        (torch.log2(scale_fp32.clamp_min(TINY)) + 127.0)
        .round()
        .to(torch.int32)
        .clamp(0, 255)
        .to(torch.uint8)
    )
    scale_e8m0 = pack_b_scale(codes).contiguous().view(torch.float8_e8m0fnu)
    return data_kn, scale_e8m0


def host_mxfp8_activation(x_bf16_or_fp32):
    """MXFP8 activation + MX_A_ZZ E8M0 scales for device ``matmul_mx`` lhs."""
    return host_quant_mxfp8(x_bf16_or_fp32, pack_zz=True, return_e8m0=True)


def host_quant_mxfp8_weight_kn(weight_nk):
    """Quantize ``[..., N, K]`` weights for the Cube MXFP8 rhs ABI.

    Returns FP8 data in ``[..., K, N]`` and E8M0 group scales in the packed
    ``MX_B_NN`` physical layout ``[..., K/32, N]``.
    """
    import torch

    data_nk, scale_ng = host_quant_mxfp8(weight_nk, return_e8m0=True)
    data_kn = data_nk.transpose(-2, -1).contiguous()
    codes_kn = scale_ng.contiguous().view(torch.uint8).transpose(-2, -1).contiguous()
    scale_nn = pack_b_scale_batched(codes_kn).view(torch.float8_e8m0fnu)
    return data_kn, scale_nn


def gen_mxfp4_weight_kn(out: int, inn: int, dequant_std: float, *, seed: int = 0):
    """Simulate MXFP4 (e2m1 + per-32 E8M0) in Cube ``[inn, out]`` layout.

    Returns packed FP4 bytes ``[inn/2, out]``, **decoded FP32** scales
    ``[inn/32, out]``, and INT16 nibble-index tensor ``[inn, out]`` ready for
    host/device LUT gather → FP8E4M3FN.
    """
    import torch

    FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    FP4_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inn, generator=g)  # [out, inn]
    assert inn % MX_GROUP == 0
    wg = w.reshape(out, inn // MX_GROUP, MX_GROUP)
    absw = wg.abs()
    codes_on = _e8m0_codes_from_amax(absw.amax(dim=-1), FP4_MAX)  # [out, inn/32]
    scale_f = e8m0_codes_to_fp32(codes_on)
    idx = torch.bucketize(absw / scale_f.unsqueeze(-1), FP4_MID).clamp_max(7)
    sign = (wg < 0).to(torch.int64)
    nibble = idx + sign * 8  # 0..15
    nibble_flat = nibble.reshape(out, inn)

    values = torch.sign(wg) * FP4_MAG[idx]
    decoded_on = (values * scale_f.unsqueeze(-1)).reshape(out, inn)
    cur_std = decoded_on.std().clamp_min(TINY)
    gain = dequant_std / cur_std
    exp_shift = int(round(math.log2(float(gain))))
    codes_on = (codes_on.to(torch.int32) + exp_shift).clamp(0, 255).to(torch.uint8)

    nibble_kn = nibble_flat.transpose(0, 1).contiguous()  # [inn, out]
    codes_kn = codes_on.transpose(0, 1).contiguous()  # [inn/32, out]

    assert inn % 2 == 0
    lo = nibble_kn[0::2, :] & 0x0F
    hi = nibble_kn[1::2, :] & 0x0F
    packed = (lo | (hi << 4)).to(torch.uint8).contiguous()  # [inn/2, out]

    indices = nibble_kn.to(torch.int16)
    scale_fp32 = e8m0_codes_to_fp32(codes_kn).contiguous()
    return packed, scale_fp32, indices


def fp4_packed_to_nibble_indices(packed):
    """Unpack FP4 bytes ``[inn/2, out]`` to INT16 nibble indices ``[inn, out]``.

    ``gen_mxfp4_weight_kn`` packs adjacent K rows on axis 0 (lo then hi nibble).
    """
    import torch

    packed_u8 = packed.contiguous().view(torch.uint8)
    half, *rest = packed_u8.shape
    lo = (packed_u8 & 0x0F).to(torch.int16)
    hi = ((packed_u8 >> 4) & 0x0F).to(torch.int16)
    out = torch.empty(half * 2, *rest, dtype=torch.int16)
    out[0::2, ...] = lo
    out[1::2, ...] = hi
    return out


def nibble_indices_to_fp8(indices):
    """Host LUT: INT16 nibble indices → FP8E4M3FN payload (same codes as device gather)."""
    import torch

    lut = torch.tensor(NIBBLE_LUT, dtype=torch.int16)
    codes = lut[indices.to(torch.int64).clamp(0, 15)]
    return (codes & 0xFF).to(torch.uint8).view(torch.float8_e4m3fn)


def mxfp4_to_mxfp8_weight_kn(weight_packed, scale_e8m0):
    """Expand a checkpoint MXFP4 weight into the current Cube MXFP8 ABI.

    ``weight_packed`` is ``[..., N, K/2]`` with adjacent K values stored low
    nibble first. ``scale_e8m0`` is ``[..., N, K/32]``.  The result is exact
    FP8E4M3 data ``[..., K, N]`` plus the unchanged E8M0 codes transposed and
    packed for ``MX_B_NN`` as ``[..., K/32, N]``.
    """
    import torch

    packed_u8 = weight_packed.contiguous().view(torch.uint8)
    scale_codes = scale_e8m0.contiguous().view(torch.uint8)
    *weight_lead, n, half_k = packed_u8.shape
    *scale_lead, scale_n, k_groups = scale_codes.shape
    if weight_lead != scale_lead or n != scale_n or half_k * 2 != k_groups * MX_GROUP:
        raise ValueError(
            "MXFP4 weight/scale shapes must be [..., N, K/2] and [..., N, K/32], "
            f"got {tuple(weight_packed.shape)} and {tuple(scale_e8m0.shape)}"
        )

    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    indices_nk = torch.stack((low, high), dim=-1).reshape(*weight_lead, n, half_k * 2)
    data_kn = nibble_indices_to_fp8(indices_nk).transpose(-2, -1).contiguous()
    codes_kn = scale_codes.transpose(-2, -1).contiguous()
    packed_codes = pack_b_scale_batched(codes_kn)
    return data_kn, packed_codes.view(torch.float8_e8m0fnu)


def gen_mxfp4_weight_kn_device(out: int, inn: int, dequant_std: float, *, seed: int = 0):
    """Generate checkpoint-shaped MXFP4, then expand it through the real bridge."""
    import torch

    packed_kn, scale_fp32, _ = gen_mxfp4_weight_kn(out, inn, dequant_std, seed=seed)
    codes_kn = (
        (torch.log2(scale_fp32.clamp_min(TINY)) + 127.0)
        .round()
        .to(torch.int32)
        .clamp(0, 255)
        .to(torch.uint8)
    )
    packed_nk = packed_kn.transpose(0, 1).contiguous()
    codes_nk = codes_kn.transpose(0, 1).contiguous().view(torch.float8_e8m0fnu)
    return mxfp4_to_mxfp8_weight_kn(packed_nk, codes_nk)


def matmul_mx_golden(a, a_scale, b, b_scale):
    """FP32 golden for MX-style matmul from data + per-group scales.

    ``a``/``b`` are ``[M,K]`` / ``[K,N]``. Scales are either decoded FP32
    ``[M, K/32]`` / ``[K/32, N]`` or logical E8M0 uint8 codes (auto-detected).
    """
    import torch

    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    a_s = a_scale
    b_s = b_scale
    if a_s.dtype != torch.float32 and a_s.dtype != torch.float64:
        a_s = e8m0_codes_to_fp32(a_s.contiguous().view(torch.uint8))
    if b_s.dtype != torch.float32 and b_s.dtype != torch.float64:
        b_s = e8m0_codes_to_fp32(b_s.contiguous().view(torch.uint8))
    a_s = a_s.to(torch.float64)
    b_s = b_s.to(torch.float64)
    k_group = torch.arange(k) // MX_GROUP
    a_scaled = a.to(torch.float64) * a_s[:, k_group]
    b_scaled = b.to(torch.float64) * b_s[k_group, :]
    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


def decode_e8m0_codes(scale_e8m0, *, side: str = "a"):
    """Unpack ZZ/NN-packed E8M0 tensor to logical uint8 codes."""
    import torch

    codes = scale_e8m0.contiguous().view(torch.uint8)
    if side == "a":
        return unpack_a_scale(codes)
    if side == "b":
        return unpack_b_scale(codes)
    raise ValueError(f"side must be 'a' or 'b', got {side!r}")


def make_nibble_lut_tensor():
    """``[1, 16]`` INT16 LUT tensor for device gather."""
    import torch

    return torch.tensor(NIBBLE_LUT, dtype=torch.int16).view(1, 16).contiguous()
