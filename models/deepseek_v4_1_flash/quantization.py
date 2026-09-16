# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch references for the checkpoint and persistent-cache MX formats."""

import torch

from models.deepseek_v4_1_flash.config import MX_GROUP

# Match decode_c2a_full vector E8M0 expand: exp((code-127)*ln2), not exact exp2.
_LN2 = 0.6931471805599453
_INV_LN2 = 1.4426950408889634


FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# FP4 nibble → FP8 E4M3 codes (same table as V4 Pro utils / PR #1210).
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


def build_mxfp4_pair_lut() -> torch.Tensor:
    """Packed-byte LUT whose INT16 entries contain two FP8 payloads (PR #1210)."""
    packed = torch.arange(256, dtype=torch.int64)
    fp8_codes = torch.tensor(NIBBLE_LUT, dtype=torch.int64)
    low_codes = fp8_codes[packed & 0x0F]
    high_codes = fp8_codes[packed >> 4]
    pairs_u16 = low_codes | (high_codes << 8)
    pairs_i16 = torch.where(pairs_u16 < 0x8000, pairs_u16, pairs_u16 - 0x10000)
    return pairs_i16.to(torch.int16).reshape(1, 256).repeat(2, 1).contiguous()


def decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 storage codes into exact FP32 powers of two.

    Cube ``matmul_mx`` applies E8M0 as a true power of two.  Keep this path
    bit-exact for weight / activation MX math.
    """
    codes = scale.contiguous().view(torch.uint8)
    return torch.exp2(codes.to(torch.float32) - 127.0)


def decode_e8m0_vector(scale: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 the way A5 vector kernels do: ``exp((code-127)*ln2)``.

    Window and index cache expand cannot use a native exp2, so they multiply
    by this decimal ``ln2`` constant.  The result can be 1 FP32 ULP off an
    exact power of two and then round into a neighboring BF16 bin.
    """
    codes = scale.contiguous().view(torch.uint8).to(torch.float32)
    return torch.exp((codes - 127.0) * _LN2)


def encode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Encode positive FP32 powers of two as UE8M0 storage codes."""
    exponent = torch.round(torch.log2(scale.float())).clamp(-127, 128)
    return (exponent + 127).to(torch.uint8)


def pack_mx_b_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pack logical ``[..., K/32, N]`` scales into the MX_B_NN physical order."""
    *leading, k_groups, output_dim = scale.shape
    if k_groups % 2 or output_dim % 16:
        raise ValueError("MX_B_NN requires an even K-group count and an output dimension divisible by 16")
    packed = scale.reshape(*leading, k_groups // 2, 2, output_dim // 16, 16)
    leading_axes = list(range(len(leading)))
    packed = packed.permute(*leading_axes, len(leading) + 2, len(leading), len(leading) + 3, len(leading) + 1)
    return packed.contiguous().reshape(*leading, k_groups, output_dim)


def unpack_mx_b_scale(scale: torch.Tensor) -> torch.Tensor:
    """Unpack the Cube MX_B_NN scale order into logical ``[..., K/32, N]`` rows."""
    *leading, k_groups, output_dim = scale.shape
    if k_groups % 2 or output_dim % 16:
        raise ValueError("MX_B_NN requires an even K-group count and an output dimension divisible by 16")
    logical = scale.reshape(*leading, output_dim // 16, k_groups // 2, 16, 2)
    leading_axes = list(range(len(leading)))
    logical = logical.permute(
        *leading_axes, len(leading) + 1, len(leading) + 3, len(leading), len(leading) + 2
    )
    return logical.contiguous().reshape(*leading, k_groups, output_dim)


def dequantize_mxfp4(packed_weight: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Decode checkpoint MXFP4 directly into an FP32 output-major matrix."""
    packed = packed_weight.contiguous().view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    indices = torch.stack((low, high), dim=-1).flatten(-2)
    values = FP4_VALUES.to(indices.device)[indices.to(torch.long)]
    scales = decode_e8m0(scale_e8m0).repeat_interleave(MX_GROUP, dim=-1)
    return values * scales


def _nearest_fp4_indices(values: torch.Tensor) -> torch.Tensor:
    table = FP4_VALUES[:8].to(values.device)
    magnitude = values.abs().unsqueeze(-1)
    distances = (magnitude - table).abs()
    min_distance = distances.amin(dim=-1, keepdim=True)
    ties = distances == min_distance
    codes = torch.arange(8, device=values.device, dtype=torch.int64)
    even_tie = torch.where(ties & ((codes % 2) == 0), codes, 8).amin(dim=-1)
    nearest = distances.argmin(dim=-1)
    # A5 TCVT(CAST_RINT) resolves exact E2M1 midpoints to the even magnitude
    # code.  argmin alone always picked the lower code and disagreed at half
    # steps such as 0.25, 1.25, and 2.5.
    index = torch.where(even_tie < 8, even_tie, nearest).to(torch.uint8)
    # Native A5 BF16→FP4 canonicalizes an exact -0 input to +0, but retains
    # the sign when a non-zero negative value rounds/underflows to FP4 zero.
    sign = torch.signbit(values).to(torch.uint8) & (values != 0).to(torch.uint8)
    return index | (sign << 3)


def _pack_fp4(indices: torch.Tensor) -> torch.Tensor:
    if indices.shape[-1] % 2:
        raise ValueError("packed FP4 requires an even logical last dimension")
    pairs = indices.unflatten(-1, (-1, 2))
    return pairs[..., 0] | (pairs[..., 1] << 4)


def _unpack_fp4(payload: torch.Tensor) -> torch.Tensor:
    packed = payload.contiguous().view(torch.uint8)
    indices = torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1).flatten(-2)
    return FP4_VALUES.to(payload.device)[indices.to(torch.long)]


def quantize_mxfp4_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize output-major ``[..., N, K]`` weights to checkpoint MXFP4 carriers."""
    if weight.shape[-1] % MX_GROUP:
        raise ValueError("MXFP4 weights require K divisible by 32")
    grouped = weight.float().unflatten(-1, (-1, MX_GROUP))
    amax = grouped.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2((amax / 6.0).clamp_min(2.0**-127)))
    scale = torch.exp2(exponent.clamp(-127, 128))
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    payload = _pack_fp4(_nearest_fp4_indices(normalized).flatten(-2))
    return payload, encode_e8m0(scale)


def quantize_mxfp8_cache(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension with the A5 OCP MXFP8 TQUANT contract."""
    if value.shape[-1] % MX_GROUP:
        raise ValueError("MXFP8 cache width must be divisible by 32")
    grouped = value.float().unflatten(-1, (-1, MX_GROUP))
    amax = grouped.abs().amax(dim=-1)
    # A5 OCP TQUANT does not compute ceil(log2(amax / 448)).  It extracts the
    # FP32 exponent field and subtracts E4M3's emax (8), so values in the top
    # part of an exponent bin intentionally saturate to +/-448.  Work from the
    # bits here as well to preserve the exact zero/subnormal clamp semantics.
    biased_exp = (amax.contiguous().view(torch.int32) >> 23) & 0xFF
    shared_exp = torch.where(biased_exp <= 8, torch.zeros_like(biased_exp), biased_exp - 8)
    scale = torch.exp2(shared_exp.to(torch.float32) - 127.0)
    payload = (grouped / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    return payload.flatten(-2).to(torch.float8_e4m3fn), shared_exp.to(torch.uint8)


def dequantize_mxfp8_cache(payload: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Decode an MXFP8 cache tensor whose groups lie on the last dimension."""
    scales = decode_e8m0_vector(scale_e8m0).repeat_interleave(MX_GROUP, dim=-1)
    return payload.float() * scales


# AscendC kv_compress_epilog_v2 ``mxfp4_bf16`` shared-exp bit constants (BF16).
_BF16_EXP_MASK = 0x7F80
_BF16_INV_BIAS = 0x7F00
_FP4_E2M1_MAX_EXP = 0x0100
_FP4_SPECIAL_INV = 0x0040
_BF16_NAN = 0x7FC0


def _mxfp4_bf16_shared_exp_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match epilog_v2 group16 scale / halfScale from per-group amax.

    ``scale`` is stored BF16; ``half_scale`` is the reciprocal used before
    Cast(BF16→FP4). Finite positive amax uses exponent-field arithmetic on the
    BF16 bit pattern (not continuous ``amax/6``).
    """
    amax_bf16 = amax.to(torch.bfloat16)
    bits = amax_bf16.view(torch.int16).to(torch.int32) & 0xFFFF
    max_exp = bits & _BF16_EXP_MASK
    finite = max_exp != _BF16_EXP_MASK
    nonzero = max_exp != 0
    # AscendC Select(dst, src0, src1, mask): mask ? src0 : src1
    # Compare LE → if max_exp <= FP4_MAX_EXP then take FP4_MAX_EXP else keep.
    max_exp = torch.where(max_exp <= _FP4_E2M1_MAX_EXP, torch.full_like(max_exp, _FP4_E2M1_MAX_EXP), max_exp)
    shared_exp = max_exp - _FP4_E2M1_MAX_EXP
    scale_bits = torch.where(finite, shared_exp, torch.full_like(shared_exp, _BF16_NAN))
    scale_bits = torch.where(nonzero, scale_bits, torch.zeros_like(scale_bits))
    half_bits = _BF16_INV_BIAS - shared_exp
    half_bits = torch.where(finite, half_bits, torch.full_like(half_bits, _BF16_NAN))
    half_bits = torch.where(nonzero, half_bits, torch.zeros_like(half_bits))
    half_bits = torch.where(shared_exp == _BF16_INV_BIAS, torch.full_like(half_bits, _FP4_SPECIAL_INV), half_bits)
    scale = scale_bits.to(torch.int16).view(torch.bfloat16)
    half_scale = half_bits.to(torch.int16).view(torch.bfloat16)
    return scale, half_scale


def quantize_mxfp4_cache(
    value: torch.Tensor, group_size: int, scale_format: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to packed E2M1 with BF16 / E8M0 / E4M3 scales.

    ``bf16`` matches A5 ``kv_compress_epilog_v2`` mode ``mxfp4_bf16`` (shared-exp
    BF16 scales). Compressed KV golden uses this format with ``group_size=16``.
    """
    if value.shape[-1] % group_size:
        raise ValueError(f"MXFP4 cache width must be divisible by {group_size}")
    grouped = value.float().unflatten(-1, (-1, group_size))
    amax = grouped.abs().amax(dim=-1)
    if scale_format == "bf16":
        stored_scale, half_scale = _mxfp4_bf16_shared_exp_scale(amax)
        # Device path materializes BF16 after Mul(x, halfScale), before the
        # native BF16→FP4 TCVT.  Preserve that intermediate rounding so exact
        # E2M1 midpoint decisions match the instruction.
        normalized = (
            grouped.mul(half_scale.float().unsqueeze(-1))
            .to(torch.bfloat16)
            .float()
            .clamp(-6.0, 6.0)
        )
    elif scale_format == "e8m0":
        # Index-cache write: ceil(log(amax/6)*INV_LN2) then exp(-e*ln2), matching
        # ``_mxfp4_quant_idx_rows`` rather than IEEE log2/exp2.
        raw_scale = (amax / 6.0).clamp_min(2.0**-9)
        log2_scale = torch.log(raw_scale.float()) * _INV_LN2
        exponent = torch.ceil(log2_scale.clamp(-127.0, 128.0))
        stored_scale = (exponent + 127.0).to(torch.uint8)
        reciprocal = torch.exp(exponent * (-_LN2))
        normalized = (
            grouped.mul(reciprocal.unsqueeze(-1))
            .to(torch.bfloat16)
            .float()
            .clamp(-6.0, 6.0)
        )
    elif scale_format == "e4m3":
        # Legacy cmp ABI; prefer ``bf16`` for A5 mainline alignment.
        raw_scale = (amax / 6.0).clamp_min(2.0**-9)
        stored_scale = raw_scale.clamp(max=448.0).to(torch.float8_e4m3fn)
        scale_value = stored_scale.float()
        normalized = (grouped / scale_value.unsqueeze(-1)).clamp(-6.0, 6.0)
    else:
        raise ValueError(f"unsupported MXFP4 cache scale format {scale_format!r}")
    payload = _pack_fp4(_nearest_fp4_indices(normalized).flatten(-2))
    return payload, stored_scale


def dequantize_mxfp4_cache(
    payload: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    scale_format: str,
) -> torch.Tensor:
    """Decode a packed E2M1 cache tensor with last-dimension MX groups."""
    if scale_format == "e8m0":
        scale_value = decode_e8m0_vector(scale)
    elif scale_format == "bf16":
        scale_value = scale.float()
    elif scale_format == "e4m3":
        scale_value = scale.float()
    else:
        raise ValueError(f"unsupported MXFP4 cache scale format {scale_format!r}")
    scales = scale_value.repeat_interleave(group_size, dim=-1)
    return _unpack_fp4(payload) * scales


def dequantize_mxfp8(weight: torch.Tensor, logical_scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Decode an input-major MXFP8 matrix using logical ``[K/32, N]`` scales."""
    scales = decode_e8m0(logical_scale_e8m0).repeat_interleave(MX_GROUP, dim=-2)
    return weight.float() * scales


def _quantize_mxfp8_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *leading, width = x.shape
    if width % MX_GROUP:
        raise ValueError("MXFP8 activations require the last dimension to be divisible by 32")
    grouped = x.float().reshape(*leading, width // MX_GROUP, MX_GROUP)
    amax = grouped.abs().amax(dim=-1)
    biased_exp = (amax.contiguous().view(torch.int32) >> 23) & 0xFF
    shared_exp = torch.where(biased_exp <= 8, torch.zeros_like(biased_exp), biased_exp - 8)
    scale = torch.exp2(shared_exp.to(torch.float32) - 127.0)
    quantized = (grouped / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale


def quantize_mxfp8_activation(x: torch.Tensor) -> torch.Tensor:
    """Round one activation scale per row and group of 32, then dequantize to FP32."""
    quantized, scale = _quantize_mxfp8_activation(x)
    return (quantized.float() * scale.unsqueeze(-1)).reshape_as(x).float()


def mxfp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed_weight_scale: torch.Tensor | None,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate a native or dynamically quantized input-major linear projection.

    ``output_dtype`` lets fused references retain the Cube FP32 accumulator
    through a following normalization.  By default the historical behavior is
    preserved and quantized projections are restored to the activation dtype.
    """
    if packed_weight_scale is None:
        result = torch.matmul(x, weight)
        return result if output_dtype is None else result.to(output_dtype)
    activation, activation_scale = _quantize_mxfp8_activation(x)
    logical_scale = unpack_mx_b_scale(packed_weight_scale)
    weight_scale = decode_e8m0(logical_scale)
    weight_groups = weight.float().unflatten(0, (-1, MX_GROUP))
    partials = torch.einsum("...gk,gkn->...gn", activation.float(), weight_groups)
    partials = partials * activation_scale.unsqueeze(-1) * weight_scale
    result = partials.sum(dim=-2)
    return result.to(x.dtype if output_dtype is None else output_dtype)
