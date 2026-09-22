# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host FP4E2M1X2 packed-payload helpers for DeepSeek-V4.1-Flash cache / weights.

Type layering (do not conflate)
-------------------------------
* **FP4** — one E2M1 floating value, **4 bits** (half a byte).
  - PyPTO IR: ``pl.FP4`` (codegen ``DataType::FP4E2M1``).
  - Tensor shapes that use ``pl.FP4`` count **logical** FP4 elements; the last
    dimension must be a positive even logical width.
* **FP4E2M1X2** — one **byte** that holds **two** FP4 values (low nibble =
  logical element ``2i``, high nibble = ``2i+1``).
  - Torch host carrier: ``torch.float4_e2m1fn_x2`` (``element_size() == 1``).
  - CANN / AICore packed typedef: ``float4_e2m1x2_t``.
  - Physical last dimension = logical FP4 count / 2.
  - Binding a Torch ``[..., N]`` ``float4_e2m1fn_x2`` tensor into pypto yields
    IR shape ``[..., 2N]`` of ``pl.FP4``.

So: ``pl.FP4`` ≠ ``FP4E2M1X2``. The former is the scalar element; the latter is
the packed storage/carrier used on the host (and historically as ``uint8``
nibbles).

Device kernels in this tree still annotate compressed / index cache payloads as
``pl.UINT8`` with the **physical** (FP4E2M1X2) last dimension. ``pl.reinterpret_view``
does not yet allow FP4↔UINT8, so the nibble publish/decode path cannot treat a
``pl.FP4`` tensor as bytes without on-chip cast (deferred). Host goldens use
``float4_e2m1fn_x2``; harnesses may ``view(torch.uint8)`` at the device boundary
when a kernel still expects the UINT8 annotation.
"""

from __future__ import annotations

import torch

# Element vs packed carrier names (see module docstring).
FP4_ELEMENT_DTYPE_NAME = "FP4"  # pl.FP4 / DataType::FP4E2M1 — 4-bit scalar
FP4E2M1X2_PACKED_NAME = "FP4E2M1X2"  # two FP4 per byte
FP4E2M1X2_TORCH_ATTR = "float4_e2m1fn_x2"
FP4E2M1X2_CANN_TYPENAME = "float4_e2m1x2_t"


def fp4e2m1x2_torch_dtype() -> torch.dtype:
    """Return the packed FP4E2M1X2 Torch dtype (two FP4 / byte)."""
    dtype = getattr(torch, FP4E2M1X2_TORCH_ATTR, None)
    if dtype is None:
        raise RuntimeError(
            "torch.float4_e2m1fn_x2 (FP4E2M1X2 packed carrier) is required for the "
            "V4.1 MXFP4 cache ABI (Torch with float4_e2m1fn_x2, e.g. 2.10+)"
        )
    return dtype


# Back-compat aliases used by earlier call sites in this change set.
fp4_torch_dtype = fp4e2m1x2_torch_dtype


def is_fp4e2m1x2_torch_dtype(dtype: torch.dtype) -> bool:
    """True when ``dtype`` is the packed FP4E2M1X2 Torch carrier (not scalar FP4)."""
    packed = getattr(torch, FP4E2M1X2_TORCH_ATTR, None)
    return packed is not None and dtype is packed


is_fp4_torch_dtype = is_fp4e2m1x2_torch_dtype


def as_fp4e2m1x2_payload(payload: torch.Tensor) -> torch.Tensor:
    """View packed MXFP4 bytes as ``float4_e2m1fn_x2`` (FP4E2M1X2) without copying.

    Accepts ``uint8`` or an existing FP4E2M1X2 carrier. This is **not** a cast
    to scalar ``pl.FP4``; it only reinterprets the one-byte two-nibble packing.
    """
    if is_fp4e2m1x2_torch_dtype(payload.dtype):
        return payload
    if payload.dtype != torch.uint8:
        raise TypeError(
            f"FP4E2M1X2 payload must be uint8 or float4_e2m1fn_x2, got {payload.dtype}"
        )
    return payload.contiguous().view(fp4e2m1x2_torch_dtype())


as_fp4_payload = as_fp4e2m1x2_payload


def as_fp4e2m1x2_uint8(payload: torch.Tensor) -> torch.Tensor:
    """View an FP4E2M1X2 (or uint8) payload as ``uint8`` for nibble / device UINT8 paths."""
    if payload.dtype == torch.uint8:
        return payload
    if not is_fp4e2m1x2_torch_dtype(payload.dtype):
        raise TypeError(
            f"FP4E2M1X2 payload must be uint8 or float4_e2m1fn_x2, got {payload.dtype}"
        )
    return payload.contiguous().view(torch.uint8)


as_fp4_uint8 = as_fp4e2m1x2_uint8


def probe_fp4e2m1x2_host_roundtrip() -> None:
    """CPU probe: uint8 nibble packing is bit-identical under float4_e2m1fn_x2 views.

    Does not exercise device load/store or ``pl.cast``. Run with::

        python -m models.deepseek_v4_1_flash._fp4_abi
    """
    dtype = fp4e2m1x2_torch_dtype()
    # Low nibble = FP4 elem 2i, high nibble = FP4 elem 2i+1 (quantization._pack_fp4).
    packed_u8 = torch.tensor([0x00, 0x1F, 0xA3, 0x70], dtype=torch.uint8)
    as_packed = as_fp4e2m1x2_payload(packed_u8)
    if as_packed.dtype is not dtype or as_packed.shape != packed_u8.shape:
        raise RuntimeError(f"unexpected FP4E2M1X2 view: dtype={as_packed.dtype} shape={as_packed.shape}")
    back = as_fp4e2m1x2_uint8(as_packed)
    if not torch.equal(back, packed_u8):
        raise RuntimeError("FP4E2M1X2 host view is not bit-identical to packed uint8")
    print(
        "[FP4 ABI] PASS host float4_e2m1fn_x2 (FP4E2M1X2) ↔ uint8 bit-identical; "
        f"packed_bytes={packed_u8.numel()} fp4_elems={packed_u8.numel() * 2}; "
        f"element={FP4_ELEMENT_DTYPE_NAME} (4-bit) packed={FP4E2M1X2_PACKED_NAME} "
        f"({FP4E2M1X2_CANN_TYPENAME}); "
        "device pl.FP4 annotation deferred (no FP4↔UINT8 reinterpret_view)"
    )


# Keep the old probe name for ``python -m ..._fp4_abi`` / docs that mentioned it.
probe_fp4_host_roundtrip = probe_fp4e2m1x2_host_roundtrip


if __name__ == "__main__":
    probe_fp4e2m1x2_host_roundtrip()
