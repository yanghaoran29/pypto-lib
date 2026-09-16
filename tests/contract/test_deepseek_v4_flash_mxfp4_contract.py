# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host contracts shared with the A5 native MXFP4 cache path."""

import torch

from models.deepseek_v4_1_flash.quantization import _nearest_fp4_indices


def test_fp4_midpoints_use_round_to_nearest_even_codes():
    positive = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    negative = -positive

    assert torch.equal(
        _nearest_fp4_indices(positive),
        torch.tensor([0, 2, 2, 4, 4, 6, 6], dtype=torch.uint8),
    )
    assert torch.equal(
        _nearest_fp4_indices(negative),
        torch.tensor([8, 10, 10, 12, 12, 14, 14], dtype=torch.uint8),
    )


def test_fp4_signed_zero_is_canonicalized_by_native_a5_cast():
    values = torch.tensor([0.0, -0.0])
    assert torch.equal(
        _nearest_fp4_indices(values),
        torch.tensor([0, 0], dtype=torch.uint8),
    )
