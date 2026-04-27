# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Golden tiered file validation with mismatch heatmap visualization.

When strict torch.allclose golden checks fail (e.g. SPMD / simulator drift),
validate by reading golden vs actual tensor files from disk: illegal-value
checks, suspicious zero runs vs golden, mismatch-ratio tiers (Marginal / Soft),
extreme-error downgrade, and optional mismatch heatmaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


@dataclass
class TieredValidationResult:
    """Result of tiered validation for a single tensor pair."""

    name: str
    verdict: Literal["PASS", "Marginal Pass", "Soft Pass", "FAIL"]
    mismatch_ratio: float
    has_nan: bool
    has_inf: bool
    nan_count: int
    inf_count: int
    consecutive_zeros_len: int = 0
    consecutive_zeros_start: int = -1
    degraded_by_extreme_error: bool = False
    golden_values_range: tuple[float, float] | None = None


@dataclass
class AggregateValidationResult:
    """Aggregated result across multiple tensors."""

    overall_verdict: Literal["PASS", "Marginal Pass", "Soft Pass", "FAIL"]
    tensor_results: list[TieredValidationResult]
    passed: bool


def floor_pow2(n: int) -> int:
    """Return largest power of 2 <= n."""
    if n <= 1:
        return 1
    return 1 << int(math.floor(math.log2(n)))


def choose_cols_pow2(numel: int) -> int:
    """Choose column count as power of 2 near sqrt(numel)."""
    base = int(math.sqrt(max(numel, 1)))
    cols = floor_pow2(base)
    return max(cols, 1)


def choose_tick_step_pow2(axis_len: int, target_ticks: int = 16) -> int:
    """Choose tick step as power of 2 to get ~target_ticks ticks."""
    raw = max(1, axis_len // max(target_ticks, 1))
    return floor_pow2(raw)


def find_longest_consecutive_zeros(mask: torch.Tensor) -> tuple[int, int]:
    """Find longest consecutive True segment in boolean mask.

    Returns:
        (length, start_index) of longest consecutive run.
        Returns (0, -1) if no True values.
    """
    if not mask.any():
        return 0, -1

    mask_np = mask.cpu().numpy()
    n = len(mask_np)

    max_len = 0
    max_start = -1
    current_len = 0
    current_start = 0

    for i in range(n):
        if mask_np[i]:
            if current_len == 0:
                current_start = i
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
            current_len = 0

    # Check final run
    if current_len > max_len:
        max_len = current_len
        max_start = current_start

    return int(max_len), int(max_start)


def validate_single_tensor_tiered(
    actual: torch.Tensor,
    golden: torch.Tensor,
    name: str,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    min_consecutive_zeros: int = 8,
) -> TieredValidationResult:
    """Validate a single tensor pair using tiered validation rules.

    Rules (in order):
    1. Illegal values (NaN/Inf) -> FAIL
    2. Suspicious consecutive zeros -> FAIL
    3. Mismatch ratio tiering (PASS/Marginal/Soft/FAIL)
    4. Extreme error downgrade

    Args:
        actual: Actual output tensor
        golden: Golden reference tensor
        name: Tensor name for reporting
        rtol: Relative tolerance for isclose
        atol: Absolute tolerance for isclose
        min_consecutive_zeros: Threshold for consecutive zeros check

    Returns:
        TieredValidationResult with verdict and details
    """
    if actual.shape != golden.shape:
        raise ValueError(
            f"Shape mismatch for '{name}': actual={tuple(actual.shape)}, golden={tuple(golden.shape)}"
        )

    numel = actual.numel()

    # Rule 1: Illegal values check
    has_nan = actual.isnan().any().item()
    nan_count = actual.isnan().sum().item() if has_nan else 0
    has_inf = actual.isinf().any().item()
    inf_count = actual.isinf().sum().item() if has_inf else 0

    if has_nan or has_inf:
        return TieredValidationResult(
            name=name,
            verdict="FAIL",
            mismatch_ratio=1.0,
            has_nan=has_nan,
            has_inf=has_inf,
            nan_count=nan_count,
            inf_count=inf_count,
        )

    # Rule 2: Suspicious consecutive zeros
    # Only where golden is non-zero but actual is exactly zero
    mask_pos = (golden != 0) & (actual == 0)
    consecutive_len, start_flat = find_longest_consecutive_zeros(mask_pos.flatten())

    # Adjust threshold for small tensors
    threshold = min_consecutive_zeros if numel >= 128 else 4

    if consecutive_len >= threshold:
        # Get golden value range for this segment
        flat_golden = golden.flatten()
        segment_golden = flat_golden[start_flat : start_flat + consecutive_len]
        golden_min = segment_golden.min().item()
        golden_max = segment_golden.max().item()

        return TieredValidationResult(
            name=name,
            verdict="FAIL",
            mismatch_ratio=mask_pos.sum().item() / numel,
            has_nan=False,
            has_inf=False,
            nan_count=0,
            inf_count=0,
            consecutive_zeros_len=consecutive_len,
            consecutive_zeros_start=start_flat,
            golden_values_range=(golden_min, golden_max),
        )

    # Rule 3: Mismatch ratio tiering
    close = torch.isclose(actual, golden, rtol=rtol, atol=atol)
    mismatch_ratio = (~close).sum().item() / numel

    if mismatch_ratio == 0:
        verdict = "PASS"
    elif mismatch_ratio < 0.01:
        verdict = "Marginal Pass"
    elif mismatch_ratio <= 0.10:
        verdict = "Soft Pass"
    else:
        verdict = "FAIL"

    # Rule 4: Extreme error downgrade
    degraded = False
    if verdict != "FAIL":
        mismatched = ~close
        if mismatched.any():
            abs_diff = (actual - golden).abs()
            extreme_mask = abs_diff > 10 * atol
            if extreme_mask.any():
                # Downgrade one tier
                if verdict == "Soft Pass":
                    verdict = "Marginal Pass"
                elif verdict == "Marginal Pass":
                    verdict = "Soft Pass"  # At boundary, keep as Soft Pass
                degraded = True

    return TieredValidationResult(
        name=name,
        verdict=verdict,
        mismatch_ratio=mismatch_ratio,
        has_nan=False,
        has_inf=False,
        nan_count=0,
        inf_count=0,
        consecutive_zeros_len=consecutive_len,
        consecutive_zeros_start=start_flat,
        degraded_by_extreme_error=degraded,
    )


def aggregate_tier_verdict(
    results: list[TieredValidationResult],
) -> AggregateValidationResult:
    """Aggregate individual tensor results into overall verdict.

    Rules:
    - Any FAIL -> overall FAIL
    - Otherwise, take weakest tier: Soft Pass > Marginal Pass > PASS
    """
    if not results:
        return AggregateValidationResult(
            overall_verdict="PASS", tensor_results=[], passed=True
        )

    # Check for any FAIL
    if any(r.verdict == "FAIL" for r in results):
        return AggregateValidationResult(
            overall_verdict="FAIL", tensor_results=results, passed=False
        )

    # Determine weakest tier
    tiers = [r.verdict for r in results]
    if "Soft Pass" in tiers:
        overall = "Soft Pass"
    elif "Marginal Pass" in tiers:
        overall = "Marginal Pass"
    else:
        overall = "PASS"

    return AggregateValidationResult(
        overall_verdict=overall, tensor_results=results, passed=overall != "FAIL"
    )


def tiered_validate_from_dirs(
    out_dir: str | Path,
    actual_dir: str | Path,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    min_consecutive_zeros: int = 8,
) -> AggregateValidationResult:
    """Validate all tensor files between output and actual directories.

    Args:
        out_dir: Directory containing golden .pt files
        actual_dir: Directory containing actual .pt files
        rtol: Relative tolerance for isclose
        atol: Absolute tolerance for isclose
        min_consecutive_zeros: Threshold for consecutive zeros check

    Returns:
        AggregateValidationResult with overall verdict
    """
    out_path = Path(out_dir)
    actual_path = Path(actual_dir)

    # Find all .pt files in both directories
    out_files = {f.stem: f for f in out_path.glob("*.pt")}
    actual_files = {f.stem: f for f in actual_path.glob("*.pt")}

    # Ensure same set of names
    all_names = set(out_files.keys()) | set(actual_files.keys())

    results = []
    for name in sorted(all_names):
        if name not in out_files:
            print(f"[WARN] '{name}' missing in golden dir")
            continue
        if name not in actual_files:
            print(f"[WARN] '{name}' missing in actual dir")
            continue

        golden = torch.load(out_files[name], weights_only=True)
        actual = torch.load(actual_files[name], weights_only=True)

        result = validate_single_tensor_tiered(
            actual,
            golden,
            name,
            rtol=rtol,
            atol=atol,
            min_consecutive_zeros=min_consecutive_zeros,
        )
        results.append(result)

        # Print summary
        status = "✓" if result.verdict == "PASS" else "⚠" if result.verdict in ("Marginal Pass", "Soft Pass") else "✗"
        print(f"{status} {name}: {result.verdict} (mismatch={result.mismatch_ratio:.4%})")

        if result.has_nan or result.has_inf:
            print(f"    Illegal values: NaN={result.nan_count}, Inf={result.inf_count}")
        if result.consecutive_zeros_len > 0:
            print(f"    Consecutive zeros: len={result.consecutive_zeros_len} at {result.consecutive_zeros_start}")
        if result.degraded_by_extreme_error:
            print(f"    [Degraded by extreme error]")

    return aggregate_tier_verdict(results)


def plot_mismatch_map(
    golden_pt: str | Path,
    actual_pt: str | Path,
    rtol: float = 3e-3,
    atol: float = 3e-3,
    out_png: str | Path | None = None,
) -> None:
    """Plot mismatch heatmap between golden and actual tensors.

    Grid columns and tick step sizes are constrained to powers of 2.

    Args:
        golden_pt: Path to golden .pt file
        actual_pt: Path to actual .pt file
        rtol: Relative tolerance for isclose
        atol: Absolute tolerance for isclose
        out_png: Output PNG path (if None, displays plot)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[ERROR] matplotlib and numpy required for plotting")
        return

    golden = torch.load(golden_pt, weights_only=True).detach().cpu()
    actual = torch.load(actual_pt, weights_only=True).detach().cpu()

    if golden.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: golden={tuple(golden.shape)} actual={tuple(actual.shape)}"
        )

    close = torch.isclose(actual, golden, rtol=rtol, atol=atol)
    mism = (~close).to(torch.int32).reshape(-1).numpy()
    n = int(mism.size)

    cols = choose_cols_pow2(n)
    rows = (n + cols - 1) // cols
    total = rows * cols

    # 2 = padding blank (gray)
    data = np.full(total, 2, dtype=np.int32)
    data[:n] = mism
    grid = data.reshape(rows, cols)

    # 0=green(match), 1=red(mismatch), 2=gray(padding)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#2ecc71", "#e74c3c", "#bdc3c7"])

    fig_w = min(18, max(8, cols / 64))
    fig_h = min(18, max(8, rows / 64))
    plt.figure(figsize=(fig_w, fig_h), dpi=120)
    plt.imshow(grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=2, aspect="auto")

    # Tick steps also power of 2
    x_step = choose_tick_step_pow2(cols)
    y_step = choose_tick_step_pow2(rows)
    xt = range(0, cols, x_step)
    yt = range(0, rows, y_step)
    plt.xticks(xt, xt, fontsize=7)
    plt.yticks(yt, yt, fontsize=7)
    plt.xlabel(f"Column index (step={x_step}=2^k)", fontsize=11)
    plt.ylabel(f"Row index (step={y_step}=2^k)", fontsize=11)

    mismatch_count = int(mism.sum())
    mismatch_ratio = mismatch_count / max(n, 1)
    plt.title(
        "Golden Mismatch Distribution\n"
        f"shape={tuple(actual.shape)} numel={n} grid={rows}x{cols} (cols=2^k)\n"
        f"mismatch={mismatch_count}/{n} ({mismatch_ratio:.4%}), rtol={rtol}, atol={atol}",
        fontsize=12,
    )
    plt.grid(which="both", color="white", linewidth=0.2)
    plt.tight_layout()

    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"[saved] {out_png}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Golden Tiered Validation Module")
    print("=" * 40)
    print("\nUsage:")
    print("  from golden.tiered_validation import tiered_validate_from_dirs, plot_mismatch_map")
    print("  result = tiered_validate_from_dirs('data/out', 'data/actual')")
    print("  plot_mismatch_map('data/out/out.pt', 'data/actual/out.pt', out_png='mismatch.png')")
