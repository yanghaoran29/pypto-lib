#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tiered golden vs actual mismatch heatmap (aligned with golden-tiered-validation skill).

Per-pixel priority (same as tiered validation):
  (1) Match at baseline rtol/atol (torch.isclose): green
  (2) Abnormal zero (golden != 0 and actual == 0): gray
  (3) Baseline fail but isclose at 2×rtol / 2×atol: yellow
  (4) Not (3) but isclose at 5×rtol / 5×atol: light red
  (5) Otherwise: red

Grid column count and axis tick steps are powers of two (enforced in this script).

Plot text (title, axes, legend) is **English** so default DejaVu Sans renders cleanly.
Optional: set ``PYPTO_MISMATCH_HEATMAP_FONT`` or ``--font-file PATH`` to ``addfont`` a custom
``.ttf`` / ``.ttc`` / ``.otf`` (inserted after DejaVu in the sans-serif chain).
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap
import numpy as np
import torch

_REGISTERED_FONT_PATHS: set[str] = set()
_plot_font_configured = False
_module_font_file_override: str | None = None


def set_mismatch_heatmap_user_font(path: str | Path | None) -> None:
    """Register a font file before plotting; clears cached font config.

    Same effect as environment variable ``PYPTO_MISMATCH_HEATMAP_FONT``.
    """
    global _module_font_file_override, _plot_font_configured
    _module_font_file_override = os.path.expanduser(str(path)) if path else None
    _plot_font_configured = False


def _path_known_to_matplotlib(sp: str) -> bool:
    for info in fm.fontManager.ttflist:
        try:
            if Path(info.fname).resolve() == Path(sp).resolve():
                return True
        except OSError:
            if info.fname == sp:
                return True
    return False


def _try_register_font_file(path: Path) -> str | None:
    """Register a font file with Matplotlib if loadable; return its family name."""
    if not path.is_file():
        return None
    try:
        sp = str(path.resolve())
    except OSError:
        sp = str(path)
    try:
        prop = fm.FontProperties(fname=sp)
        name = prop.get_name()
    except Exception:
        return None
    if not _path_known_to_matplotlib(sp) and sp not in _REGISTERED_FONT_PATHS:
        try:
            fm.fontManager.addfont(sp)
            _REGISTERED_FONT_PATHS.add(sp)
        except Exception:
            return None
    return name


def _configure_plot_fonts_if_needed() -> None:
    """Use DejaVu Sans; optionally prepend user font from env / module override."""
    global _plot_font_configured
    if _plot_font_configured:
        return

    chain: list[str] = ["DejaVu Sans"]
    for src in filter(
        None,
        (
            _module_font_file_override,
            (os.environ.get("PYPTO_MISMATCH_HEATMAP_FONT") or "").strip() or None,
        ),
    ):
        n = _try_register_font_file(Path(os.path.expanduser(src)))
        if n is not None and n not in chain:
            chain.insert(1, n)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = chain
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"Glyph \d+ .* missing from font",
    )
    _plot_font_configured = True


# Category codes -> colors (incl. padding)
# 0 green 1 yellow 2 light red 3 red 4 abnormal zero gray 5 padding
_CATEGORY_COLORS = [
    "#27ae60",
    "#f1c40f",
    "#f5b7b1",
    "#c0392b",
    "#7f8c8d",
    "#ecf0f1",
]


def floor_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << int(math.floor(math.log2(n)))


def choose_cols_pow2(numel: int) -> int:
    base = int(math.sqrt(max(numel, 1)))
    cols = floor_pow2(base)
    return max(cols, 1)


def choose_tick_step_pow2(axis_len: int, target_ticks: int = 16) -> int:
    raw = max(1, axis_len // max(target_ticks, 1))
    return floor_pow2(raw)


def classify_pixels(
    golden: torch.Tensor,
    actual: torch.Tensor,
    rtol: float,
    atol: float,
) -> torch.Tensor:
    """int64 tensor same shape as golden, values 0..4 (5 reserved for grid padding)."""
    g = golden.detach().cpu().float()
    a = actual.detach().cpu().float()

    base_ok = torch.isclose(a, g, rtol=rtol, atol=atol, equal_nan=False)
    ok_2x = torch.isclose(a, g, rtol=2.0 * rtol, atol=2.0 * atol, equal_nan=False)
    ok_5x = torch.isclose(a, g, rtol=5.0 * rtol, atol=5.0 * atol, equal_nan=False)
    abnormal_zero = (g != 0) & (a == 0)

    cat = torch.full_like(g, 3, dtype=torch.int64)
    cat[base_ok] = 0
    not_ok = ~base_ok
    cat[not_ok & abnormal_zero] = 4
    rest = not_ok & ~abnormal_zero
    cat[rest & ok_2x] = 1
    cat[rest & (~ok_2x) & ok_5x] = 2
    return cat


def plot_mismatch_map_tensors(
    golden: torch.Tensor,
    actual: torch.Tensor,
    rtol: float = 3e-3,
    atol: float = 3e-3,
    out_png: str | Path | None = None,
) -> None:
    """Render tiered mismatch heatmap from in-memory tensors (same rules as skill)."""
    _configure_plot_fonts_if_needed()

    golden = golden.detach().cpu()
    actual = actual.detach().cpu()

    if golden.shape != actual.shape:
        raise ValueError(f"shape mismatch: golden={tuple(golden.shape)} actual={tuple(actual.shape)}")

    cat = classify_pixels(golden, actual, rtol=rtol, atol=atol)
    flat = cat.reshape(-1).numpy()
    n = int(flat.size)

    cols = choose_cols_pow2(n)
    rows = (n + cols - 1) // cols
    total = rows * cols

    data = np.full(total, 5, dtype=np.int32)
    data[:n] = flat.astype(np.int32, copy=False)
    grid = data.reshape(rows, cols)

    cmap = ListedColormap(_CATEGORY_COLORS)
    fig_w = min(18, max(8, cols / 64))
    fig_h = min(18, max(8, rows / 64))
    plt.figure(figsize=(fig_w, fig_h), dpi=120)
    plt.imshow(grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=5, aspect="auto")

    x_step = choose_tick_step_pow2(cols)
    y_step = choose_tick_step_pow2(rows)
    xt = np.arange(0, cols, x_step)
    yt = np.arange(0, rows, y_step)
    plt.xticks(xt, xt, fontsize=7)
    plt.yticks(yt, yt, fontsize=7)
    plt.xlabel(f"Column index (tick step={x_step}, power of 2)", fontsize=11)
    plt.ylabel(f"Row index (tick step={y_step}, power of 2)", fontsize=11)

    base_close = torch.isclose(actual, golden, rtol=rtol, atol=atol)
    mismatch_count = int((~base_close).sum().item())
    mismatch_ratio = mismatch_count / max(n, 1)
    plt.title(
        "Golden mismatch map (tiered)\n"
        f"shape={tuple(actual.shape)}  numel={n}  grid={rows}x{cols} (cols=2^k)\n"
        f"baseline mismatch={mismatch_count}/{n} ({mismatch_ratio:.4%}), rtol={rtol}, atol={atol}\n"
        "Legend: green=match | gray=abnormal zero | yellow=within 2× | "
        "light red=2×..5× band | red=>5× | pale gray=padding",
        fontsize=11,
    )
    plt.grid(which="both", color="white", linewidth=0.2)
    plt.tight_layout()

    legend_elems = [
        mpatches.Patch(color=_CATEGORY_COLORS[0], label="(1) match"),
        mpatches.Patch(color=_CATEGORY_COLORS[4], label="(2) abnormal zero"),
        mpatches.Patch(color=_CATEGORY_COLORS[1], label="(3) within 2× tol"),
        mpatches.Patch(color=_CATEGORY_COLORS[2], label="(4) within 2×..5× tol"),
        mpatches.Patch(color=_CATEGORY_COLORS[3], label="(5) other error"),
        mpatches.Patch(color=_CATEGORY_COLORS[5], label="padding"),
    ]
    plt.legend(handles=legend_elems, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=8)

    if out_png:
        p = Path(out_png)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[saved] {p.resolve()}")
    else:
        plt.show()
    plt.close()


def plot_mismatch_map(
    golden_pt: str | Path,
    actual_pt: str | Path,
    rtol: float = 3e-3,
    atol: float = 3e-3,
    out_png: str | Path | None = None,
) -> None:
    golden = torch.load(str(golden_pt), weights_only=True)
    if not isinstance(golden, torch.Tensor):
        raise TypeError(f"expected Tensor in {golden_pt}, got {type(golden)}")
    actual = torch.load(str(actual_pt), weights_only=True)
    if not isinstance(actual, torch.Tensor):
        raise TypeError(f"expected Tensor in {actual_pt}, got {type(actual)}")
    plot_mismatch_map_tensors(golden, actual, rtol=rtol, atol=atol, out_png=out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot golden vs actual mismatch heatmap (tiered colors).")
    ap.add_argument("--golden", required=True, help="Path to golden .pt file")
    ap.add_argument("--actual", required=True, help="Path to actual .pt file")
    ap.add_argument("--rtol", type=float, default=3e-3)
    ap.add_argument("--atol", type=float, default=3e-3)
    ap.add_argument("--out", "-o", default=None, help="Output .png path")
    ap.add_argument(
        "--font-file",
        default=None,
        metavar="PATH",
        help="Optional .ttf/.ttc/.otf to register (same as PYPTO_MISMATCH_HEATMAP_FONT)",
    )
    args = ap.parse_args()
    if args.font_file:
        set_mismatch_heatmap_user_font(args.font_file)
    plot_mismatch_map(
        golden_pt=args.golden,
        actual_pt=args.actual,
        rtol=args.rtol,
        atol=args.atol,
        out_png=args.out,
    )


if __name__ == "__main__":
    main()
