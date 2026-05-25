# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""由 ``block_table`` 驱动的分页投影 + 分页 RMSNorm 样例。

两个 ``pl.parallel`` 循环, 每个内部各套一个 ``pl.at`` 作用域, 分别落到
**AIC (纯 cube)** 与 **AIV (纯 vec)** 两条流水。整体结构对齐 Qwen3 /
DeepSeek paged-KV-cache decode 流程中的一小段, 精简为 "cube 生产者 +
vec 消费者" 一对, 以便把焦点放在 ``block_table`` 间接寻址上。

Stage 1 —— ``paged_proj`` (AIC, 纯 cube):
    每个页号 ob 上, 对 ``x`` 的一片 ([PAGE_M, HIDDEN]) 与 ``w1`` 做
    K 维 ``pl.range`` 顺序累加 (首拍 ``pl.matmul``, 其余 ``pl.matmul_acc``),
    FP32 累加结果直接 ``pl.assemble`` 连续写入中间张量 ``paged_y`` (FP32)。
    pl.at 内只有 cube + MTE, 没有 vec 算子 —— 编译器只会下出 AIC 函数。
    不引用 ``block_table``。

Stage 2 —— ``paged_rmsnorm`` (AIV, 纯 vec):
    每个输出页号 ob 上, 仅用一次 ``pl.tensor.read(block_table, [ob])``
    解出物理源页号 ``page_id``, 从 ``paged_y`` gather 一片 ([PAGE_M, N1])
    FP32 数据, 做标准 RMSNorm:
        sq_sum  = row_sum(y * y)              # [PAGE_M, 1]
        inv_rms = rsqrt(sq_sum / N1 + EPS)
        out     = col_expand_mul(
                     row_expand_mul(y, inv_rms),
                     gamma,                    # [1, N1]
                  )
    block_table 把 ``paged_y`` 的行做了一次置换, RMSNorm 实际跑在乱序数据
    之上。pl.at 内只有 vec + MTE, 编译器只会下出 AIV 函数。

每个 ``pl.at`` 至多触及 ``block_table`` 一次 (实际只有 stage 2 用到)。
page_id 在 pl.at 入口一次读出, 与 decode 内核里 ``qk_pbid`` 的写法同形。

============================================================================
Qwen3 / DeepSeek 等大模型代码中具有相似功能的代码
============================================================================

Stage 1 (AIC 纯 cube projection) 对照:
  * Qwen3 QKV 投影 —— models/qwen3/14b/decode_layer.py 中
    ``name_hint="q_proj"`` / ``"k_proj"`` / ``"v_proj"`` 三个 pl.at:
      - 外层 ``for q0 in pl.parallel(0, hidden, Q_OUT_STEP)`` 切 N 维。
      - 内层 K 维累加, 首拍 ``pl.matmul`` 之后 ``pl.matmul_acc`` (Qwen3
        原版用 ``pl.pipeline(..., stage=2)`` 软流水, 本样例为简化用
        ``pl.range`` 顺序累加, 语义等价)。
      - 原版末尾还有 ``pl.cast`` + ``pl.assemble``, 本 stage 1 为了做成
        纯 AIC, 把 cast 删除、直接 assemble FP32 累加结果。
    与本 stage 1 在 cube 部分完全同构, 只是把外层 parallel 维度由
    N (输出列) 换成 M (页编号), 方便后续按页索引重排。

  * DeepSeek V4 paged-KV 写入 —— models/deepseek/v4/decode_swa.py 的 KV
    投影 + 写 cache 路径, stage 1 末尾的 ``pl.assemble(paged_y, ...)`` 在
    那里对应 ``pl.assemble(kv_cache_flat, ...)`` 的连续写。

Stage 2 (AIV 纯 vec, RMSNorm + block_table gather) 对照:
  * Qwen3 q_norm / k_norm —— models/qwen3/14b/decode_layer.py 的
    ``name_hint="qk_norm"`` pl.at, 紧跟 q_proj/k_proj 之后对 head_dim 这一
    维做 RMSNorm; 本 stage 2 与之同形 (row_sum -> rsqrt -> row/col
    expand_mul), 只是把待归一化的 tile 来源从 "线性顺序 GM" 换成
    "block_table gather 后的 paged_y"。

  * Qwen3 paged attention 的 QK / SV matmul —— 同一份 decode_layer.py,
    ``name_hint="fa_qks"`` / ``"fa_svo"`` 这两个 pl.at:
        qk_pbid = pl.cast(
            pl.tensor.read(block_table, [qk_block_table_idx]),
            pl.INDEX,
        )
        qk_cache_row0 = layer_cache_base
                        + (qk_pbid * NUM_KV_HEADS + kvh0) * BLOCK_SIZE
        k_tile0 = k_cache[qk_cache_row0 : qk_cache_row0 + BLOCK_SIZE, :]
    用 block_table 把逻辑 token 位置翻译成物理 KV 页。
    *差别:* qwen3 在 pl.range(ctx_blocks) 里反复读 block_table, 一个
    pl.at 会读多次; 本样例为满足 "每个 pl.at 只用 block_table 一项",
    把读提到 pl.at 入口, 一次读出 ``page_id`` 后由整个 RMSNorm 共享。

  * DeepSeek V4 SWA 的 KV scatter ——
    models/deepseek/v4/decode_attention_swa.py L160-172:
        blk_id  = pl.cast(pl.read(block_table_flat, [b]), pl.INDEX)
        dst_row = blk_id * BLOCK_SIZE + ori_slot
        kv_cache_flat = pl.assemble(kv_cache_flat, kv[...], [dst_row, 0])
    同样的 "一次 block_table 读 + 用结果做 row 偏移" 模式, 只是那里用于
    scatter (写 kv_cache), 本 stage 2 用于 gather (读 paged_y)。

  * DeepSeek V4 sparse attention 的 block 选择 ——
    models/deepseek/v4/prefill_sparse_attn.py L138:
        gather_blk = pl.cast(
            pl.read(ori_block_table_flat, [gather_block_pos]),
            pl.INDEX,
        )
    与本 stage 2 入口的 ``page_id = pl.cast(pl.tensor.read(...), pl.INDEX)``
    完全同形, 都是用一次 block_table 查表换得物理页号供下游使用。

"""

import pypto.language as pl

# ---------------------------------------------------------------------------
# Problem shape — small but representative of an LLM projection layer.
# Tile shape ([PAGE_M, N{1,2}] = [16, 256], K_CHUNK = 128) matches the
# validated cube/vec budget of examples/advanced/multi_proj.py so both
# stages fit within Mat (524288 B) and Vec (196608 B) buffer limits.
# ---------------------------------------------------------------------------
NUM_PAGES = 16
PAGE_M    = 16                          # rows per page (cube-friendly M)
BATCH     = NUM_PAGES * PAGE_M          # 256

HIDDEN    = 2048                        # stage-1 K
N1        = 256                         # stage-1 N (= stage-2 RMSNorm width)

K1_CHUNK  = 128                         # stage-1 K tile  -> 16 K-iterations

EPS       = 1.0e-6                      # RMSNorm epsilon


@pl.jit
def paged_consumer_block_table(
    x:           pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    w1:          pl.Tensor[[HIDDEN, N1],    pl.BF16],
    gamma:       pl.Tensor[[1, N1],         pl.FP32],
    block_table: pl.Tensor[[NUM_PAGES],     pl.INT32],
    out:         pl.Out[pl.Tensor[[BATCH, N1], pl.FP32]],
):
    k1_blocks = HIDDEN // K1_CHUNK

    # Cooperative intermediate filled by stage 1 (cube), gathered by stage 2
    # (vec). Per coding-style §4, multi-pl.at intermediates live in
    # orchestration via pl.create_tensor. Kept FP32 so stage 1 stays a pure
    # cube region (no cast epilogue).
    paged_y = pl.create_tensor([NUM_PAGES * PAGE_M, N1], dtype=pl.FP32)

    # ---------- Stage 1: pure-cube projection (AIC) ----------
    # Only matmul / matmul_acc + slice / assemble — no vec op, so the
    # compiler lowers this pl.at to an AIC-only function.
    for ob in pl.parallel(0, NUM_PAGES):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="paged_proj"):
            m0 = ob * PAGE_M
            acc1 = pl.create_tensor([PAGE_M, N1], dtype=pl.FP32)
            tile_x0  = pl.slice(x,  [PAGE_M,   K1_CHUNK], [m0, 0])
            tile_w10 = pl.slice(w1, [K1_CHUNK, N1],       [0,  0])
            acc1 = pl.matmul(tile_x0, tile_w10, out_dtype=pl.FP32)
            for kb in pl.range(1, k1_blocks):
                k0 = kb * K1_CHUNK
                tile_x  = pl.slice(x,  [PAGE_M,   K1_CHUNK], [m0, k0])
                tile_w1 = pl.slice(w1, [K1_CHUNK, N1],       [k0,  0])
                acc1 = pl.matmul_acc(acc1, tile_x, tile_w1)
            paged_y = pl.assemble(paged_y, acc1, [m0, 0])

    # ---------- Stage 2: pure-vec RMSNorm with block_table gather (AIV) ----------
    # Only vec ops (mul / add / row_sum / rsqrt / row_expand_mul /
    # col_expand_mul) + slice / assemble — no matmul, so the compiler lowers
    # this pl.at to an AIV-only function. Mirrors the q_norm / k_norm pattern
    # at models/qwen3/14b/decode_layer.py:240-244.
    for ob in pl.parallel(0, NUM_PAGES):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="paged_rmsnorm"):
            # The single block_table read for this pl.at activation;
            # reused as the row base of the gathered tile below.
            page_id = pl.cast(pl.tensor.read(block_table, [ob]), pl.INDEX)
            src_row = page_id * PAGE_M
            out_m0  = ob * PAGE_M

            y_tile     = pl.slice(paged_y, [PAGE_M, N1], [src_row, 0])
            gamma_tile = pl.slice(gamma,   [1,      N1], [0,       0])

            sq_sum  = pl.row_sum(pl.mul(y_tile, y_tile))   # [PAGE_M, 1]
            inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / N1), EPS))
            normed  = pl.col_expand_mul(
                pl.row_expand_mul(y_tile, inv_rms),
                gamma_tile,
            )
            out = pl.assemble(out, normed, [out_m0, 0])

    return out


def build_tensor_specs():
    import torch

    from golden import TensorSpec

    scale_h = HIDDEN ** 0.5

    def init_x():
        return torch.rand(BATCH, HIDDEN) - 0.5

    def init_w1():
        return (torch.rand(HIDDEN, N1) - 0.5) / scale_h

    def init_gamma():
        # Per-column RMSNorm scale; small jitter around 1.0, mirroring the
        # q_norm / k_norm weight shape (~1.0) in Qwen3.
        return 1.0 + 0.1 * (torch.rand(1, N1) - 0.5)

    def init_block_table():
        # Random permutation of [0, NUM_PAGES) drives the page shuffle.
        return torch.randperm(NUM_PAGES).to(torch.int32)

    return [
        TensorSpec("x",           [BATCH, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("w1",          [HIDDEN, N1],    torch.bfloat16, init_value=init_w1),
        TensorSpec("gamma",       [1, N1],         torch.float32,  init_value=init_gamma),
        TensorSpec("block_table", [NUM_PAGES],     torch.int32,    init_value=init_block_table),
        TensorSpec("out",         [BATCH, N1],     torch.float32,  is_output=True),
    ]


def golden_paged_consumer_block_table(tensors):
    import torch

    x_f32     = tensors["x"].float()
    w1_f32    = tensors["w1"].float()
    gamma_f32 = tensors["gamma"].float()                # [1, N1]
    btab      = tensors["block_table"]

    # Stage 1: y = x @ w1 (FP32, contiguous layout — matches the FP32
    # paged_y the AIC kernel assembles).
    y_f32 = x_f32 @ w1_f32                              # [BATCH, N1]

    # Stage 2: per page, gather via block_table then RMSNorm.
    out_f32 = torch.zeros(BATCH, N1, dtype=torch.float32)
    for ob in range(NUM_PAGES):
        page_id = int(btab[ob].item())
        src     = y_f32[page_id * PAGE_M : (page_id + 1) * PAGE_M, :]   # [PAGE_M, N1]
        mean_sq = (src * src).mean(dim=-1, keepdim=True)                # [PAGE_M, 1]
        inv_rms = torch.rsqrt(mean_sq + EPS)
        out_f32[ob * PAGE_M : (ob + 1) * PAGE_M, :] = src * inv_rms * gamma_f32
    tensors["out"][:] = out_f32


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=paged_consumer_block_table,
        specs=build_tensor_specs(),
        golden_fn=golden_paged_consumer_block_table,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=4e-3,
        atol=4e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
