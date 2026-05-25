# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
============================================================================
Qwen3 / DeepSeek 等大模型中类似写法对照
============================================================================

* per-batch 动态 ctx_len 读取 + 用作循环界 ——
  models/qwen3/14b/decode_layer.py:268-271:
      for b in pl.parallel(user_batch):
          ctx_len    = pl.tensor.read(seq_lens, [b])      # ← 同样的接口
          ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
          ...
          for sb in pl.range(ctx_blocks):                 # 后续循环界
  本样例同样在 batch 外层用 ``pl.tensor.read(seq_lens, [b])`` 取每个
  batch 的真实长度, 再用它驱动后续 ``pl.parallel`` / ``pl.range``。

"""

import pypto.language as pl

# ---------------------------------------------------------------------------
# 静态形状: BATCH / HEAD_DIM / MAX_SEQ; 实际每个 batch 的 seq_len 来自
# 运行期 INT32 Tensor seq_lens[BATCH], 不同 batch 之间可以不同
# ---------------------------------------------------------------------------
BATCH     = 16                          # 外层 batch 数
HEAD_DIM  = 64                          # 中间维 (每个 batch 每个 token 的特征数)
MAX_SEQ   = 128                         # 第三维静态上限, 实际每个 batch 用前 seq_lens[b] 列
SEQ_TILE  = 16                          # 每个 Stage 1 pl.at 处理 SEQ_TILE 列
REPEAT    = 100                         # 重复计算，防止Stage1计算太快，构建TensorMap时将已完成的Stage1依赖优化掉
EPS       = 1.0e-6                      # RMSNorm epsilon


@pl.jit
def dyn_seq_reduce(
    x:        pl.Tensor[[BATCH, HEAD_DIM, MAX_SEQ], pl.FP32],
    seq_lens: pl.Tensor[[BATCH],                    pl.INT32],
    gamma:    pl.Tensor[[1, MAX_SEQ],               pl.FP32],
    sum_out:  pl.Out[pl.Tensor[[BATCH, HEAD_DIM, SEQ_TILE], pl.FP32]],
):
    for b in pl.parallel(BATCH):
        # 运行期读取本 batch 的真实长度 (codegen -> get_tensor_data<int32_t>)。
        # host 端保证 seq_lens[b] 是 SEQ_TILE 的整数倍 (build_tensor_specs
        # 里 assert), 这里直接整除得到本 batch 的块数。
        seq_len_b    = pl.tensor.read(seq_lens, [b])
        seq_blocks_b = seq_len_b // SEQ_TILE                          # 动态计数

        # Per-batch 协作中间缓冲: 在 pl.parallel(BATCH) 内部 create_tensor,
        # 每个 batch 拿到独立的 2D buffer (不同 batch 不同 GM 指针), 这样
        # TensorMap 自然按指针 hash 分桶 -- 跨 batch 不会再有 Stage 1 ->
        # 别的 batch Stage 2 的误依赖。
        # 模式同 qwen3 decode_layer.py:366-373 的
        #   `all_raw_scores0 = pl.create_tensor([...], dtype=pl.FP32)`
        # 在 `for gi in pl.parallel(0, TOTAL_Q_GROUPS, 2):` 内部分配的写法。
        stage12_y_b = pl.create_tensor([HEAD_DIM, MAX_SEQ], dtype=pl.FP32)

        # ---------- Stage 1: SiLU + RMSNorm 合并 (动态 pl.parallel + pl.at) ----------
        # 每个 pl.at 处理 x[b, :, s0:s0+SEQ_TILE] -> [HEAD_DIM, SEQ_TILE]
        # FP32 tile (rows=HEAD_DIM=64, col_bytes=256, 对齐), 一次性做 SiLU
        # 和 RMSNorm 两步, 中间不出 pl.at。
        for sb in pl.parallel(seq_blocks_b):
            s0 = sb * SEQ_TILE
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage1_silu_rmsnorm"):
                # 3D slice -> 2D tile (整 batch b 的一个 [HEAD_DIM, SEQ_TILE] 块)
                tile_3d    = pl.slice(x, [1, HEAD_DIM, SEQ_TILE], [b, 0, s0])
                tile       = pl.reshape(tile_3d, [HEAD_DIM, SEQ_TILE])
                gamma_tile = pl.slice(gamma, [1, SEQ_TILE], [0, s0])

                # REPEAT 次级联 SiLU + RMSNorm, 把 stage 1 的 vec 计算量
                # 放大 REPEAT 倍 (上一次的 normed 作为下一次的输入)。
                # 用 pl.range 而不是原生 range —— @pl.jit 不接受 Python
                # 原生 range, 必须用 pl.range / pl.unroll / pl.parallel
                # 等之一; pl.range 在运行期顺序展开, 比 pl.unroll 编译期
                # 展开 50 份 IR 体量更友好。
                normed = tile
                for _ in pl.range(REPEAT):
                    # SiLU: x / (1 + exp(-x))
                    denom = pl.add(pl.exp(pl.neg(normed)), 1.0)
                    silu  = pl.mul(normed, pl.recip(denom))           # [HEAD_DIM, SEQ_TILE]
                    # RMSNorm (per-row, 在 SEQ_TILE 维上归一化)
                    sq_sum  = pl.row_sum(pl.mul(silu, silu))          # [HEAD_DIM, 1]
                    inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / SEQ_TILE), EPS))
                    normed  = pl.col_expand_mul(
                        pl.row_expand_mul(silu, inv_rms),
                        gamma_tile,
                    )

                # 直接写回 2D 的 per-batch 缓冲, 不再需要 reshape 到 3D
                stage12_y_b = pl.assemble(stage12_y_b, normed, [0, s0])

        # ---------- Stage 2: 跨 seq 块累加 (pl.at 内部放动态 pl.range) ----------
        # 单个 pl.at 覆盖整个 HEAD_DIM (无外层 pl.parallel 切分)。acc 形状
        # [HEAD_DIM, SEQ_TILE] (rows=HEAD_DIM=64, col_bytes=256, 对齐),
        # 通过动态计数的 pl.range(seq_blocks_b) 把每个 stage12_y_b 子块按
        # 位累加到 acc 上 (块内同位求和)。
        # 注意瓦片改名为 ``blk`` 而不是复用 ``tile`` —— @pl.jit 前端把
        # Python 变量名当作整个函数体内的 SSA 符号, 同名变量只能绑定一种
        # (shape, dtype) 组合。
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2_reduce"):
            acc = pl.full([HEAD_DIM, SEQ_TILE], dtype=pl.FP32, value=0.0)
            for sb in pl.range(seq_blocks_b):                         # ← 动态计数
                s0  = sb * SEQ_TILE
                blk = pl.slice(stage12_y_b, [HEAD_DIM, SEQ_TILE], [0, s0])
                # REPEAT 次 add 把 stage 2 的 vec 计算量也放大 REPEAT 倍;
                # 等价于 acc += REPEAT * blk, 但显式写成 REPEAT 次 pl.add
                # 以保留 op-by-op 的指令数。同样必须用 pl.range 而不是
                # 原生 range, 才能被 @pl.jit 接受。
                for _ in pl.range(REPEAT):
                    acc = pl.add(acc, blk)
            acc_3d  = pl.reshape(acc, [1, HEAD_DIM, SEQ_TILE])
            sum_out = pl.assemble(sum_out, acc_3d, [b, 0, 0])

    return sum_out


def build_tensor_specs():
    """每个 batch 独立指定一个 seq_len, 必须是 SEQ_TILE 的整数倍且 <= MAX_SEQ。"""
    import torch

    from golden import TensorSpec

    # 演示用: 每个 batch 取不同的 seq_len, 都是 SEQ_TILE 的整数倍且 <= MAX_SEQ。
    # 用 torch.randint 在 [1, MAX_SEQ/SEQ_TILE] 区间随机生成块数, 再乘回
    # SEQ_TILE 得到对齐的 seq_len; 不同 BATCH 大小都能自动覆盖。
    max_blocks     = MAX_SEQ // SEQ_TILE
    generator      = torch.Generator().manual_seed(0)
    block_counts   = torch.randint(1, max_blocks + 1, (BATCH,), generator=generator, dtype=torch.int32)
    seq_lens_value = block_counts * SEQ_TILE                          # [BATCH], 每个值 ∈ [SEQ_TILE, MAX_SEQ]
    assert seq_lens_value.shape[0] == BATCH
    assert (seq_lens_value % SEQ_TILE == 0).all(), (
        f"每个 seq_len 必须是 SEQ_TILE={SEQ_TILE} 的整数倍, got {seq_lens_value.tolist()}"
    )
    assert (seq_lens_value <= MAX_SEQ).all(), (
        f"每个 seq_len 必须 <= MAX_SEQ={MAX_SEQ}, got {seq_lens_value.tolist()}"
    )
    def init_x():
        # 整个 [BATCH, HEAD_DIM, MAX_SEQ] 都填随机数; 每个 batch 末尾超出
        # seq_lens[b] 的列在 kernel 里不会被读到 (Stage 1 只跑 seq_blocks_b
        # 次, Stage 2 同理), 所以不需要特意清零。
        return torch.rand(BATCH, HEAD_DIM, MAX_SEQ) - 0.5

    def init_seq_lens():
        return seq_lens_value.clone()

    def init_gamma():
        return 1.0 + 0.1 * (torch.rand(1, MAX_SEQ) - 0.5)

    return [
        TensorSpec("x",        [BATCH, HEAD_DIM, MAX_SEQ],   torch.float32, init_value=init_x),
        TensorSpec("seq_lens", [BATCH],                       torch.int32,   init_value=init_seq_lens),
        TensorSpec("gamma",    [1, MAX_SEQ],                  torch.float32, init_value=init_gamma),
        TensorSpec("sum_out",  [BATCH, HEAD_DIM, SEQ_TILE],   torch.float32, is_output=True),
    ]


def golden_dyn_seq_reduce(tensors):
    """与 kernel 算子严格一一对齐的 reference 实现。

    每步注释指出对应的 ``pl.*`` 算子, 计算顺序 / 形状 / 中间量都镜像
    kernel 内的写法 (例如 SiLU 用 ``x / (1 + exp(-x))`` 而不是
    ``x * sigmoid(x)``, RMSNorm 用 ``sum * (1/N)`` 而不是 ``mean``,
    把 ``stage12_y`` 这个中间缓冲也显式建出来)。
    """
    import torch

    x        = tensors["x"].float()                                   # [BATCH, HEAD_DIM, MAX_SEQ]
    seq_lens = tensors["seq_lens"]
    gamma    = tensors["gamma"].float()                               # [1, MAX_SEQ]

    out = torch.zeros(BATCH, HEAD_DIM, SEQ_TILE, dtype=torch.float32)

    for b in range(BATCH):
        # 镜像 kernel: seq_len_b = pl.tensor.read(seq_lens, [b])
        sl = int(seq_lens[b].item())
        # 镜像 kernel: seq_blocks_b = seq_len_b // SEQ_TILE
        seq_blocks_b = sl // SEQ_TILE

        # 镜像 kernel: stage12_y_b = pl.create_tensor([HEAD_DIM, MAX_SEQ], FP32)
        # 每个 batch 一份独立的 2D 缓冲, 跨 batch 不共享 (与 kernel 一致)
        stage12_y_b = torch.zeros(HEAD_DIM, MAX_SEQ, dtype=torch.float32)

        # ---------- Stage 1: REPEAT 次级联 SiLU + RMSNorm ----------
        for sb in range(seq_blocks_b):                                # pl.parallel(seq_blocks_b)
            s0         = sb * SEQ_TILE
            tile       = x[b, :, s0 : s0 + SEQ_TILE]                  # pl.slice + pl.reshape -> [HEAD_DIM, SEQ_TILE]
            gamma_tile = gamma[:, s0 : s0 + SEQ_TILE]                 # pl.slice -> [1, SEQ_TILE]

            # 镜像 kernel: normed = tile; for _ in pl.range(REPEAT): silu+rmsnorm
            normed = tile
            for _ in range(REPEAT):
                # SiLU: x / (1 + exp(-x))
                #   <- kernel: pl.mul(normed, pl.recip(pl.add(pl.exp(pl.neg(normed)), 1.0)))
                denom = torch.exp(-normed) + 1.0
                silu  = normed * (1.0 / denom)                        # [HEAD_DIM, SEQ_TILE]

                # RMSNorm: row_sum -> *(1/N) -> rsqrt(... + EPS) -> row/col expand_mul
                sq_sum  = (silu * silu).sum(dim=-1, keepdim=True)     # [HEAD_DIM, 1]
                inv_rms = torch.rsqrt(sq_sum * (1.0 / SEQ_TILE) + EPS)
                normed  = (silu * inv_rms) * gamma_tile

            # 镜像 kernel: stage12_y_b = pl.assemble(stage12_y_b, normed, [0, s0])
            stage12_y_b[:, s0 : s0 + SEQ_TILE] = normed

        # ---------- Stage 2: 跨 seq 块累加, 每个 blk 做 REPEAT 次 add ----------
        # 镜像 kernel: acc = pl.full([HEAD_DIM, SEQ_TILE], 0.0)
        acc = torch.zeros(HEAD_DIM, SEQ_TILE, dtype=torch.float32)
        for sb in range(seq_blocks_b):                                # pl.range(seq_blocks_b)
            s0  = sb * SEQ_TILE
            blk = stage12_y_b[:, s0 : s0 + SEQ_TILE]                  # pl.slice -> [HEAD_DIM, SEQ_TILE]
            # 镜像 kernel: for _ in pl.range(REPEAT): acc = pl.add(acc, blk)
            for _ in range(REPEAT):
                acc = acc + blk
        # 镜像 kernel: sum_out = pl.assemble(sum_out, acc_3d, [b, 0, 0])
        out[b, :, :] = acc

    tensors["sum_out"][:] = out


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
        fn=dyn_seq_reduce,
        specs=build_tensor_specs(),
        golden_fn=golden_dyn_seq_reduce,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-2,
        atol=1e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
