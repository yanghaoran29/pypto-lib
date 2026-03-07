# pypto.program: Qwen3SingleLayerDecode
import pypto.language as pl

@pl.program
class Qwen3SingleLayerDecode:
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_0(self, hidden_states_0: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 0)], kb_0_out: pl.Scalar[pl.INDEX], sq_sum_1: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 1)], sq_sum_iter_2_outer_l0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 2)]) -> pl.Tensor[[16, 1], pl.FP32]:
        for kb_0_in, (sq_sum_iter_2_outer_l1,) in pl.parallel(0, 4, 1, init_values=(sq_sum_iter_2_outer_l0,)):
            k0_0: pl.Scalar[pl.INDEX] = (0 + (kb_0_out * 4 + kb_0_in) * 1) * 128
            _t0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 3)] = pl.tensor.view(hidden_states_0, [16, 128], [0, k0_0])
            x_chunk_0: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 4)] = pl.tensor.cast(_t0, target_type=pl.FP32, mode=2)
            _t1: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 5)] = pl.tensor.mul(x_chunk_0, x_chunk_0)
            _t2: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 6)] = pl.tensor.row_sum(_t1)
            sq_sum_4: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 7)] = pl.tensor.add(sq_sum_iter_2_outer_l1, _t2)
            sq_sum_iter_2_outer_l1_rv: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 8)] = pl.yield_(sq_sum_4)
        return sq_sum_iter_2_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_2(self, b_0: pl.Scalar[pl.INDEX], cos_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 0)], cos_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 1)], k_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 2)], k_cache_iter_1: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 3)], k_cache_iter_3_outer_l0: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 4)], k_proj_1: pl.Tensor[[16, 1024], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 5)], kvh_0_out: pl.Scalar[pl.INDEX], pos_0: pl.Scalar[pl.INT32], sin_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 6)], sin_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 7)], v_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 8)], v_cache_iter_1: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 9)], v_cache_iter_3_outer_l0: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 10)], v_proj_1: pl.Tensor[[16, 1024], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 11)]) -> tuple[pl.Tensor[[524288, 128], pl.BFLOAT16], pl.Tensor[[524288, 128], pl.BFLOAT16]]:
        mem_vec_12: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 0, 12)
        mem_vec_13: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 0, 13)
        mem_vec_14: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 0, 14)
        mem_vec_15: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 0, 15)
        cos_hi_0_tile: pl.Tile[[1, 128 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 128 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 0, 12)] = pl.block.load(cos_hi_0, [0, 0], [1, 128 // 2], [1, 128 // 2], target_memory=pl.MemorySpace.Vec)
        cos_lo_0_tile: pl.Tile[[1, 128 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 128 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 0, 13)] = pl.block.load(cos_lo_0, [0, 0], [1, 128 // 2], [1, 128 // 2], target_memory=pl.MemorySpace.Vec)
        sin_hi_0_tile: pl.Tile[[1, 128 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 128 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 0, 14)] = pl.block.load(sin_hi_0, [0, 0], [1, 128 // 2], [1, 128 // 2], target_memory=pl.MemorySpace.Vec)
        sin_lo_0_tile: pl.Tile[[1, 128 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 128 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 0, 15)] = pl.block.load(sin_lo_0, [0, 0], [1, 128 // 2], [1, 128 // 2], target_memory=pl.MemorySpace.Vec)
        for kvh_0_in, (k_cache_iter_3_outer_l1, v_cache_iter_3_outer_l1) in pl.parallel(0, 4, 1, init_values=(k_cache_iter_3_outer_l0, v_cache_iter_3_outer_l0)):
            kv_col_0: pl.Scalar[pl.INDEX] = (0 + (kvh_0_out * 4 + kvh_0_in) * 1) * 128
            _t9: pl.Tensor[[1, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 16)] = pl.tensor.view(k_proj_1, [1, 128], [b_0, kv_col_0])
            k_row_0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 17)] = pl.tensor.cast(_t9, target_type=pl.FP32, mode=2)
            k_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 18)] = pl.tensor.view(k_row_0, [1, 128 // 2], [0, 0])
            k_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 19)] = pl.tensor.view(k_row_0, [1, 128 // 2], [0, 128 // 2])
            k_rot_0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 20)] = pl.tensor.create([1, 128], dtype=pl.FP32)
            _t10: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 21)] = pl.tensor.col_expand_mul(k_lo_0, cos_lo_0)
            _t11: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 22)] = pl.tensor.col_expand_mul(k_hi_0, sin_lo_0)
            _t12: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 23)] = pl.tensor.sub(_t10, _t11)
            k_rot_1: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 24)] = pl.tensor.assemble(k_rot_0, _t12, [0, 0])
            _t13: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 25)] = pl.tensor.col_expand_mul(k_hi_0, cos_hi_0)
            _t14: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 26)] = pl.tensor.col_expand_mul(k_lo_0, sin_hi_0)
            _t15: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 27)] = pl.tensor.add(_t13, _t14)
            k_rot_2: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 28)] = pl.tensor.assemble(k_rot_1, _t15, [0, 128 // 2])
            cache_row_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + (0 + (kvh_0_out * 4 + kvh_0_in) * 1) * 4096 + pl.cast(pos_0, pl.INDEX)
            _t16: pl.Tensor[[1, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 29)] = pl.tensor.cast(k_rot_2, target_type=pl.BFLOAT16, mode=2)
            k_cache_5: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 30)] = pl.tensor.assemble(k_cache_iter_3_outer_l1, _t16, [cache_row_0, 0])
            _t17: pl.Tensor[[1, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 31)] = pl.tensor.view(v_proj_1, [1, 128], [b_0, kv_col_0])
            v_cache_5: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 32)] = pl.tensor.assemble(v_cache_iter_3_outer_l1, _t17, [cache_row_0, 0])
            k_cache_iter_3_outer_l1_rv, v_cache_iter_3_outer_l1_rv = pl.yield_(k_cache_5, v_cache_5)
        return k_cache_iter_3_outer_l1_rv, v_cache_iter_3_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_5(self, k0_5: pl.Scalar[pl.INDEX], k0_iter_9_outer_l0: pl.Scalar[pl.INDEX], kb_5_out: pl.Scalar[pl.INDEX], resid1_iter_1_outer_l0_rv: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 0)], sq_sum_6: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 1)], sq_sum_iter_7_outer_l0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 2)], x_chunk_2: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 3)], x_chunk_iter_4_outer_l0: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 4)]) -> tuple[pl.Scalar[pl.INDEX], pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 128], pl.FP32]]:
        for kb_5_in, (k0_iter_9_outer_l1, sq_sum_iter_7_outer_l1, x_chunk_iter_4_outer_l1) in pl.parallel(0, 4, 1, init_values=(k0_iter_9_outer_l0, sq_sum_iter_7_outer_l0, x_chunk_iter_4_outer_l0)):
            k0_11: pl.Scalar[pl.INDEX] = (0 + (kb_5_out * 4 + kb_5_in) * 1) * 128
            x_chunk_6: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 5)] = pl.tensor.view(resid1_iter_1_outer_l0_rv, [16, 128], [0, k0_11])
            _t40: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 6)] = pl.tensor.mul(x_chunk_6, x_chunk_6)
            _t41: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 7)] = pl.tensor.row_sum(_t40)
            sq_sum_9: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 8)] = pl.tensor.add(sq_sum_iter_7_outer_l1, _t41)
            k0_iter_9_outer_l1_rv, sq_sum_iter_7_outer_l1_rv, x_chunk_iter_4_outer_l1_rv = pl.yield_(k0_11, sq_sum_9, x_chunk_6)
        return k0_iter_9_outer_l1_rv, sq_sum_iter_7_outer_l1_rv, x_chunk_iter_4_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_7(self, down_proj_acc_3: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 0)], o0_2: pl.Scalar[pl.INDEX], o0_iter_4_outer_l0: pl.Scalar[pl.INDEX], ob_3_out: pl.Scalar[pl.INDEX], out_0: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 1)], out_iter_1_outer_l0: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 2)], resid1_iter_1_outer_l0_rv: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 3)]) -> tuple[pl.Scalar[pl.INDEX], pl.Tensor[[16, 5120], pl.BFLOAT16]]:
        for ob_3_in, (o0_iter_4_outer_l1, out_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(o0_iter_4_outer_l0, out_iter_1_outer_l0)):
            o0_6: pl.Scalar[pl.INDEX] = (0 + (ob_3_out * 8 + ob_3_in) * 1) * 64
            _t52: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 4)] = pl.tensor.view(down_proj_acc_3, [16, 64], [0, o0_6])
            _t53: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 5)] = pl.tensor.view(resid1_iter_1_outer_l0_rv, [16, 64], [0, o0_6])
            down_acc_0: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 6)] = pl.tensor.add(_t52, _t53)
            _t54: pl.Tensor[[16, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 7)] = pl.tensor.cast(down_acc_0, target_type=pl.BFLOAT16, mode=2)
            out_3: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 8)] = pl.tensor.assemble(out_iter_1_outer_l1, _t54, [0, o0_6])
            o0_iter_4_outer_l1_rv, out_iter_1_outer_l1_rv = pl.yield_(o0_6, out_3)
        return o0_iter_4_outer_l1_rv, out_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.Orchestration)
    def qwen3_decode_layer(self, hidden_states_0: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 0)], cache_pos_0: pl.Tensor[[16], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 1)], rope_cos_0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 2)], rope_sin_0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 3)], k_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 4)], v_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 5)], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 20480, 6)], wq_0: pl.Tensor[[5120, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 52428800, 7)], wk_0: pl.Tensor[[5120, 1024], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 10485760, 8)], wv_0: pl.Tensor[[5120, 1024], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 10485760, 9)], wo_0: pl.Tensor[[5120, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 52428800, 10)], post_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 20480, 11)], w_gate_0: pl.Tensor[[5120, 25600], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 262144000, 12)], w_up_0: pl.Tensor[[5120, 25600], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 262144000, 13)], w_down_0: pl.Tensor[[25600, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 262144000, 14)], out_0: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 15)]) -> pl.Tensor[[16, 5120], pl.BFLOAT16]:
        q_proj_0: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 16)] = pl.tensor.create([16, 5120], dtype=pl.BFLOAT16)
        k_proj_0: pl.Tensor[[16, 1024], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 17)] = pl.tensor.create([16, 1024], dtype=pl.BFLOAT16)
        v_proj_0: pl.Tensor[[16, 1024], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 18)] = pl.tensor.create([16, 1024], dtype=pl.BFLOAT16)
        attn_out_0: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 19)] = pl.tensor.create([16, 5120], dtype=pl.FP32)
        resid1_0: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 20)] = pl.tensor.create([16, 5120], dtype=pl.FP32)
        post_norm_0: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 21)] = pl.tensor.create([16, 5120], dtype=pl.BFLOAT16)
        sq_sum_0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 22)] = pl.tensor.create([16, 1], dtype=pl.FP32)
        sq_sum_1: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 23)] = pl.tensor.mul(sq_sum_0, 0.0)
        for kb_0_out, (sq_sum_iter_2_outer_l0,) in pl.range(0, 10, 1, init_values=(sq_sum_1,)):
            sq_sum_iter_2_outer_l1_rv: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 24)] = self.qwen3_decode_layer_incore_0(hidden_states_0, kb_0_out, sq_sum_1, sq_sum_iter_2_outer_l0)
            sq_sum_iter_2_outer_l0_rv: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 25)] = pl.yield_(sq_sum_iter_2_outer_l1_rv)
        _t3: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 26)] = pl.tensor.mul(sq_sum_iter_2_outer_l0_rv, 0.000195313)
        _t4: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 27)] = pl.tensor.add(_t3, 1e-06)
        inv_rms_0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 28)] = pl.tensor.rsqrt(_t4)
        q_proj_acc_0: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 29)] = pl.tensor.create([16, 5120], dtype=pl.FP32)
        k_proj_acc_0: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 30)] = pl.tensor.create([16, 1024], dtype=pl.FP32)
        v_proj_acc_0: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 31)] = pl.tensor.create([16, 1024], dtype=pl.FP32)
        q_proj_acc_1: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 32)] = pl.tensor.mul(q_proj_acc_0, 0.0)
        k_proj_acc_1: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 33)] = pl.tensor.mul(k_proj_acc_0, 0.0)
        v_proj_acc_1: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 34)] = pl.tensor.mul(v_proj_acc_0, 0.0)
        for kb_1, (k0_iter_1, k_proj_acc_iter_2, q_proj_acc_iter_2, v_proj_acc_iter_2, x_chunk_iter_1) in pl.range(0, 40, 1, init_values=(k0_0, k_proj_acc_1, q_proj_acc_1, v_proj_acc_1, x_chunk_0)):
            k0_3: pl.Scalar[pl.INDEX] = kb_1 * 128
            x_chunk_bf16_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 36)] = pl.tensor.view(hidden_states_0, [16, 128], [0, k0_3])
            x_chunk_3: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 37)] = pl.tensor.cast(x_chunk_bf16_0, target_type=pl.FP32, mode=2)
            gamma_0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 38)] = pl.tensor.view(input_rms_weight_0, [1, 128], [0, k0_3])
            _t5: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 39)] = pl.tensor.row_expand_mul(x_chunk_3, inv_rms_0)
            normed_0: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 40)] = pl.tensor.col_expand_mul(_t5, gamma_0)
            normed_bf16_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 41)] = pl.tensor.cast(normed_0, target_type=pl.BFLOAT16, mode=2)
            for ob_0_out, (k_proj_acc_iter_4_outer_l0, q_proj_acc_iter_4_outer_l0, v_proj_acc_iter_4_outer_l0) in pl.range(0, 10, 1, init_values=(k_proj_acc_iter_2, q_proj_acc_iter_2, v_proj_acc_iter_2)):
                ret: pl.Tuple([pl.Tensor[[16, 1024], pl.FP32], pl.Tensor[[16, 5120], pl.FP32], pl.Tensor[[16, 1024], pl.FP32]]) = self.call_group(qwen3_decode_layer_incore_1_group, k0_3, k_proj_acc_1, k_proj_acc_iter_2, k_proj_acc_iter_4_outer_l0, normed_bf16_0, ob_0_out, q_proj_acc_1, q_proj_acc_iter_2, q_proj_acc_iter_4_outer_l0, v_proj_acc_1, v_proj_acc_iter_2, v_proj_acc_iter_4_outer_l0, wk_0, wq_0, wv_0)
                k_proj_acc_iter_4_outer_l1_rv: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 42)] = ret[0]
                q_proj_acc_iter_4_outer_l1_rv: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 43)] = ret[1]
                v_proj_acc_iter_4_outer_l1_rv: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 44)] = ret[2]
                k_proj_acc_iter_4_outer_l0_rv, q_proj_acc_iter_4_outer_l0_rv, v_proj_acc_iter_4_outer_l0_rv = pl.yield_(k_proj_acc_iter_4_outer_l1_rv, q_proj_acc_iter_4_outer_l1_rv, v_proj_acc_iter_4_outer_l1_rv)
            k0_2, k_proj_acc_3, q_proj_acc_3, v_proj_acc_3, x_chunk_2 = pl.yield_(k0_3, k_proj_acc_iter_4_outer_l0_rv, q_proj_acc_iter_4_outer_l0_rv, v_proj_acc_iter_4_outer_l0_rv, x_chunk_3)
        q_proj_1: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 52)] = pl.tensor.cast(q_proj_acc_3, target_type=pl.BFLOAT16, mode=2)
        k_proj_1: pl.Tensor[[16, 1024], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 53)] = pl.tensor.cast(k_proj_acc_3, target_type=pl.BFLOAT16, mode=2)
        v_proj_1: pl.Tensor[[16, 1024], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 54)] = pl.tensor.cast(v_proj_acc_3, target_type=pl.BFLOAT16, mode=2)
        for b_0, (attn_out_iter_1, k_cache_iter_1, v_cache_iter_1) in pl.parallel(0, 16, 1, init_values=(attn_out_0, k_cache_0, v_cache_0), chunk=4):
            pos_0: pl.Scalar[pl.INT32] = pl.tensor.read(cache_pos_0, [b_0])
            ctx_len_0: pl.Scalar[pl.INDEX] = pl.cast(pos_0, pl.INDEX) + 1
            ctx_blocks_0: pl.Scalar[pl.INDEX] = (ctx_len_0 + 64 - 1) // 64
            cos_row_0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 55)] = pl.tensor.view(rope_cos_0, [1, 128], [pos_0, 0])
            sin_row_0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 56)] = pl.tensor.view(rope_sin_0, [1, 128], [pos_0, 0])
            cos_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 57)] = pl.tensor.view(cos_row_0, [1, 128 // 2], [0, 0])
            cos_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 58)] = pl.tensor.view(cos_row_0, [1, 128 // 2], [0, 128 // 2])
            sin_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 59)] = pl.tensor.view(sin_row_0, [1, 128 // 2], [0, 0])
            sin_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 60)] = pl.tensor.view(sin_row_0, [1, 128 // 2], [0, 128 // 2])
            for kvh_0_out, (k_cache_iter_3_outer_l0, v_cache_iter_3_outer_l0) in pl.range(0, 2, 1, init_values=(k_cache_iter_1, v_cache_iter_1)):
                ret: pl.Tuple([pl.Tensor[[524288, 128], pl.BFLOAT16], pl.Tensor[[524288, 128], pl.BFLOAT16]]) = self.qwen3_decode_layer_incore_2(b_0, cos_hi_0, cos_lo_0, k_cache_0, k_cache_iter_1, k_cache_iter_3_outer_l0, k_proj_1, kvh_0_out, pos_0, sin_hi_0, sin_lo_0, v_cache_0, v_cache_iter_1, v_cache_iter_3_outer_l0, v_proj_1)
                k_cache_iter_3_outer_l1_rv: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 61)] = ret[0]
                v_cache_iter_3_outer_l1_rv: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 62)] = ret[1]
                k_cache_iter_3_outer_l0_rv, v_cache_iter_3_outer_l0_rv = pl.yield_(k_cache_iter_3_outer_l1_rv, v_cache_iter_3_outer_l1_rv)
            attn_row_0: pl.Tensor[[1, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 20480, 65)] = pl.tensor.create([1, 5120], dtype=pl.FP32)
            attn_row_1: pl.Tensor[[1, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 20480, 66)] = pl.tensor.mul(attn_row_0, 0.0)
            for h_0_out, (attn_row_iter_2_outer_l0, kvh_iter_1_outer_l0) in pl.range(0, 8, 1, init_values=(attn_row_1, kvh_0)):
                ret: pl.Tuple([pl.Tensor[[1, 5120], pl.FP32], pl.Scalar[pl.INDEX]]) = self.call_group(qwen3_decode_layer_incore_3_group, attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_iter_3_outer_l0_rv, kvh_0, kvh_iter_1_outer_l0, q_proj_1, sin_hi_0, sin_lo_0, v_cache_iter_3_outer_l0_rv)
                attn_row_iter_2_outer_l1_rv: pl.Tensor[[1, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 20480, 67)] = ret[0]
                kvh_iter_1_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[1]
                attn_row_iter_2_outer_l0_rv, kvh_iter_1_outer_l0_rv = pl.yield_(attn_row_iter_2_outer_l1_rv, kvh_iter_1_outer_l1_rv)
            attn_out_3: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 69)] = pl.tensor.assemble(attn_out_iter_1, attn_row_iter_2_outer_l0_rv, [b_0, 0])
            attn_out_2, k_cache_2, v_cache_2 = pl.yield_(attn_out_3, k_cache_iter_3_outer_l0_rv, v_cache_iter_3_outer_l0_rv)
        for ob_1_out, (k0_iter_4_outer_l0, kb_iter_2_outer_l0, resid1_iter_1_outer_l0) in pl.range(0, 10, 1, init_values=(k0_2, kb_1, resid1_0)):
            ret: pl.Tuple([pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[16, 5120], pl.FP32]]) = self.call_group(qwen3_decode_layer_incore_4_group, attn_out_2, hidden_states_0, k0_2, k0_iter_4_outer_l0, kb_1, kb_iter_2_outer_l0, ob_1_out, resid1_0, resid1_iter_1_outer_l0, wo_0)
            k0_iter_4_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[0]
            kb_iter_2_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[1]
            resid1_iter_1_outer_l1_rv: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 73)] = ret[2]
            k0_iter_4_outer_l0_rv, kb_iter_2_outer_l0_rv, resid1_iter_1_outer_l0_rv = pl.yield_(k0_iter_4_outer_l1_rv, kb_iter_2_outer_l1_rv, resid1_iter_1_outer_l1_rv)
        sq_sum_5: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 75)] = pl.tensor.create([16, 1], dtype=pl.FP32)
        sq_sum_6: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 76)] = pl.tensor.mul(sq_sum_5, 0.0)
        for kb_5_out, (k0_iter_9_outer_l0, sq_sum_iter_7_outer_l0, x_chunk_iter_4_outer_l0) in pl.range(0, 10, 1, init_values=(k0_5, sq_sum_6, x_chunk_2)):
            ret: pl.Tuple([pl.Scalar[pl.INDEX], pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 128], pl.FP32]]) = self.qwen3_decode_layer_incore_5(k0_5, k0_iter_9_outer_l0, kb_5_out, resid1_iter_1_outer_l0_rv, sq_sum_6, sq_sum_iter_7_outer_l0, x_chunk_2, x_chunk_iter_4_outer_l0)
            k0_iter_9_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[0]
            sq_sum_iter_7_outer_l1_rv: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 77)] = ret[1]
            x_chunk_iter_4_outer_l1_rv: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 78)] = ret[2]
            k0_iter_9_outer_l0_rv, sq_sum_iter_7_outer_l0_rv, x_chunk_iter_4_outer_l0_rv = pl.yield_(k0_iter_9_outer_l1_rv, sq_sum_iter_7_outer_l1_rv, x_chunk_iter_4_outer_l1_rv)
        _t42: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 81)] = pl.tensor.mul(sq_sum_iter_7_outer_l0_rv, 0.000195313)
        _t43: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 82)] = pl.tensor.add(_t42, 1e-06)
        inv_rms_1: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 83)] = pl.tensor.rsqrt(_t43)
        down_proj_acc_0: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 84)] = pl.tensor.create([16, 5120], dtype=pl.FP32)
        down_proj_acc_1: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 85)] = pl.tensor.mul(down_proj_acc_0, 0.0)
        for kb_6, (gamma_iter_1, k0_iter_12, normed_iter_1, normed_bf16_iter_1, post_norm_iter_1, x_chunk_iter_7) in pl.range(0, 40, 1, init_values=(gamma_0, k0_iter_9_outer_l0_rv, normed_0, normed_bf16_0, post_norm_0, x_chunk_iter_4_outer_l0_rv)):
            k0_14: pl.Scalar[pl.INDEX] = kb_6 * 128
            x_chunk_9: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 86)] = pl.tensor.view(resid1_iter_1_outer_l0_rv, [16, 128], [0, k0_14])
            gamma_3: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 87)] = pl.tensor.view(post_rms_weight_0, [1, 128], [0, k0_14])
            _t44: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 88)] = pl.tensor.row_expand_mul(x_chunk_9, inv_rms_1)
            normed_3: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 89)] = pl.tensor.col_expand_mul(_t44, gamma_3)
            normed_bf16_3: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 90)] = pl.tensor.cast(normed_3, target_type=pl.BFLOAT16, mode=2)
            post_norm_3: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 91)] = pl.tensor.assemble(post_norm_iter_1, normed_bf16_3, [0, k0_14])
            gamma_2, k0_13, normed_2, normed_bf16_2, post_norm_2, x_chunk_8 = pl.yield_(gamma_3, k0_14, normed_3, normed_bf16_3, post_norm_3, x_chunk_9)
        for ob_2, (down_proj_acc_iter_2, k0_iter_15, kb_iter_7, o0_iter_1) in pl.range(0, 400, 1, init_values=(down_proj_acc_1, k0_13, kb_6, o0_0)):
            o0_3: pl.Scalar[pl.INDEX] = ob_2 * 64
            gate_acc_0: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 97)] = pl.tensor.create([16, 64], dtype=pl.FP32)
            up_acc_0: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 98)] = pl.tensor.create([16, 64], dtype=pl.FP32)
            gate_acc_1: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 99)] = pl.tensor.mul(gate_acc_0, 0.0)
            up_acc_1: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 100)] = pl.tensor.mul(up_acc_0, 0.0)
            for kb_9, (gate_acc_iter_2, k0_iter_17, up_acc_iter_2) in pl.range(0, 40, 1, init_values=(gate_acc_1, k0_iter_15, up_acc_1)):
                k0_19: pl.Scalar[pl.INDEX] = kb_9 * 128
                post_chunk_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 101)] = pl.tensor.view(post_norm_2, [16, 128], [0, k0_19])
                wg_0: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 102)] = pl.tensor.view(w_gate_0, [128, 64], [k0_19, o0_3])
                wu_0: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 103)] = pl.tensor.view(w_up_0, [128, 64], [k0_19, o0_3])
                _t45: pl.Tensor[[16, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 104)] = pl.tensor.matmul(post_chunk_0, wg_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                gate_acc_4: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 105)] = pl.tensor.add(gate_acc_iter_2, _t45)
                _t46: pl.Tensor[[16, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 106)] = pl.tensor.matmul(post_chunk_0, wu_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                up_acc_4: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 107)] = pl.tensor.add(up_acc_iter_2, _t46)
                gate_acc_3, k0_18, up_acc_3 = pl.yield_(gate_acc_4, k0_19, up_acc_4)
            _t47: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 110)] = pl.tensor.neg(gate_acc_3)
            _t48: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 111)] = pl.tensor.exp(_t47)
            _t49: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 112)] = pl.tensor.add(_t48, 1.0)
            sigmoid_0: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 113)] = pl.tensor.recip(_t49)
            _t50: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 114)] = pl.tensor.mul(gate_acc_3, sigmoid_0)
            mlp_chunk_0: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 115)] = pl.tensor.mul(_t50, up_acc_3)
            mlp_chunk_bf16_0: pl.Tensor[[16, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 116)] = pl.tensor.cast(mlp_chunk_0, target_type=pl.BFLOAT16, mode=2)
            for dob_0_out, (down_proj_acc_iter_4_outer_l0,) in pl.range(0, 10, 1, init_values=(down_proj_acc_iter_2,)):
                down_proj_acc_iter_4_outer_l1_rv: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 117)] = self.call_group(qwen3_decode_layer_incore_6_group, dob_0_out, down_proj_acc_1, down_proj_acc_iter_2, down_proj_acc_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0)
                down_proj_acc_iter_4_outer_l0_rv: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 118)] = pl.yield_(down_proj_acc_iter_4_outer_l1_rv)
            down_proj_acc_3, k0_16, kb_8, o0_2 = pl.yield_(down_proj_acc_iter_4_outer_l0_rv, k0_18, kb_9, o0_3)
        for ob_3_out, (o0_iter_4_outer_l0, out_iter_1_outer_l0) in pl.range(0, 10, 1, init_values=(o0_2, out_0)):
            ret: pl.Tuple([pl.Scalar[pl.INDEX], pl.Tensor[[16, 5120], pl.BFLOAT16]]) = self.qwen3_decode_layer_incore_7(down_proj_acc_3, o0_2, o0_iter_4_outer_l0, ob_3_out, out_0, out_iter_1_outer_l0, resid1_iter_1_outer_l0_rv)
            o0_iter_4_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[0]
            out_iter_1_outer_l1_rv: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 120)] = ret[1]
            o0_iter_4_outer_l0_rv, out_iter_1_outer_l0_rv = pl.yield_(o0_iter_4_outer_l1_rv, out_iter_1_outer_l1_rv)
        return out_iter_1_outer_l0_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_1_aic(self, k0_3: pl.Scalar[pl.INDEX], k_proj_acc_1: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 0)], k_proj_acc_iter_2: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 1)], k_proj_acc_iter_4_outer_l0: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 2)], normed_bf16_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 3)], ob_0_out: pl.Scalar[pl.INDEX], q_proj_acc_1: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 4)], q_proj_acc_iter_2: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 5)], q_proj_acc_iter_4_outer_l0: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 6)], v_proj_acc_1: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 7)], v_proj_acc_iter_2: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 8)], v_proj_acc_iter_4_outer_l0: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 9)], wk_0: pl.Tensor[[5120, 1024], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 10485760, 10)], wq_0: pl.Tensor[[5120, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 52428800, 11)], wv_0: pl.Tensor[[5120, 1024], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 10485760, 12)]) -> tuple[pl.Tensor[[16, 1024], pl.FP32], pl.Tensor[[16, 5120], pl.FP32], pl.Tensor[[16, 1024], pl.FP32]]:
        pl.comm.aic_initialize_pipe()
        for ob_0_in, (k_proj_acc_iter_4_outer_l1, q_proj_acc_iter_4_outer_l1, v_proj_acc_iter_4_outer_l1) in pl.parallel(0, 8, 1, init_values=(k_proj_acc_iter_4_outer_l0, q_proj_acc_iter_4_outer_l0, v_proj_acc_iter_4_outer_l0)):
            if 0 + (ob_0_out * 8 + ob_0_in) * 1 < 80:
                q0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 64
                wq_chunk_0__h0: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 13)] = pl.comm.tpop_from_aiv(0)
                wq_chunk_0__h1: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 14)] = pl.comm.tpop_from_aiv(1)
                wq_chunk_0__tmp: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 15)] = pl.tensor.create(__list__(128, 64), dtype=pl.BFLOAT16)
                wq_chunk_0__mid: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 16)] = pl.tensor.assemble(wq_chunk_0__tmp, wq_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                wq_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 17)] = pl.tensor.assemble(wq_chunk_0__mid, wq_chunk_0__h1, __list__(64, 0))
                pl.comm.tfree_to_aiv(1)
                _t6: pl.Tensor[[16, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 18)] = pl.tensor.matmul(normed_bf16_0, wq_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 19)] = pl.tensor.view(_t6, __list__(8, 64), __list__(0, 0))
                __half1__: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 20)] = pl.tensor.view(_t6, __list__(8, 64), __list__(8, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
            else:
                q_proj_acc_7: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 21)] = pl.yield_(q_proj_acc_iter_4_outer_l1)
            if 0 + (ob_0_out * 8 + ob_0_in) * 1 < 16:
                kv0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 64
                wk_chunk_0__h0: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 22)] = pl.comm.tpop_from_aiv(0)
                wk_chunk_0__h1: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 23)] = pl.comm.tpop_from_aiv(1)
                wk_chunk_0__tmp: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 24)] = pl.tensor.create(__list__(128, 64), dtype=pl.BFLOAT16)
                wk_chunk_0__mid: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 25)] = pl.tensor.assemble(wk_chunk_0__tmp, wk_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                wk_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 26)] = pl.tensor.assemble(wk_chunk_0__mid, wk_chunk_0__h1, __list__(64, 0))
                pl.comm.tfree_to_aiv(1)
                wv_chunk_0__h0: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 27)] = pl.comm.tpop_from_aiv(0)
                wv_chunk_0__h1: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 28)] = pl.comm.tpop_from_aiv(1)
                wv_chunk_0__tmp: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 29)] = pl.tensor.create(__list__(128, 64), dtype=pl.BFLOAT16)
                wv_chunk_0__mid: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 30)] = pl.tensor.assemble(wv_chunk_0__tmp, wv_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                wv_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 31)] = pl.tensor.assemble(wv_chunk_0__mid, wv_chunk_0__h1, __list__(64, 0))
                pl.comm.tfree_to_aiv(1)
                _t7: pl.Tensor[[16, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 32)] = pl.tensor.matmul(normed_bf16_0, wk_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 33)] = pl.tensor.view(_t7, __list__(8, 64), __list__(0, 0))
                __half1__: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 34)] = pl.tensor.view(_t7, __list__(8, 64), __list__(8, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
                _t8: pl.Tensor[[16, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 35)] = pl.tensor.matmul(normed_bf16_0, wv_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 36)] = pl.tensor.view(_t8, __list__(8, 64), __list__(0, 0))
                __half1__: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 37)] = pl.tensor.view(_t8, __list__(8, 64), __list__(8, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
            else:
                k_proj_acc_7, v_proj_acc_7 = pl.yield_(k_proj_acc_iter_4_outer_l1, v_proj_acc_iter_4_outer_l1)
            k_proj_acc_iter_4_outer_l1_rv, q_proj_acc_iter_4_outer_l1_rv, v_proj_acc_iter_4_outer_l1_rv = pl.yield_(k_proj_acc_7, q_proj_acc_7, v_proj_acc_7)
        return k_proj_acc_iter_4_outer_l1_rv, q_proj_acc_iter_4_outer_l1_rv, v_proj_acc_iter_4_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_1_aiv(self, k0_3: pl.Scalar[pl.INDEX], k_proj_acc_1: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 0)], k_proj_acc_iter_2: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 1)], k_proj_acc_iter_4_outer_l0: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 2)], normed_bf16_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 3)], ob_0_out: pl.Scalar[pl.INDEX], q_proj_acc_1: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 4)], q_proj_acc_iter_2: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 5)], q_proj_acc_iter_4_outer_l0: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 6)], v_proj_acc_1: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 7)], v_proj_acc_iter_2: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 8)], v_proj_acc_iter_4_outer_l0: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 9)], wk_0: pl.Tensor[[5120, 1024], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 10485760, 10)], wq_0: pl.Tensor[[5120, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 52428800, 11)], wv_0: pl.Tensor[[5120, 1024], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 10485760, 12)], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[16, 1024], pl.FP32], pl.Tensor[[16, 5120], pl.FP32], pl.Tensor[[16, 1024], pl.FP32]]:
        pl.comm.aiv_initialize_pipe()
        for ob_0_in, (k_proj_acc_iter_4_outer_l1, q_proj_acc_iter_4_outer_l1, v_proj_acc_iter_4_outer_l1) in pl.parallel(0, 8, 1, init_values=(k_proj_acc_iter_4_outer_l0, q_proj_acc_iter_4_outer_l0, v_proj_acc_iter_4_outer_l0)):
            if 0 + (ob_0_out * 8 + ob_0_in) * 1 < 80:
                q0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 64
                q_prev_0: pl.Tensor[[8, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 13)] = pl.tensor.view(q_proj_acc_iter_4_outer_l1, [8, 64], [0, q0_0])
                wq_chunk_0: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 14)] = pl.tensor.view(wq_0, [64, 64], [k0_3 + AIV_IDX * 64, q0_0])
                pl.comm.tpush_to_aic(wq_chunk_0, AIV_IDX)
                _t6: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 16)] = pl.comm.tpop_from_aic(AIV_IDX)
                q_next_0: pl.Tensor[[8, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 19)] = pl.tensor.add(q_prev_0, _t6)
                pl.comm.tfree_to_aic(AIV_IDX)
                q_proj_acc_6: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 21)] = pl.tensor.assemble(q_proj_acc_iter_4_outer_l1, q_next_0, [0 + AIV_IDX * 8, q0_0])
                q_proj_acc_7: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 22)] = pl.yield_(q_proj_acc_6)
            else:
                q_proj_acc_7: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 22)] = pl.yield_(q_proj_acc_iter_4_outer_l1)
            if 0 + (ob_0_out * 8 + ob_0_in) * 1 < 16:
                kv0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 64
                k_prev_0: pl.Tensor[[8, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 23)] = pl.tensor.view(k_proj_acc_iter_4_outer_l1, [8, 64], [0, kv0_0])
                v_prev_0: pl.Tensor[[8, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 24)] = pl.tensor.view(v_proj_acc_iter_4_outer_l1, [8, 64], [0, kv0_0])
                wk_chunk_0: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 25)] = pl.tensor.view(wk_0, [64, 64], [k0_3 + AIV_IDX * 64, kv0_0])
                pl.comm.tpush_to_aic(wk_chunk_0, AIV_IDX)
                wv_chunk_0: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 27)] = pl.tensor.view(wv_0, [64, 64], [k0_3 + AIV_IDX * 64, kv0_0])
                pl.comm.tpush_to_aic(wv_chunk_0, AIV_IDX)
                _t7: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 29)] = pl.comm.tpop_from_aic(AIV_IDX)
                k_next_0: pl.Tensor[[8, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 32)] = pl.tensor.add(k_prev_0, _t7)
                pl.comm.tfree_to_aic(AIV_IDX)
                _t8: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 33)] = pl.comm.tpop_from_aic(AIV_IDX)
                v_next_0: pl.Tensor[[8, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 36)] = pl.tensor.add(v_prev_0, _t8)
                pl.comm.tfree_to_aic(AIV_IDX)
                k_proj_acc_6: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 38)] = pl.tensor.assemble(k_proj_acc_iter_4_outer_l1, k_next_0, [0 + AIV_IDX * 8, kv0_0])
                v_proj_acc_6: pl.Tensor[[16, 1024], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 40)] = pl.tensor.assemble(v_proj_acc_iter_4_outer_l1, v_next_0, [0 + AIV_IDX * 8, kv0_0])
                k_proj_acc_7, v_proj_acc_7 = pl.yield_(k_proj_acc_6, v_proj_acc_6)
            else:
                k_proj_acc_7, v_proj_acc_7 = pl.yield_(k_proj_acc_iter_4_outer_l1, v_proj_acc_iter_4_outer_l1)
            k_proj_acc_iter_4_outer_l1_rv, q_proj_acc_iter_4_outer_l1_rv, v_proj_acc_iter_4_outer_l1_rv = pl.yield_(k_proj_acc_7, q_proj_acc_7, v_proj_acc_7)
        return k_proj_acc_iter_4_outer_l1_rv, q_proj_acc_iter_4_outer_l1_rv, v_proj_acc_iter_4_outer_l1_rv
    @pl.function_group(aic="qwen3_decode_layer_incore_1_aic", aiv="qwen3_decode_layer_incore_1_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_decode_layer_incore_1_group:
        """Parameter passing:
          call_group(qwen3_decode_layer_incore_1_group, k0_3, k_proj_acc_1, k_proj_acc_iter_2, k_proj_acc_iter_4_outer_l0, normed_bf16_0, ob_0_out, q_proj_acc_1, q_proj_acc_iter_2, q_proj_acc_iter_4_outer_l0, v_proj_acc_1, v_proj_acc_iter_2, v_proj_acc_iter_4_outer_l0, wk_0, wq_0, wv_0)
            → qwen3_decode_layer_incore_1_aic(k0_3, k_proj_acc_1, k_proj_acc_iter_2, k_proj_acc_iter_4_outer_l0, normed_bf16_0, ob_0_out, q_proj_acc_1, q_proj_acc_iter_2, q_proj_acc_iter_4_outer_l0, v_proj_acc_1, v_proj_acc_iter_2, v_proj_acc_iter_4_outer_l0, wk_0, wq_0, wv_0)
            → qwen3_decode_layer_incore_1_aiv(k0_3, k_proj_acc_1, k_proj_acc_iter_2, k_proj_acc_iter_4_outer_l0, normed_bf16_0, ob_0_out, q_proj_acc_1, q_proj_acc_iter_2, q_proj_acc_iter_4_outer_l0, v_proj_acc_1, v_proj_acc_iter_2, v_proj_acc_iter_4_outer_l0, wk_0, wq_0, wv_0, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_3_aic(self, attn_row_1: pl.Tensor[[1, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 20480, 0)], attn_row_iter_2_outer_l0: pl.Tensor[[1, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 20480, 1)], b_0: pl.Scalar[pl.INDEX], cos_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 2)], cos_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 3)], ctx_blocks_0: pl.Scalar[pl.INDEX], ctx_len_0: pl.Scalar[pl.INDEX], h_0_out: pl.Scalar[pl.INDEX], k_cache_iter_3_outer_l0_rv: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 4)], kvh_0: pl.Scalar[pl.INDEX], kvh_iter_1_outer_l0: pl.Scalar[pl.INDEX], q_proj_1: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 5)], sin_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 6)], sin_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 7)], v_cache_iter_3_outer_l0_rv: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 8)]) -> tuple[pl.Tensor[[1, 5120], pl.FP32], pl.Scalar[pl.INDEX]]:
        pl.comm.aic_initialize_pipe()
        for h_0_in, (attn_row_iter_2_outer_l1, kvh_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(attn_row_iter_2_outer_l0, kvh_iter_1_outer_l0)):
            kvh_3: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) // 8
            q_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 128
            q_rot_bf16_0: pl.Tensor[[1, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 9)] = pl.comm.tpop_from_aiv(0)
            q_rot_bf16_0__discard: pl.Tensor[[1, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 10)] = pl.comm.tpop_from_aiv(1)
            pl.comm.tfree_to_aiv(1)
            for sb_0 in pl.range(0, ctx_blocks_0, 1):
                s0_0: pl.Scalar[pl.INDEX] = sb_0 * 64
                valid_len_0: pl.Scalar[pl.INDEX] = min(64, ctx_len_0 - s0_0)
                cache_row0_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_3 * 4096 + s0_0
                k_tile_0__h0: pl.Tensor[[32, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 11)] = pl.comm.tpop_from_aiv(0)
                k_tile_0__h1: pl.Tensor[[32, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 12)] = pl.comm.tpop_from_aiv(1)
                k_tile_0__tmp: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 13)] = pl.tensor.create(__list__(64, 128), dtype=pl.BFLOAT16)
                k_tile_0__mid: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 14)] = pl.tensor.assemble(k_tile_0__tmp, k_tile_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                k_tile_0: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 15)] = pl.tensor.assemble(k_tile_0__mid, k_tile_0__h1, __list__(32, 0))
                pl.comm.tfree_to_aiv(1)
                v_tile_0__h0: pl.Tensor[[32, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 16)] = pl.comm.tpop_from_aiv(0)
                v_tile_0__h1: pl.Tensor[[32, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 17)] = pl.comm.tpop_from_aiv(1)
                v_tile_0__tmp: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 18)] = pl.tensor.create(__list__(64, 128), dtype=pl.BFLOAT16)
                v_tile_0__mid: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 19)] = pl.tensor.assemble(v_tile_0__tmp, v_tile_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                v_tile_0: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 20)] = pl.tensor.assemble(v_tile_0__mid, v_tile_0__h1, __list__(32, 0))
                pl.comm.tfree_to_aiv(1)
                _t25: pl.Tensor[[1, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 21)] = pl.tensor.matmul(q_rot_bf16_0, k_tile_0, a_trans=False, b_trans=True, c_matrix_nz=False)
                scores_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 22)] = pl.tensor.mul(_t25, 0.0883883)
                _t29: pl.Tensor[[1, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 23)] = pl.comm.tpop_from_aiv()
                oi_tmp_0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 24)] = pl.tensor.matmul(_t29, v_tile_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                pl.comm.tfree_to_aiv()
                if sb_0 == 0:
                    oi_4: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 25)] = oi_tmp_0
                    li_6, mi_6, oi_6 = pl.yield_(oi_4)
                else:

                pl.yield_(li_6, mi_6, oi_6)
            pl.comm.tfree_to_aiv(0)
            attn_row_iter_2_outer_l1_rv, kvh_iter_1_outer_l1_rv = pl.yield_(kvh_3)
        return attn_row_iter_2_outer_l1_rv, kvh_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_3_aiv(self, attn_row_1: pl.Tensor[[1, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 20480, 0)], attn_row_iter_2_outer_l0: pl.Tensor[[1, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 20480, 1)], b_0: pl.Scalar[pl.INDEX], cos_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 2)], cos_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 3)], ctx_blocks_0: pl.Scalar[pl.INDEX], ctx_len_0: pl.Scalar[pl.INDEX], h_0_out: pl.Scalar[pl.INDEX], k_cache_iter_3_outer_l0_rv: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 4)], kvh_0: pl.Scalar[pl.INDEX], kvh_iter_1_outer_l0: pl.Scalar[pl.INDEX], q_proj_1: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 5)], sin_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 6)], sin_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 7)], v_cache_iter_3_outer_l0_rv: pl.Tensor[[524288, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 134217728, 8)], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[1, 5120], pl.FP32], pl.Scalar[pl.INDEX]]:
        mem_vec_9: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 0, 9)
        mem_vec_10: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 0, 10)
        mem_vec_11: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 0, 11)
        mem_vec_12: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 0, 12)
        cos_hi_0_tile: pl.Tile[[1, 128 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 128 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 0, 9)] = pl.block.load(cos_hi_0, [0, 0], [1, 128 // 2], [1, 128 // 2], target_memory=pl.MemorySpace.Vec)
        cos_lo_0_tile: pl.Tile[[1, 128 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 128 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 0, 10)] = pl.block.load(cos_lo_0, [0, 0], [1, 128 // 2], [1, 128 // 2], target_memory=pl.MemorySpace.Vec)
        sin_hi_0_tile: pl.Tile[[1, 128 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 128 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 0, 11)] = pl.block.load(sin_hi_0, [0, 0], [1, 128 // 2], [1, 128 // 2], target_memory=pl.MemorySpace.Vec)
        sin_lo_0_tile: pl.Tile[[1, 128 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 128 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 0, 12)] = pl.block.load(sin_lo_0, [0, 0], [1, 128 // 2], [1, 128 // 2], target_memory=pl.MemorySpace.Vec)
        pl.comm.aiv_initialize_pipe()
        for h_0_in, (attn_row_iter_2_outer_l1, kvh_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(attn_row_iter_2_outer_l0, kvh_iter_1_outer_l0)):
            kvh_3: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) // 8
            q_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 128
            _t18: pl.Tensor[[1, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 13)] = pl.tensor.view(q_proj_1, [1, 64], [b_0, q_col_0 + AIV_IDX * 64])
            q_row_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 15)] = pl.tensor.cast(_t18, target_type=pl.FP32, mode=2)
            q_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 17)] = pl.tensor.deep_view(q_row_0, [1, 128 // 2], [0, 0])
            q_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 18)] = pl.tensor.deep_view(q_row_0, [1, 128 // 2], [0, 128 // 2])
            q_rot_0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 19)] = pl.tensor.create([1, 128], dtype=pl.FP32)
            _t19: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 20)] = pl.tensor.col_expand_mul(q_lo_0, cos_lo_0)
            _t20: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 21)] = pl.tensor.col_expand_mul(q_hi_0, sin_lo_0)
            _t21: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 22)] = pl.tensor.sub(_t19, _t20)
            q_rot_1: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 23)] = pl.tensor.assemble(q_rot_0, _t21, [0, 0])
            _t22: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 24)] = pl.tensor.col_expand_mul(q_hi_0, cos_hi_0)
            _t23: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 25)] = pl.tensor.col_expand_mul(q_lo_0, sin_hi_0)
            _t24: pl.Tensor[[1, 128 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 26)] = pl.tensor.add(_t22, _t23)
            q_rot_2: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 27)] = pl.tensor.assemble(q_rot_1, _t24, [0, 128 // 2])
            q_rot_bf16_0: pl.Tensor[[1, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 28)] = pl.tensor.cast(q_rot_2, target_type=pl.BFLOAT16, mode=2)
            pl.comm.tpush_to_aic(q_rot_bf16_0, AIV_IDX)
            oi_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 29)] = pl.tensor.create([1, 64], dtype=pl.FP32)
            li_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 30)] = pl.tensor.create([1, 1], dtype=pl.FP32)
            mi_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 31)] = pl.tensor.create([1, 1], dtype=pl.FP32)
            oi_1: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 33)] = pl.tensor.mul(oi_0, 0.0)
            li_1: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 34)] = pl.tensor.mul(li_0, 0.0)
            mi_1: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 35)] = pl.tensor.mul(mi_0, 0.0)
            for sb_0, (li_iter_2, mi_iter_2, oi_iter_2) in pl.range(0, ctx_blocks_0, 1, init_values=(li_1, mi_1, oi_1)):
                s0_0: pl.Scalar[pl.INDEX] = sb_0 * 64
                valid_len_0: pl.Scalar[pl.INDEX] = min(64, ctx_len_0 - s0_0)
                cache_row0_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_3 * 4096 + s0_0
                k_tile_0: pl.Tensor[[32, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 36)] = pl.tensor.view(k_cache_iter_3_outer_l0_rv, [32, 128], [cache_row0_0 + AIV_IDX * 32, 0])
                pl.comm.tpush_to_aic(k_tile_0, AIV_IDX)
                v_tile_0: pl.Tensor[[32, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 38)] = pl.tensor.view(v_cache_iter_3_outer_l0_rv, [32, 128], [cache_row0_0 + AIV_IDX * 32, 0])
                pl.comm.tpush_to_aic(v_tile_0, AIV_IDX)
                exp_pad_0: pl.Tensor[[1, 32], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 40)] = pl.tensor.create([1, 32], dtype=pl.FP32)
                exp_pad_1: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 42)] = pl.tensor.mul(exp_pad_0, 0.0)
            ctx_0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 46)] = pl.tensor.row_expand_div(oi_3, li_3)
            attn_row_4: pl.Tensor[[1, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 20480, 47)] = pl.tensor.assemble(attn_row_iter_2_outer_l1, ctx_0, [0, q_col_0])
            attn_row_iter_2_outer_l1_rv, kvh_iter_1_outer_l1_rv = pl.yield_(attn_row_4, kvh_3)
        return attn_row_iter_2_outer_l1_rv, kvh_iter_1_outer_l1_rv
    @pl.function_group(aic="qwen3_decode_layer_incore_3_aic", aiv="qwen3_decode_layer_incore_3_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_decode_layer_incore_3_group:
        """Parameter passing:
          call_group(qwen3_decode_layer_incore_3_group, attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_iter_3_outer_l0_rv, kvh_0, kvh_iter_1_outer_l0, q_proj_1, sin_hi_0, sin_lo_0, v_cache_iter_3_outer_l0_rv)
            → qwen3_decode_layer_incore_3_aic(attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_iter_3_outer_l0_rv, kvh_0, kvh_iter_1_outer_l0, q_proj_1, sin_hi_0, sin_lo_0, v_cache_iter_3_outer_l0_rv)
            → qwen3_decode_layer_incore_3_aiv(attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_iter_3_outer_l0_rv, kvh_0, kvh_iter_1_outer_l0, q_proj_1, sin_hi_0, sin_lo_0, v_cache_iter_3_outer_l0_rv, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_4_aic(self, attn_out_2: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 0)], hidden_states_0: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 1)], k0_2: pl.Scalar[pl.INDEX], k0_iter_4_outer_l0: pl.Scalar[pl.INDEX], kb_1: pl.Scalar[pl.INDEX], kb_iter_2_outer_l0: pl.Scalar[pl.INDEX], ob_1_out: pl.Scalar[pl.INDEX], resid1_0: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 2)], resid1_iter_1_outer_l0: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 3)], wo_0: pl.Tensor[[5120, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 52428800, 4)]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[16, 5120], pl.FP32]]:
        pl.comm.aic_initialize_pipe()
        for ob_1_in, (k0_iter_4_outer_l1, kb_iter_2_outer_l1, resid1_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(k0_iter_4_outer_l0, kb_iter_2_outer_l0, resid1_iter_1_outer_l0)):
            o0_0: pl.Scalar[pl.INDEX] = (0 + (ob_1_out * 8 + ob_1_in) * 1) * 64
            for kb_4, (k0_iter_6,) in pl.range(0, 40, 1, init_values=(k0_iter_4_outer_l1,)):
                k0_8: pl.Scalar[pl.INDEX] = kb_4 * 128
                a_chunk_0__h0: pl.Tensor[[8, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 5)] = pl.comm.tpop_from_aiv(0)
                a_chunk_0__h1: pl.Tensor[[8, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 6)] = pl.comm.tpop_from_aiv(1)
                a_chunk_0__tmp: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 7)] = pl.tensor.create(__list__(16, 128), dtype=pl.BFLOAT16)
                a_chunk_0__mid: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 8)] = pl.tensor.assemble(a_chunk_0__tmp, a_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                a_chunk_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 9)] = pl.tensor.assemble(a_chunk_0__mid, a_chunk_0__h1, __list__(8, 0))
                pl.comm.tfree_to_aiv(1)
                w_chunk_0__h0: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 10)] = pl.comm.tpop_from_aiv(0)
                w_chunk_0__h1: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 11)] = pl.comm.tpop_from_aiv(1)
                w_chunk_0__tmp: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 12)] = pl.tensor.create(__list__(128, 64), dtype=pl.BFLOAT16)
                w_chunk_0__mid: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 13)] = pl.tensor.assemble(w_chunk_0__tmp, w_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                w_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 14)] = pl.tensor.assemble(w_chunk_0__mid, w_chunk_0__h1, __list__(64, 0))
                pl.comm.tfree_to_aiv(1)
                _t37: pl.Tensor[[16, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 15)] = pl.tensor.matmul(a_chunk_0, w_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 16)] = pl.tensor.view(_t37, __list__(8, 64), __list__(0, 0))
                __half1__: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 17)] = pl.tensor.view(_t37, __list__(8, 64), __list__(8, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
                k0_7: pl.Scalar[pl.INDEX] = pl.yield_(k0_8)
            k0_iter_4_outer_l1_rv, kb_iter_2_outer_l1_rv, resid1_iter_1_outer_l1_rv = pl.yield_(k0_7, kb_4)
        return k0_iter_4_outer_l1_rv, kb_iter_2_outer_l1_rv, resid1_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_4_aiv(self, attn_out_2: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 0)], hidden_states_0: pl.Tensor[[16, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 163840, 1)], k0_2: pl.Scalar[pl.INDEX], k0_iter_4_outer_l0: pl.Scalar[pl.INDEX], kb_1: pl.Scalar[pl.INDEX], kb_iter_2_outer_l0: pl.Scalar[pl.INDEX], ob_1_out: pl.Scalar[pl.INDEX], resid1_0: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 2)], resid1_iter_1_outer_l0: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 3)], wo_0: pl.Tensor[[5120, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 52428800, 4)], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[16, 5120], pl.FP32]]:
        pl.comm.aiv_initialize_pipe()
        for ob_1_in, (k0_iter_4_outer_l1, kb_iter_2_outer_l1, resid1_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(k0_iter_4_outer_l0, kb_iter_2_outer_l0, resid1_iter_1_outer_l0)):
            o0_0: pl.Scalar[pl.INDEX] = (0 + (ob_1_out * 8 + ob_1_in) * 1) * 64
            o_acc_0: pl.Tensor[[8, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 5)] = pl.tensor.create([8, 64], dtype=pl.FP32)
            o_acc_1: pl.Tensor[[16, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 7)] = pl.tensor.mul(o_acc_0, 0.0)
            for kb_4, (k0_iter_6, o_acc_iter_2) in pl.range(0, 40, 1, init_values=(k0_iter_4_outer_l1, o_acc_1)):
                k0_8: pl.Scalar[pl.INDEX] = kb_4 * 128
                _t36: pl.Tensor[[8, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 8)] = pl.tensor.view(attn_out_2, [8, 128], [0 + AIV_IDX * 8, k0_8])
                a_chunk_0: pl.Tensor[[8, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 10)] = pl.tensor.cast(_t36, target_type=pl.BFLOAT16, mode=2)
                pl.comm.tpush_to_aic(a_chunk_0, AIV_IDX)
                w_chunk_0: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 12)] = pl.tensor.view(wo_0, [64, 64], [k0_8 + AIV_IDX * 64, o0_0])
                pl.comm.tpush_to_aic(w_chunk_0, AIV_IDX)
                _t37: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 14)] = pl.comm.tpop_from_aic(AIV_IDX)
                o_acc_4: pl.Tensor[[8, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 16)] = pl.tensor.add(o_acc_iter_2, _t37)
                pl.comm.tfree_to_aic(AIV_IDX)
                k0_7, o_acc_3 = pl.yield_(k0_8, o_acc_4)
            _t38: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 19)] = pl.tensor.view(hidden_states_0, [8, 64], [0 + AIV_IDX * 8, o0_0])
            resid_0: pl.Tensor[[8, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 21)] = pl.tensor.cast(_t38, target_type=pl.FP32, mode=2)
            _t39: pl.Tensor[[8, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 23)] = pl.tensor.add(o_acc_3, resid_0)
            resid1_3: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 25)] = pl.tensor.assemble(resid1_iter_1_outer_l1, _t39, [0 + AIV_IDX * 8, o0_0])
            k0_iter_4_outer_l1_rv, kb_iter_2_outer_l1_rv, resid1_iter_1_outer_l1_rv = pl.yield_(k0_7, kb_4, resid1_3)
        return k0_iter_4_outer_l1_rv, kb_iter_2_outer_l1_rv, resid1_iter_1_outer_l1_rv
    @pl.function_group(aic="qwen3_decode_layer_incore_4_aic", aiv="qwen3_decode_layer_incore_4_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_decode_layer_incore_4_group:
        """Parameter passing:
          call_group(qwen3_decode_layer_incore_4_group, attn_out_2, hidden_states_0, k0_2, k0_iter_4_outer_l0, kb_1, kb_iter_2_outer_l0, ob_1_out, resid1_0, resid1_iter_1_outer_l0, wo_0)
            → qwen3_decode_layer_incore_4_aic(attn_out_2, hidden_states_0, k0_2, k0_iter_4_outer_l0, kb_1, kb_iter_2_outer_l0, ob_1_out, resid1_0, resid1_iter_1_outer_l0, wo_0)
            → qwen3_decode_layer_incore_4_aiv(attn_out_2, hidden_states_0, k0_2, k0_iter_4_outer_l0, kb_1, kb_iter_2_outer_l0, ob_1_out, resid1_0, resid1_iter_1_outer_l0, wo_0, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_6_aic(self, dob_0_out: pl.Scalar[pl.INDEX], down_proj_acc_1: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 0)], down_proj_acc_iter_2: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 1)], down_proj_acc_iter_4_outer_l0: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 2)], mlp_chunk_bf16_0: pl.Tensor[[16, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 3)], o0_3: pl.Scalar[pl.INDEX], w_down_0: pl.Tensor[[25600, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 262144000, 4)]) -> pl.Tensor[[16, 5120], pl.FP32]:
        pl.comm.aic_initialize_pipe()
        for dob_0_in, (down_proj_acc_iter_4_outer_l1,) in pl.parallel(0, 8, 1, init_values=(down_proj_acc_iter_4_outer_l0,)):
            d0_0: pl.Scalar[pl.INDEX] = (0 + (dob_0_out * 8 + dob_0_in) * 1) * 64
            w_down_chunk_0__h0: pl.Tensor[[32, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 5)] = pl.comm.tpop_from_aiv(0)
            w_down_chunk_0__h1: pl.Tensor[[32, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 6)] = pl.comm.tpop_from_aiv(1)
            w_down_chunk_0__tmp: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 7)] = pl.tensor.create(__list__(64, 64), dtype=pl.BFLOAT16)
            w_down_chunk_0__mid: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 8)] = pl.tensor.assemble(w_down_chunk_0__tmp, w_down_chunk_0__h0, __list__(0, 0))
            pl.comm.tfree_to_aiv(0)
            w_down_chunk_0: pl.Tensor[[64, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 9)] = pl.tensor.assemble(w_down_chunk_0__mid, w_down_chunk_0__h1, __list__(32, 0))
            pl.comm.tfree_to_aiv(1)
            _t51: pl.Tensor[[16, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 10)] = pl.tensor.matmul(mlp_chunk_bf16_0, w_down_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
            __half0__: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 11)] = pl.tensor.view(_t51, __list__(8, 64), __list__(0, 0))
            __half1__: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 12)] = pl.tensor.view(_t51, __list__(8, 64), __list__(8, 0))
            pl.comm.tpush_to_aiv(__half0__, 0)
            pl.comm.tpush_to_aiv(__half1__, 1)
        return down_proj_acc_iter_4_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_6_aiv(self, dob_0_out: pl.Scalar[pl.INDEX], down_proj_acc_1: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 0)], down_proj_acc_iter_2: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 1)], down_proj_acc_iter_4_outer_l0: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 2)], mlp_chunk_bf16_0: pl.Tensor[[16, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 3)], o0_3: pl.Scalar[pl.INDEX], w_down_0: pl.Tensor[[25600, 5120], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 262144000, 4)], AIV_IDX: pl.Scalar[pl.INDEX]) -> pl.Tensor[[16, 5120], pl.FP32]:
        pl.comm.aiv_initialize_pipe()
        for dob_0_in, (down_proj_acc_iter_4_outer_l1,) in pl.parallel(0, 8, 1, init_values=(down_proj_acc_iter_4_outer_l0,)):
            d0_0: pl.Scalar[pl.INDEX] = (0 + (dob_0_out * 8 + dob_0_in) * 1) * 64
            down_prev_0: pl.Tensor[[8, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 5)] = pl.tensor.view(down_proj_acc_iter_4_outer_l1, [8, 64], [0, d0_0])
            w_down_chunk_0: pl.Tensor[[32, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 6)] = pl.tensor.view(w_down_0, [32, 64], [o0_3 + AIV_IDX * 32, d0_0])
            pl.comm.tpush_to_aic(w_down_chunk_0, AIV_IDX)
            _t51: pl.Tensor[[8, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 8)] = pl.comm.tpop_from_aic(AIV_IDX)
            down_next_0: pl.Tensor[[8, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 11)] = pl.tensor.add(down_prev_0, _t51)
            pl.comm.tfree_to_aic(AIV_IDX)
            down_proj_acc_6: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 13)] = pl.tensor.assemble(down_proj_acc_iter_4_outer_l1, down_next_0, [0 + AIV_IDX * 8, d0_0])
            down_proj_acc_iter_4_outer_l1_rv: pl.Tensor[[16, 5120], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 327680, 14)] = pl.yield_(down_proj_acc_6)
        return down_proj_acc_iter_4_outer_l1_rv
    @pl.function_group(aic="qwen3_decode_layer_incore_6_aic", aiv="qwen3_decode_layer_incore_6_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_decode_layer_incore_6_group:
        """Parameter passing:
          call_group(qwen3_decode_layer_incore_6_group, dob_0_out, down_proj_acc_1, down_proj_acc_iter_2, down_proj_acc_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0)
            → qwen3_decode_layer_incore_6_aic(dob_0_out, down_proj_acc_1, down_proj_acc_iter_2, down_proj_acc_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0)
            → qwen3_decode_layer_incore_6_aiv(dob_0_out, down_proj_acc_1, down_proj_acc_iter_2, down_proj_acc_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0, AIV_IDX=<runtime>)
        """
        pass
