# pypto.program: Qwen3SingleLayerDecode
import pypto.language as pl

@pl.program
class Qwen3SingleLayerDecode:
    @pl.function
    def qwen3_decode_layer(self, hidden_states_0: pl.Tensor[[16, 5120], pl.BFLOAT16], cache_pos_0: pl.Tensor[[16], pl.INT32], rope_cos_0: pl.Tensor[[4096, 128], pl.FP32], rope_sin_0: pl.Tensor[[4096, 128], pl.FP32], k_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], wq_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], wk_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], wv_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], wo_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], post_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], w_gate_0: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_up_0: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_down_0: pl.Tensor[[25600, 5120], pl.BFLOAT16], out_0: pl.Tensor[[16, 5120], pl.BFLOAT16]) -> pl.Tensor[[16, 5120], pl.BFLOAT16]:
        q_proj_0: pl.Tensor[[16, 5120], pl.BFLOAT16] = pl.tensor.create([16, 5120], dtype=pl.BFLOAT16)
        k_proj_0: pl.Tensor[[16, 1024], pl.BFLOAT16] = pl.tensor.create([16, 1024], dtype=pl.BFLOAT16)
        v_proj_0: pl.Tensor[[16, 1024], pl.BFLOAT16] = pl.tensor.create([16, 1024], dtype=pl.BFLOAT16)
        attn_out_0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32)
        resid1_0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32)
        post_norm_0: pl.Tensor[[16, 5120], pl.BFLOAT16] = pl.tensor.create([16, 5120], dtype=pl.BFLOAT16)
        with pl.auto_incore():
            sq_sum_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
            sq_sum_1: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.mul(sq_sum_0, 0.0)
            for kb_0_out, (sq_sum_iter_2_outer,) in pl.range(0, 10, 1, init_values=(sq_sum_1,)):
                for kb_0_in, (sq_sum_iter_2_inner,) in pl.parallel(0, 4, 1, init_values=(sq_sum_iter_2_outer,)):
                    k0_0: pl.Scalar[pl.INDEX] = (0 + (kb_0_out * 4 + kb_0_in) * 1) * 128
                    _t0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [16, 128], [0, k0_0])
                    x_chunk_0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.cast(_t0, target_type=pl.FP32, mode=2)
                    _t1: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.mul(x_chunk_0, x_chunk_0)
                    _t2: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(_t1)
                    sq_sum_4: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(sq_sum_iter_2_inner, _t2)
                    sq_sum_iter_2_inner_rv: pl.Tensor[[16, 1], pl.FP32] = pl.yield_(sq_sum_4)
                sq_sum_iter_2_outer_rv: pl.Tensor[[16, 1], pl.FP32] = pl.yield_(sq_sum_iter_2_inner_rv)
            _t3: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.mul(sq_sum_iter_2_outer_rv, 0.000195313)
            _t4: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(_t3, 1e-06)
            inv_rms_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.rsqrt(_t4)
            q_proj_acc_0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32)
            k_proj_acc_0: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.create([16, 1024], dtype=pl.FP32)
            v_proj_acc_0: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.create([16, 1024], dtype=pl.FP32)
            q_proj_acc_1: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.mul(q_proj_acc_0, 0.0)
            k_proj_acc_1: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.mul(k_proj_acc_0, 0.0)
            v_proj_acc_1: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.mul(v_proj_acc_0, 0.0)
            for kb_1, (k0_iter_1, k_proj_acc_iter_2, q_proj_acc_iter_2, v_proj_acc_iter_2, x_chunk_iter_1) in pl.range(0, 40, 1, init_values=(k0_0, k_proj_acc_1, q_proj_acc_1, v_proj_acc_1, x_chunk_0)):
                k0_3: pl.Scalar[pl.INDEX] = kb_1 * 128
                x_chunk_bf16_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [16, 128], [0, k0_3])
                x_chunk_3: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.cast(x_chunk_bf16_0, target_type=pl.FP32, mode=2)
                gamma_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(input_rms_weight_0, [1, 128], [0, k0_3])
                _t5: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_3, inv_rms_0)
                normed_0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.col_expand_mul(_t5, gamma_0)
                normed_bf16_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.cast(normed_0, target_type=pl.BFLOAT16, mode=2)
                for ob_0_out, (k_proj_acc_iter_4_outer, q_proj_acc_iter_4_outer, v_proj_acc_iter_4_outer) in pl.range(0, 10, 1, init_values=(k_proj_acc_iter_2, q_proj_acc_iter_2, v_proj_acc_iter_2)):
                    for ob_0_in, (k_proj_acc_iter_4_inner, q_proj_acc_iter_4_inner, v_proj_acc_iter_4_inner) in pl.parallel(0, 8, 1, init_values=(k_proj_acc_iter_4_outer, q_proj_acc_iter_4_outer, v_proj_acc_iter_4_outer)):
                        if 0 + (ob_0_out * 8 + ob_0_in) * 1 < 80:
                            q0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 64
                            q_prev_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.view(q_proj_acc_iter_4_inner, [16, 64], [0, q0_0])
                            wq_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.tensor.view(wq_0, [128, 64], [k0_3, q0_0])
                            _t6: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.tensor.matmul(normed_bf16_0, wq_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                            q_next_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(q_prev_0, _t6)
                            q_proj_acc_6: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_acc_iter_4_inner, q_next_0, [0, q0_0])
                            q_proj_acc_7: pl.Tensor[[16, 5120], pl.FP32] = pl.yield_(q_proj_acc_6)
                        else:
                            q_proj_acc_7: pl.Tensor[[16, 5120], pl.FP32] = pl.yield_(q_proj_acc_iter_4_inner)
                        if 0 + (ob_0_out * 8 + ob_0_in) * 1 < 16:
                            kv0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 64
                            k_prev_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.view(k_proj_acc_iter_4_inner, [16, 64], [0, kv0_0])
                            v_prev_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.view(v_proj_acc_iter_4_inner, [16, 64], [0, kv0_0])
                            wk_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.tensor.view(wk_0, [128, 64], [k0_3, kv0_0])
                            wv_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.tensor.view(wv_0, [128, 64], [k0_3, kv0_0])
                            _t7: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.tensor.matmul(normed_bf16_0, wk_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                            k_next_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(k_prev_0, _t7)
                            _t8: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.tensor.matmul(normed_bf16_0, wv_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                            v_next_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(v_prev_0, _t8)
                            k_proj_acc_6: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_acc_iter_4_inner, k_next_0, [0, kv0_0])
                            v_proj_acc_6: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(v_proj_acc_iter_4_inner, v_next_0, [0, kv0_0])
                            k_proj_acc_7, v_proj_acc_7 = pl.yield_(k_proj_acc_6, v_proj_acc_6)
                        else:
                            k_proj_acc_7, v_proj_acc_7 = pl.yield_(k_proj_acc_iter_4_inner, v_proj_acc_iter_4_inner)
                        k_proj_acc_iter_4_inner_rv, q_proj_acc_iter_4_inner_rv, v_proj_acc_iter_4_inner_rv = pl.yield_(k_proj_acc_7, q_proj_acc_7, v_proj_acc_7)
                    k_proj_acc_iter_4_outer_rv, q_proj_acc_iter_4_outer_rv, v_proj_acc_iter_4_outer_rv = pl.yield_(k_proj_acc_iter_4_inner_rv, q_proj_acc_iter_4_inner_rv, v_proj_acc_iter_4_inner_rv)
                k0_2, k_proj_acc_3, q_proj_acc_3, v_proj_acc_3, x_chunk_2 = pl.yield_(k0_3, k_proj_acc_iter_4_outer_rv, q_proj_acc_iter_4_outer_rv, v_proj_acc_iter_4_outer_rv, x_chunk_3)
            q_proj_1: pl.Tensor[[16, 5120], pl.BFLOAT16] = pl.tensor.cast(q_proj_acc_3, target_type=pl.BFLOAT16, mode=2)
            k_proj_1: pl.Tensor[[16, 1024], pl.BFLOAT16] = pl.tensor.cast(k_proj_acc_3, target_type=pl.BFLOAT16, mode=2)
            v_proj_1: pl.Tensor[[16, 1024], pl.BFLOAT16] = pl.tensor.cast(v_proj_acc_3, target_type=pl.BFLOAT16, mode=2)
        for b_0, (attn_out_iter_1, k_cache_iter_1, v_cache_iter_1) in pl.parallel(0, 16, 1, init_values=(attn_out_0, k_cache_0, v_cache_0), chunk=4):
            pos_0: pl.Scalar[pl.INT32] = pl.tensor.read(cache_pos_0, [b_0])
            ctx_len_0: pl.Scalar[pl.INDEX] = pl.cast(pos_0, pl.INDEX) + 1
            ctx_blocks_0: pl.Scalar[pl.INDEX] = (ctx_len_0 + 64 - 1) // 64
            cos_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(rope_cos_0, [1, 128], [pos_0, 0])
            sin_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(rope_sin_0, [1, 128], [pos_0, 0])
            cos_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(cos_row_0, [1, 128 // 2], [0, 0])
            cos_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(cos_row_0, [1, 128 // 2], [0, 128 // 2])
            sin_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(sin_row_0, [1, 128 // 2], [0, 0])
            sin_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(sin_row_0, [1, 128 // 2], [0, 128 // 2])
            with pl.auto_incore():
                for kvh_0_out, (k_cache_iter_3_outer, v_cache_iter_3_outer) in pl.range(0, 2, 1, init_values=(k_cache_iter_1, v_cache_iter_1)):
                    for kvh_0_in, (k_cache_iter_3_inner, v_cache_iter_3_inner) in pl.parallel(0, 4, 1, init_values=(k_cache_iter_3_outer, v_cache_iter_3_outer)):
                        kv_col_0: pl.Scalar[pl.INDEX] = (0 + (kvh_0_out * 4 + kvh_0_in) * 1) * 128
                        _t9: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.view(k_proj_1, [1, 128], [b_0, kv_col_0])
                        k_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(_t9, target_type=pl.FP32, mode=2)
                        k_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(k_row_0, [1, 128 // 2], [0, 0])
                        k_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(k_row_0, [1, 128 // 2], [0, 128 // 2])
                        k_rot_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
                        _t10: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(k_lo_0, cos_lo_0)
                        _t11: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(k_hi_0, sin_lo_0)
                        _t12: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.sub(_t10, _t11)
                        k_rot_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot_0, _t12, [0, 0])
                        _t13: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(k_hi_0, cos_hi_0)
                        _t14: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(k_lo_0, sin_hi_0)
                        _t15: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.add(_t13, _t14)
                        k_rot_2: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot_1, _t15, [0, 128 // 2])
                        cache_row_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + (0 + (kvh_0_out * 4 + kvh_0_in) * 1) * 4096 + pl.cast(pos_0, pl.INDEX)
                        _t16: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.cast(k_rot_2, target_type=pl.BFLOAT16, mode=2)
                        k_cache_5: pl.Tensor[[524288, 128], pl.BFLOAT16] = pl.tensor.assemble(k_cache_iter_3_inner, _t16, [cache_row_0, 0])
                        _t17: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.view(v_proj_1, [1, 128], [b_0, kv_col_0])
                        v_cache_5: pl.Tensor[[524288, 128], pl.BFLOAT16] = pl.tensor.assemble(v_cache_iter_3_inner, _t17, [cache_row_0, 0])
                        k_cache_iter_3_inner_rv, v_cache_iter_3_inner_rv = pl.yield_(k_cache_5, v_cache_5)
                    k_cache_iter_3_outer_rv, v_cache_iter_3_outer_rv = pl.yield_(k_cache_iter_3_inner_rv, v_cache_iter_3_inner_rv)
            with pl.auto_incore():
                attn_row_0: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.create([1, 5120], dtype=pl.FP32)
                attn_row_1: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.mul(attn_row_0, 0.0)
                for h_0_out, (attn_row_iter_2_outer, kvh_iter_1_outer) in pl.range(0, 8, 1, init_values=(attn_row_1, kvh_0)):
                    for h_0_in, (attn_row_iter_2_inner, kvh_iter_1_inner) in pl.parallel(0, 8, 1, init_values=(attn_row_iter_2_outer, kvh_iter_1_outer)):
                        kvh_3: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) // 8
                        q_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 128
                        _t18: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.view(q_proj_1, [1, 128], [b_0, q_col_0])
                        q_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(_t18, target_type=pl.FP32, mode=2)
                        q_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(q_row_0, [1, 128 // 2], [0, 0])
                        q_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(q_row_0, [1, 128 // 2], [0, 128 // 2])
                        q_rot_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
                        _t19: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_lo_0, cos_lo_0)
                        _t20: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_hi_0, sin_lo_0)
                        _t21: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.sub(_t19, _t20)
                        q_rot_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot_0, _t21, [0, 0])
                        _t22: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_hi_0, cos_hi_0)
                        _t23: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_lo_0, sin_hi_0)
                        _t24: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.add(_t22, _t23)
                        q_rot_2: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot_1, _t24, [0, 128 // 2])
                        q_rot_bf16_0: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.cast(q_rot_2, target_type=pl.BFLOAT16, mode=2)
                        oi_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
                        li_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                        mi_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                        oi_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.mul(oi_0, 0.0)
                        li_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(li_0, 0.0)
                        mi_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(mi_0, 0.0)
                        for sb_0, (li_iter_2, mi_iter_2, oi_iter_2) in pl.range(0, ctx_blocks_0, 1, init_values=(li_1, mi_1, oi_1)):
                            s0_0: pl.Scalar[pl.INDEX] = sb_0 * 64
                            valid_len_0: pl.Scalar[pl.INDEX] = min(64, ctx_len_0 - s0_0)
                            cache_row0_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_3 * 4096 + s0_0
                            k_tile_0: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.tensor.view(k_cache_iter_3_outer_rv, [64, 128], [cache_row0_0, 0])
                            v_tile_0: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.tensor.view(v_cache_iter_3_outer_rv, [64, 128], [cache_row0_0, 0])
                            _t25: pl.Tensor[[1, 64], pl.BFLOAT16] = pl.tensor.matmul(q_rot_bf16_0, k_tile_0, a_trans=False, b_trans=True, c_matrix_nz=False)
                            scores_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.mul(_t25, 0.0883883)
                            scores_valid_0: pl.Tensor[[1, valid_len], pl.FP32] = pl.tensor.view(scores_0, [1, valid_len_0], [0, 0])
                            _t26: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_max(scores_valid_0)
                            cur_mi_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(_t26, target_type=pl.FP32, mode=2)
                            _t27: pl.Tensor[[1, valid_len], pl.FP32] = pl.tensor.row_expand_sub(scores_valid_0, cur_mi_0)
                            exp_scores_0: pl.Tensor[[1, valid_len], pl.FP32] = pl.tensor.exp(_t27)
                            _t28: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_sum(exp_scores_0)
                            cur_li_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(_t28, target_type=pl.FP32, mode=2)
                            exp_pad_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.create([1, 64], dtype=pl.FP32)
                            exp_pad_1: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.mul(exp_pad_0, 0.0)
                            exp_pad_2: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(exp_pad_1, exp_scores_0, [0, 0])
                            _t29: pl.Tensor[[1, 64], pl.BFLOAT16] = pl.tensor.cast(exp_pad_2, target_type=pl.BFLOAT16, mode=2)
                            oi_tmp_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.matmul(_t29, v_tile_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            if sb_0 == 0:
                                oi_4: pl.Tensor[[1, 128], pl.FP32] = oi_tmp_0
                                li_4: pl.Tensor[[1, 1], pl.FP32] = cur_li_0
                                mi_4: pl.Tensor[[1, 1], pl.FP32] = cur_mi_0
                                li_6, mi_6, oi_6 = pl.yield_(li_4, mi_4, oi_4)
                            else:
                                mi_new_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.maximum(mi_iter_2, cur_mi_0)
                                _t30: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.sub(mi_iter_2, mi_new_0)
                                alpha_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(_t30)
                                _t31: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.sub(cur_mi_0, mi_new_0)
                                beta_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(_t31)
                                _t32: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(alpha_0, li_iter_2)
                                _t33: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(beta_0, cur_li_0)
                                li_5: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(_t32, _t33)
                                _t34: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_mul(oi_iter_2, alpha_0)
                                _t35: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_mul(oi_tmp_0, beta_0)
                                oi_5: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.add(_t34, _t35)
                                mi_5: pl.Tensor[[1, 1], pl.FP32] = mi_new_0
                                li_6, mi_6, oi_6 = pl.yield_(li_5, mi_5, oi_5)
                            li_3, mi_3, oi_3 = pl.yield_(li_6, mi_6, oi_6)
                        ctx_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_div(oi_3, li_3)
                        attn_row_4: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.assemble(attn_row_iter_2_inner, ctx_0, [0, q_col_0])
                        attn_row_iter_2_inner_rv, kvh_iter_1_inner_rv = pl.yield_(attn_row_4, kvh_3)
                    attn_row_iter_2_outer_rv, kvh_iter_1_outer_rv = pl.yield_(attn_row_iter_2_inner_rv, kvh_iter_1_inner_rv)
                attn_out_3: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(attn_out_iter_1, attn_row_iter_2_outer_rv, [b_0, 0])
            attn_out_2, k_cache_2, v_cache_2 = pl.yield_(attn_out_3, k_cache_iter_3_outer_rv, v_cache_iter_3_outer_rv)
        with pl.auto_incore():
            for ob_1_out, (k0_iter_4_outer, kb_iter_2_outer, resid1_iter_1_outer) in pl.range(0, 10, 1, init_values=(k0_2, kb_1, resid1_0)):
                for ob_1_in, (k0_iter_4_inner, kb_iter_2_inner, resid1_iter_1_inner) in pl.parallel(0, 8, 1, init_values=(k0_iter_4_outer, kb_iter_2_outer, resid1_iter_1_outer)):
                    o0_0: pl.Scalar[pl.INDEX] = (0 + (ob_1_out * 8 + ob_1_in) * 1) * 64
                    o_acc_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32)
                    o_acc_1: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(o_acc_0, 0.0)
                    for kb_4, (k0_iter_6, o_acc_iter_2) in pl.range(0, 40, 1, init_values=(k0_iter_4_inner, o_acc_1)):
                        k0_8: pl.Scalar[pl.INDEX] = kb_4 * 128
                        _t36: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.view(attn_out_2, [16, 128], [0, k0_8])
                        a_chunk_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.cast(_t36, target_type=pl.BFLOAT16, mode=2)
                        w_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.tensor.view(wo_0, [128, 64], [k0_8, o0_0])
                        _t37: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.tensor.matmul(a_chunk_0, w_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                        o_acc_4: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(o_acc_iter_2, _t37)
                        k0_7, o_acc_3 = pl.yield_(k0_8, o_acc_4)
                    _t38: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [16, 64], [0, o0_0])
                    resid_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.cast(_t38, target_type=pl.FP32, mode=2)
                    _t39: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(o_acc_3, resid_0)
                    resid1_3: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(resid1_iter_1_inner, _t39, [0, o0_0])
                    k0_iter_4_inner_rv, kb_iter_2_inner_rv, resid1_iter_1_inner_rv = pl.yield_(k0_7, kb_4, resid1_3)
                k0_iter_4_outer_rv, kb_iter_2_outer_rv, resid1_iter_1_outer_rv = pl.yield_(k0_iter_4_inner_rv, kb_iter_2_inner_rv, resid1_iter_1_inner_rv)
            sq_sum_5: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
            sq_sum_6: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.mul(sq_sum_5, 0.0)
            for kb_5_out, (k0_iter_9_outer, sq_sum_iter_7_outer, x_chunk_iter_4_outer) in pl.range(0, 10, 1, init_values=(k0_5, sq_sum_6, x_chunk_2)):
                for kb_5_in, (k0_iter_9_inner, sq_sum_iter_7_inner, x_chunk_iter_4_inner) in pl.parallel(0, 4, 1, init_values=(k0_iter_9_outer, sq_sum_iter_7_outer, x_chunk_iter_4_outer)):
                    k0_11: pl.Scalar[pl.INDEX] = (0 + (kb_5_out * 4 + kb_5_in) * 1) * 128
                    x_chunk_6: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.view(resid1_iter_1_outer_rv, [16, 128], [0, k0_11])
                    _t40: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.mul(x_chunk_6, x_chunk_6)
                    _t41: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(_t40)
                    sq_sum_9: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(sq_sum_iter_7_inner, _t41)
                    k0_iter_9_inner_rv, sq_sum_iter_7_inner_rv, x_chunk_iter_4_inner_rv = pl.yield_(k0_11, sq_sum_9, x_chunk_6)
                k0_iter_9_outer_rv, sq_sum_iter_7_outer_rv, x_chunk_iter_4_outer_rv = pl.yield_(k0_iter_9_inner_rv, sq_sum_iter_7_inner_rv, x_chunk_iter_4_inner_rv)
            _t42: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.mul(sq_sum_iter_7_outer_rv, 0.000195313)
            _t43: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(_t42, 1e-06)
            inv_rms_1: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.rsqrt(_t43)
            down_proj_acc_0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32)
            down_proj_acc_1: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.mul(down_proj_acc_0, 0.0)
            for kb_6, (gamma_iter_1, k0_iter_12, normed_iter_1, normed_bf16_iter_1, post_norm_iter_1, x_chunk_iter_7) in pl.range(0, 40, 1, init_values=(gamma_0, k0_iter_9_outer_rv, normed_0, normed_bf16_0, post_norm_0, x_chunk_iter_4_outer_rv)):
                k0_14: pl.Scalar[pl.INDEX] = kb_6 * 128
                x_chunk_9: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.view(resid1_iter_1_outer_rv, [16, 128], [0, k0_14])
                gamma_3: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(post_rms_weight_0, [1, 128], [0, k0_14])
                _t44: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_9, inv_rms_1)
                normed_3: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.col_expand_mul(_t44, gamma_3)
                normed_bf16_3: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.cast(normed_3, target_type=pl.BFLOAT16, mode=2)
                post_norm_3: pl.Tensor[[16, 5120], pl.BFLOAT16] = pl.tensor.assemble(post_norm_iter_1, normed_bf16_3, [0, k0_14])
                gamma_2, k0_13, normed_2, normed_bf16_2, post_norm_2, x_chunk_8 = pl.yield_(gamma_3, k0_14, normed_3, normed_bf16_3, post_norm_3, x_chunk_9)
            for ob_2, (down_proj_acc_iter_2, k0_iter_15, kb_iter_7, o0_iter_1) in pl.range(0, 400, 1, init_values=(down_proj_acc_1, k0_13, kb_6, o0_0)):
                o0_3: pl.Scalar[pl.INDEX] = ob_2 * 64
                gate_acc_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32)
                up_acc_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32)
                gate_acc_1: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(gate_acc_0, 0.0)
                up_acc_1: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(up_acc_0, 0.0)
                for kb_9, (gate_acc_iter_2, k0_iter_17, up_acc_iter_2) in pl.range(0, 40, 1, init_values=(gate_acc_1, k0_iter_15, up_acc_1)):
                    k0_19: pl.Scalar[pl.INDEX] = kb_9 * 128
                    post_chunk_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.view(post_norm_2, [16, 128], [0, k0_19])
                    wg_0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.tensor.view(w_gate_0, [128, 64], [k0_19, o0_3])
                    wu_0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.tensor.view(w_up_0, [128, 64], [k0_19, o0_3])
                    _t45: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.tensor.matmul(post_chunk_0, wg_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                    gate_acc_4: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(gate_acc_iter_2, _t45)
                    _t46: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.tensor.matmul(post_chunk_0, wu_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                    up_acc_4: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(up_acc_iter_2, _t46)
                    gate_acc_3, k0_18, up_acc_3 = pl.yield_(gate_acc_4, k0_19, up_acc_4)
                _t47: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.neg(gate_acc_3)
                _t48: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.exp(_t47)
                _t49: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(_t48, 1.0)
                sigmoid_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.recip(_t49)
                _t50: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(gate_acc_3, sigmoid_0)
                mlp_chunk_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(_t50, up_acc_3)
                mlp_chunk_bf16_0: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.tensor.cast(mlp_chunk_0, target_type=pl.BFLOAT16, mode=2)
                for dob_0_out, (down_proj_acc_iter_4_outer,) in pl.range(0, 10, 1, init_values=(down_proj_acc_iter_2,)):
                    for dob_0_in, (down_proj_acc_iter_4_inner,) in pl.parallel(0, 8, 1, init_values=(down_proj_acc_iter_4_outer,)):
                        d0_0: pl.Scalar[pl.INDEX] = (0 + (dob_0_out * 8 + dob_0_in) * 1) * 64
                        down_prev_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.view(down_proj_acc_iter_4_inner, [16, 64], [0, d0_0])
                        w_down_chunk_0: pl.Tensor[[64, 64], pl.BFLOAT16] = pl.tensor.view(w_down_0, [64, 64], [o0_3, d0_0])
                        _t51: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.tensor.matmul(mlp_chunk_bf16_0, w_down_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                        down_next_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(down_prev_0, _t51)
                        down_proj_acc_6: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(down_proj_acc_iter_4_inner, down_next_0, [0, d0_0])
                        down_proj_acc_iter_4_inner_rv: pl.Tensor[[16, 5120], pl.FP32] = pl.yield_(down_proj_acc_6)
                    down_proj_acc_iter_4_outer_rv: pl.Tensor[[16, 5120], pl.FP32] = pl.yield_(down_proj_acc_iter_4_inner_rv)
                down_proj_acc_3, k0_16, kb_8, o0_2 = pl.yield_(down_proj_acc_iter_4_outer_rv, k0_18, kb_9, o0_3)
            for ob_3_out, (o0_iter_4_outer, out_iter_1_outer) in pl.range(0, 10, 1, init_values=(o0_2, out_0)):
                for ob_3_in, (o0_iter_4_inner, out_iter_1_inner) in pl.parallel(0, 8, 1, init_values=(o0_iter_4_outer, out_iter_1_outer)):
                    o0_6: pl.Scalar[pl.INDEX] = (0 + (ob_3_out * 8 + ob_3_in) * 1) * 64
                    _t52: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.view(down_proj_acc_3, [16, 64], [0, o0_6])
                    _t53: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.view(resid1_iter_1_outer_rv, [16, 64], [0, o0_6])
                    down_acc_0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(_t52, _t53)
                    _t54: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.tensor.cast(down_acc_0, target_type=pl.BFLOAT16, mode=2)
                    out_3: pl.Tensor[[16, 5120], pl.BFLOAT16] = pl.tensor.assemble(out_iter_1_inner, _t54, [0, o0_6])
                    o0_iter_4_inner_rv, out_iter_1_inner_rv = pl.yield_(o0_6, out_3)
                o0_iter_4_outer_rv, out_iter_1_outer_rv = pl.yield_(o0_iter_4_inner_rv, out_iter_1_inner_rv)
        return out_iter_1_outer_rv