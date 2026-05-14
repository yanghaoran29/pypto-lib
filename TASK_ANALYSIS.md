# Qwen3-14B Decode Task Analysis Report

**File**: `models/qwen3/14b/qwen3_14b_decode.py`  
**Date**: 2026-05-13  
**Model**: Qwen3-14B (single-layer decode forward pass)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total unique task types** | 17 |
| **Total task invocations (single pass)** | 24,274 |
| **Scopes** | 3 (Input Proj, Attention, Output+MLP) |
| **Parallelization levels** | 2-3 (batch, blocks, context) |

---

## Task Distribution by Scope

### Scope 1: Input Projection & Normalization
**Lines: 162-285**  
**Purpose**: RMSNorm + Q/K/V projections

| Task | Count | Loop Structure | Description |
|------|-------|-----------------|-------------|
| `rmsnorm` | 3 | `for b0 in pl.parallel(0, batch_padded, BATCH_TILE)` | Input normalization (3 iterations for batch_padded=48, BATCH_TILE=16) |
| `q_proj` | 20 | `for q0 in pl.parallel(0, HIDDEN, Q_OUT_CHUNK)` | Query projection (HIDDEN=5120 / Q_OUT_CHUNK=256 = 20) |
| `k_proj` | 4 | `for kv0 in pl.parallel(0, KV_HIDDEN, KV_OUT_CHUNK)` | Key projection (KV_HIDDEN=1024 / KV_OUT_CHUNK=256 = 4) |
| `v_proj` | 4 | **Same loop as k_proj** | Value projection (same loop iteration count) |
| `qk_norm` | 24 | `for b0 ... for h in pl.range(num_kv_heads)` | Per-head normalization (3 × 8 = 24) |
| **Subtotal** | **55** | Parallel + Sequential | |

**Dependency Flow**:
```
rmsnorm → q_proj + k_proj + v_proj → qk_norm
```

**Characteristics**:
- Batch-level parallelism: 3 tiles (batch_padded/BATCH_TILE)
- Output chunk parallelism: 20 (Q) and 4 (K/V)
- Per-head sequential processing: 8 KV heads

---

### Scope 2: Attention (RoPE, QK, Softmax, SV, Online-Softmax)
**Lines: 287-520**  
**Purpose**: Grouped-query attention computation

| Task | Count | Loop Structure | Description |
|------|-------|-----------------|-------------|
| `rope_kv_cache` | 384 | `for b in pl.parallel(user_batch); for ki in pl.range(num_kv_heads)` | RoPE + cache update (48 × 8 = 384) |
| `qk_matmul` | 960 | `for b in pl.parallel(user_batch); for gi in pl.parallel(total_q_groups, 2)` | QK matmul, 2 query groups/iter (48 × 20 = 960) |
| `softmax` | 7,680 | `... for sb in pl.range(ctx_blocks)` | Softmax per context block (960 × 8 ctx_blocks ≈ 7,680) |
| `sv_matmul` | 7,680 | `... for sb in pl.range(ctx_blocks)` | SV matmul per context block (960 × 8 = 7,680) |
| `online_softmax` | 6,720 | `... for sb in pl.range(1, ctx_blocks)` | Online softmax accumulation (960 × 7 = 6,720) |
| **Subtotal** | **23,424** | Highly nested parallel loops | |

**Dependency Flow**:
```
rope_kv_cache → qk_matmul → softmax → sv_matmul → online_softmax
```

**Characteristics**:
- **User batch parallelism**: 48 (per-request processing)
- **Query group parallelism**: 20 (total_q_groups/2 chunks, processes 2 groups per iteration)
- **Context block loop**: ~8 blocks (ctx_len=4096 / BLOCK_SIZE=256)
- **Sequential constraint**: Each task must complete before next in scope
- **NOTE**: Most task invocations concentrated here (96% of total)

**Context Block Calculation**:
```python
ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
          ≈ (4096 + 256 - 1) // 256 = 16  # or fewer for shorter sequences
```
*Above calculation assumes MAX_SEQ=4096, actual value is runtime-dependent*

---

### Scope 3: Output Projection + MLP + Residual
**Lines: 521-640**  
**Purpose**: Output attention + feed-forward network

| Task | Count | Loop Structure | Description |
|------|-------|-----------------|-------------|
| `out_proj_residual` | 60 | `for b0 in pl.parallel(...); for ob in pl.parallel(0, out_proj_n_blocks)` | Attention output proj (3 × 20 = 60) |
| `post_rmsnorm` | 3 | `for b0 in pl.parallel(0, batch_padded, BATCH_TILE)` | Post-attention normalization (3 iterations) |
| `gate_proj` | 204 | `for b0 in pl.parallel(...); for ob in pl.range(mlp_out_blocks)` | MLP gate (3 × 68 = 204) |
| `up_proj` | 204 | **Same loop as gate_proj** | MLP up projection (same loop) |
| `silu` | 204 | **Same loop as gate_proj** | SiLU activation (same loop) |
| `down_proj` | 60 | `for b0 in pl.parallel(...); for dob in pl.range(down_out_blocks)` | MLP down (3 × 20 = 60) |
| `down_proj_residual` | 60 | **Same loop as down_proj** | MLP output residual (same loop) |
| **Subtotal** | **795** | Parallel blocks | |

**Dependency Flow**:
```
out_proj_residual → post_rmsnorm → (gate_proj + up_proj + silu) → down_proj → down_proj_residual
                                      [same loop]                    [same loop]
```

**Characteristics**:
- Batch-level parallelism: 3 tiles
- Output block parallelism: 20 (attention proj), 68 (MLP), 20 (down proj)
- Task grouping: gate/up/silu share one loop, down/residual share another
- Sequential across task groups (within same batch tile)

**Block Calculations**:
```python
out_proj_n_blocks = HIDDEN // OUT_PROJ_N_CHUNK = 5120 // 256 = 20
mlp_out_blocks = INTERMEDIATE // MLP_OUT_CHUNK = 17408 // 256 ≈ 68
down_out_blocks = HIDDEN // DOWN_OUT_CHUNK = 5120 // 256 = 20
```

---

## Cross-Scope Data Flow

```
┌─────────────────┐
│  Input Tensors  │
├─────────────────┤
│ hidden_states   │
│ q_weight        │
│ k_weight        │
│ v_weight        │
└────────┬────────┘
         │
    ┌────▼──────────────────┐
    │   SCOPE 1             │
    │ Input Projection      │
    ├───────────────────────┤
    │ rmsnorm               │
    │ q_proj/k_proj/v_proj  │
    │ qk_norm (per-head)    │
    └────┬──────────────────┘
         │ q_proj_norm, k_proj_norm, v_proj
         │
    ┌────▼──────────────────────────┐
    │   SCOPE 2                      │
    │ Attention Computation          │
    ├────────────────────────────────┤
    │ rope_kv_cache (KV cache write) │
    │ qk_matmul (score)              │
    │ softmax (normalize)            │
    │ sv_matmul (context blend)      │
    │ online_softmax (accumulate)    │
    └────┬───────────────────────────┘
         │ attn_out (attention output)
         │
    ┌────▼──────────────────────────┐
    │   SCOPE 3                      │
    │ Output Projection + MLP        │
    ├────────────────────────────────┤
    │ out_proj_residual              │
    │ post_rmsnorm                   │
    │ gate_proj + up_proj + silu     │
    │ down_proj + down_proj_residual │
    └────┬───────────────────────────┘
         │
    ┌────▼────────────────┐
    │  Output Tensors     │
    ├─────────────────────┤
    │ out[batch, hidden]  │
    └─────────────────────┘
```

**Critical Barrier**: Scope 2 must complete **all user_batch iterations** before Scope 3 begins
- This is due to the sequential loop nesting: `for b in pl.parallel(user_batch):` must fully complete
- No per-batch pipelining between Scope 2 and 3 in current code structure

---

## Task Characteristics Summary

### Parallelism Opportunities

| Level | Scope 1 | Scope 2 | Scope 3 |
|-------|---------|---------|---------|
| **Batch** | 3 tiles (parallel) | 48 requests (parallel) | 3 tiles (parallel) |
| **Spatial/Block** | Output chunks (Q: 20, K/V: 4) | Query groups (20 parallel pairs) | Output blocks (20-68 parallel) |
| **Sequential** | Per-head (8 heads) | Context blocks (~8 blocks) | Output blocks (within tile) |

### Task Criticality

**Critical Path** (determines latency):
1. `softmax` & `sv_matmul` - **Bottleneck**: 7,680 invocations each, sequential context loop
2. `rope_kv_cache` - 384 sequential iterations over user_batch
3. `qk_matmul` - 960 iterations feeding into softmax

**High-frequency Tasks**:
1. `softmax` - 7,680 times (31.6% of all invocations)
2. `sv_matmul` - 7,680 times (31.6% of all invocations)
3. `qk_matmul` - 960 times (3.9% of all invocations)

**Low-frequency Tasks**:
- `post_rmsnorm` - only 3 times
- `v_proj` & `k_proj` - 4 times each

### Optimization Opportunities

1. **Loop Fusion**: Combine qk_matmul + softmax + sv_matmul to reduce intermediate tensor writes
2. **Pipeline Between Scopes**: Batch-level pipelining (Scope 2 batch i → Scope 3 batch i) instead of barrier
3. **Context Block Optimization**: Reduce context_blocks through block size tuning (currently BLOCK_SIZE=256)
4. **Task Grouping**: gate_proj/up_proj/silu already grouped; similar for down_proj/residual

---

## Execution Time Estimation

For reference (rough estimation):

| Scope | Component | Approx Operations | Notes |
|-------|-----------|-------------------|-------|
| **Scope 1** | RMSNorm | ~1.7M | Input: 48×5120, H=5120 |
| | Q/K/V Projections | | Matmuls: 48×5120×5120 (Q), 48×1024×1024 (K/V) |
| **Scope 2** | All attention | ~402.8M | Per-sample: 8.4M ops × 48 samples |
| | (includes softmax, sv_matmul, etc.) | | Includes context block loops |
| **Scope 3** | Output proj + MLP | ~9.3M | 48×5120 (out_proj) + 48×17408 (MLP) |

Total: ~414M operations for single forward pass

---

## Key Constants

```python
BATCH = 48                  # User batch size
BATCH_TILE = 16            # Tile size for memory efficiency
NUM_HEADS = 40             # Total heads
NUM_KV_HEADS = 8           # KV cache heads (grouped query)
Q_GROUPS = 5               # Queries per KV head
HIDDEN = 5120              # Embedding dimension (40 × 128)
INTERMEDIATE = 17408       # MLP hidden size
HEAD_DIM = 128             # Per-head dimension
BLOCK_SIZE = 256           # Sequence cache block size
MAX_SEQ = 4096             # Max sequence length
```

---

## Recommendations for Analysis

1. **Profiling focus**: Scope 2 dominates execution (96% of task invocations)
   - Target `softmax` and `sv_matmul` for optimization
   
2. **Concurrency analysis**: 
   - Scope 1: ~3 batch tiles can be processed in parallel
   - Scope 2: ~48 requests + 20 query group pairs (960 total parallel opportunities)
   - Scope 3: ~3 batch tiles + 20-68 output blocks
   
3. **Memory access patterns**:
   - `rope_kv_cache`: Sequential writes to k_cache/v_cache (potential bottleneck)
   - `softmax`/`sv_matmul`: Read from cache repeatedly (context block loop)
   
4. **Vectorization**:
   - `online_softmax`: Row-wise operations (likely vectorizable)
   - `silu`: Element-wise activation (highly vectorizable)

---

End of Report
