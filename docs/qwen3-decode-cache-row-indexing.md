# Qwen3-14B Decode：KV 行下标、`slot`/`pbid` 与 `pl.at` 任务划分

本文档依据 `models/qwen3/14b/qwen3_14b_decode.py` 当前实现，统一说明：

- paged layout 下 **`cache_row`（写）** 与 **`cache_row0`（读 tile 起点）** 的公式与关系；
- golden 初始化下 **`slot` / `slot_block`** 与 **`pbid`** 的区分（**量纲不同，不可混用区间**）；
- **`rope_kv_cache`**、**`qk_matmul`** 与 **`sv_matmul`** 中 **`pl.at` / `pl.parallel`** 的嵌套与实例个数，以及 **`sv_matmul`** 对 **`v_cache`** / **`all_exp_padded`** 的读下标。

默认常量来自该文件：`BLOCK_SIZE = SEQ_TILE`（常为 256）、`MAX_SEQ`（常为 4096）、`num_kv_heads`、`head_dim`、`q_groups`、`total_q_groups` 等以源码为准。

---

## 1. 符号与单行下标公式

| 记号 | 代码含义 |
|------|----------|
| \(B\) | `BLOCK_SIZE` |
| \(H\) | `num_kv_heads` |
| \(M\) | `max_blocks_per_seq`，即 `(max_seq + B - 1) // B` |
| \(b\) | batch 索引 |
| \(\mathrm{ctx\_len}\) | `seq_lens[b]` |
| \(\mathrm{pos}\) | 本步最后一个 token 下标：`pos = ctx_len - 1` |
| \(\mathrm{ctx\_blocks}\) | `(ctx_len + B - 1) // B` |

**`k_cache` / `v_cache` 的扁平行下标**（物理块 id 为 \(P\)，KV 头 \(k\in\{0,\ldots,H-1\}\)，块内偏移 \(o\in\{0,\ldots,B-1\}\)）：

\[
\mathrm{row}(P,k,o) = (P\cdot H + k)\cdot B + o.
\]

---

## 2. Golden：`slot_mapping` 与 `block_table`

### 2.1 `init_slot_mapping`

\[
\mathrm{phys\_block} = bM + \left\lfloor\frac{\mathrm{pos}}{B}\right\rfloor,
\qquad
\mathrm{slot} = \mathrm{phys\_block}\cdot B + (\mathrm{pos}\bmod B).
\]

故

\[
\boxed{\mathrm{slot} = bMB + \mathrm{pos}},\quad \mathrm{pos}=\mathrm{ctx\_len}-1.
\]

初始化函数里局部变量名为 `slots`，写入 `TensorSpec("slot_mapping", ...)`；运行时 **`slot_mapping[b]`** 即当初 **`slots[b]`**。

### 2.2 `slot_block` 与 `slot_offset`

\[
\mathrm{slot\_block}=\left\lfloor\frac{\mathrm{slot}}{B}\right\rfloor,
\qquad
\mathrm{slot\_offset}=\mathrm{slot}\bmod B.
\]

因 \(bMB\equiv 0\pmod B\)，有

\[
\boxed{\mathrm{slot\_block} = bM + \left\lfloor\frac{\mathrm{pos}}{B}\right\rfloor},
\qquad
\boxed{\mathrm{slot\_offset} = \mathrm{pos}\bmod B}.
\]


### 2.3 `init_block_table`

`torch.arange(batch * M)` 时，对 \(\mathrm{sb}\in\{0,\ldots,\mathrm{ctx\_blocks}-1\}\)：

\[
\boxed{\mathrm{pbid}(\mathrm{sb}) = \texttt{block\_table}[bM+\mathrm{sb}] = bM + \mathrm{sb}}
\]

（仅在该初始化下成立；一般推理 **`pbid` 总由 `block_table` 表项决定**。）

### 2.4 前提 \(\mathrm{ctx\_len} \le \texttt{max\_seq}\)：`slot // BLOCK_SIZE`（即 \(\mathrm{slot\_block}\)）的块 id 区间

本仓库中 **`rope_cos` / `rope_sin`** 第一维为 **`max_seq`**，**`block_table`** 每条序列预留 **`M = max_blocks_per_seq`** 个块槽；与之一致时约定 **\(\mathrm{ctx\_len} \le \texttt{max\_seq}\)**（从而 \(\mathrm{pos}=\mathrm{ctx\_len}-1 \le \texttt{max\_seq}-1\)）。

此时令 \(L=\lfloor \mathrm{pos}/B\rfloor\)。由 \(M=\lceil \texttt{max\_seq}/B\rceil\) 可得 **\(0 \le L \le M-1\)**（最大 \(\mathrm{pos}\) 为 \(\texttt{max\_seq}-1\) 时，\(L=\lfloor(\texttt{max\_seq}-1)/B\rfloor \le M-1\)）。

代入 §2.2：

\[
\mathrm{slot\_block}=bM+L
\quad\Rightarrow\quad
\boxed{\mathrm{slot\_block}\in\{bM,\,bM+1,\,\ldots,\,(b+1)M-1\}}.
\]

按整数理解半开区间，即

\[
\boxed{\left\lfloor \dfrac{\mathrm{slot}}{B} \right\rfloor \in \bigl[b\cdot M,\ (b+1)\cdot M\bigr)}.
\]

---

## 3. 源码锚点：`block_table_base`

在 `for b in pl.parallel(user_batch):` 内应为：

```text
block_table_base = b * max_blocks_per_seq
block_table_idx  = block_table_base + sb   # sb ∈ [0, ctx_blocks)
pbid             = block_table[block_table_idx]
```

**`block_table_idx` 实际访问区间**：

\[
[bM,\ bM+\mathrm{ctx\_blocks}) \subseteq [bM,\ (b+1)M).
\]

本步只遍历 **`ctx_blocks`** 个表项，**不是**必然扫满长度为 \(M\) 的整段（除非 \(\mathrm{ctx\_blocks}=M\)）。

---

## 4. `rope_kv_cache`：写 `k_cache`、`v_cache`、`all_q_padded`

**嵌套**：`for b in pl.parallel(user_batch):` → `with pl.at(..., "rope_kv_cache")` → `for ki in pl.range(num_kv_heads):`。

**行号**：

\[
\boxed{\mathrm{cache\_row}(\mathrm{ki}) = (\mathrm{slot\_block}\cdot H + \mathrm{ki})\cdot B + \mathrm{slot\_offset}},
\quad \mathrm{ki}\in\{0,\ldots,H-1\}.
\]

**性质**：相邻 \(\mathrm{ki}\) 的 \(\mathrm{cache\_row}\) 相差 \(B\)；本步共写 **\(H\) 行** KV，**不是**填满某个连续区间内每一个行号。

**半开包络**（从下界起长度 \(HB\)、包含全部写地址的最短半开区间）：

\[
\bigl[\mathrm{slot\_block}\cdot HB + \mathrm{slot\_offset},\ (\mathrm{slot\_block}+1)\cdot HB + \mathrm{slot\_offset}\bigr).
\]

同一块 **`pl.at`** 内还为 RoPE 后的 Q 写 **`all_q_padded`**；后续 **`qk_matmul`** 除 **`k_cache`** 外也依赖 **`all_q_padded`**。

---

## 5. `qk_matmul`：读 `k_cache` tile

**嵌套**：同一 `for b` 内 → `for gi in pl.parallel(0, total_q_groups, 2):` → `with pl.at(..., "qk_matmul")` → `for sb in pl.range(ctx_blocks):`。

对每个 **`sb`**，在同一 **`pl.at`** 内计算 **两组** KV 头（由 **`gi0`/`gi1`** 推出 **`kvh0`/`kvh1`**）：

\[
\texttt{cache\_row0}(\texttt{kvh}) = (\mathrm{pbid}\cdot H + \texttt{kvh})\cdot B,
\qquad
\texttt{k\_tile}=\texttt{k\_cache}[\texttt{cache\_row0}:\texttt{cache\_row0}+B,:].
\]

即半开行区间

\[
\bigl[(\mathrm{pbid}\cdot H+\texttt{kvh})B,\ (\mathrm{pbid}\cdot H+\texttt{kvh}+1)B\bigr).
\]

---

## 6. `sv_matmul`：读 `v_cache` tile、用 `softmax` 的 `exp`

**嵌套**：与 **`qk_matmul`** 相同——同一 `for b` 内 → `for gi in pl.parallel(0, total_q_groups, 2):` → `with pl.at(..., "sv_matmul")` → `for sb in pl.range(ctx_blocks):`。

**`block_table` / `pbid`**：与 **`qk_matmul`** 完全一致——对每个 **`sb`**：

```text
block_table_idx = block_table_base + sb
pbid            = block_table[block_table_idx]
```

**`cache_row0` 与 `v_tile`**：对每个 **`kvh ∈ {kvh0, kvh1}`**（由 **`gi0`/`gi1`** 推出），行起点公式与 **`qk_matmul` 读 `k_cache` 时相同**：

\[
\texttt{cache\_row0}(\texttt{kvh}) = (\mathrm{pbid}\cdot H + \texttt{kvh})\cdot B,
\qquad
\texttt{v\_tile}=\texttt{v\_cache}[\texttt{cache\_row0}:\texttt{cache\_row0}+B,:].
\]

**`exp_tile`**：来自 **`softmax`** 阶段写入的 **`all_exp_padded0/1`**，按 **`sb`** 切片（与 **`qk_matmul`** 里 **`all_raw_scores`** 的 **`sb`** 行块对齐）：

\[
\texttt{exp\_tile}=\texttt{all\_exp\_padded}[\texttt{sb}\cdot Q_{\mathrm{pad}}:(\texttt{sb}+1)\cdot Q_{\mathrm{pad}},:],
\]

其中 \(Q_{\mathrm{pad}}=\texttt{Q\_HEAD\_PAD}\)（源码常量）。

**算子语义**：对每个 **`(sb, kvh)`** 做一次 **`pl.matmul(exp_tile, v_tile, out_dtype=FP32)`**（**非** `b_trans`），得到该块上的 **`oi_tmp`**，再 **`pl.assemble`** 到 **`all_oi_tmp0/1`** 的 **`[sb * Q_HEAD_PAD, 0]`** 起点。即 **softmax 权重（按块）× V tile → 部分 attention 输出**，供后续 **`online_softmax`** 跨 **`sb`** 累积。

**与 `qk_matmul` 的关系**：**同一 `pbid`、同一 `cache_row0` 公式**；区别仅为张量槽（**`k_cache` vs `v_cache`**）与左操作数（**`q @ K^T` vs `exp @ V`**）。**`rope_kv_cache`** 对 **`k_cache` / `v_cache`** 使用**同一** \(\mathrm{cache\_row}(\mathrm{ki})\) 写行；故 **`sv_matmul`** 与 **`qk_matmul`** 相对 **`rope`** 写地址的对应关系仍由 **§7**（及 **§2.3** golden 下 **`pbid` 与 `slot_block`** 的特例）刻画。

---

## 7. `cache_row` 与 `cache_row0` 的关系

当 **`pbid = slot_block`** 且 **`kvh = ki`** 时：

\[
\boxed{\mathrm{cache\_row}(\mathrm{ki}) = \texttt{cache\_row0}(\texttt{kvh}) + \mathrm{slot\_offset}}.
\]

在 **§2.3 的 `arange` golden** 下，取 **\(\mathrm{sb}=\lfloor \mathrm{pos}/B\rfloor\)** 则 **\(\mathrm{pbid}=bM+\mathrm{sb}=\mathrm{slot\_block}\)**。此时刚写入的 **K / V** 落在该 **`sb`** 对应 **`k_tile` / `v_tile`** 的第 **`slot_offset`** 行（0-based）。

---

## 8. `pbid` 与 `slot`：量纲与区间（必须区分）

| 量 | 含义 | golden 下典型取值范围（数量级） |
|----|------|----------------------------------|
| **`slot`** | 扁平槽位下标（已含 \(B\) 缩放进块内偏移） | 与 \(bMB+\mathrm{pos}\) 同阶；在 **§2.1 golden** 且 \(\mathrm{pos}\le\texttt{max\_seq}-1\) 时 \(\mathrm{slot}\in[bMB,\ bMB+\texttt{max\_seq}-1]\) |
| **`slot_block`** | \(\lfloor \mathrm{slot}/B\rfloor\)，物理块 id | **若 \(\mathrm{ctx\_len}\le\texttt{max\_seq}\)**：\(\mathrm{slot\_block}\in[bM,\,(b+1)M)\)（整数，见 **§2.4**） |
| **`pbid`** | **物理块 id（整数）** | **\(\mathrm{pbid}\in\{bM,\ldots,bM+\mathrm{ctx\_blocks}-1\}\)**，为 **\(O(M)\)** |

**相等条件**：一般 **`pbid = block_table[bM+sb]`**。仅当表为 **`arange`** 且 **`sb=\lfloor\mathrm{pos}/B\rfloor\)** 时 **`pbid = slot_block = \lfloor\mathrm{slot}/B\rfloor\)**。对其它 **`sb`**，**`pbid` 不必等于** **`slot_block`**。

---

## 9. 数据依赖（同 `b`）

对固定 **`b`**：

- **`qk_matmul`** 读 **`k_cache`**、**`all_q_padded`**，依赖**同一次 `b` 迭代中、排在前面**的 **`rope_kv_cache`**，以及历史步已写入的 cache。
- **`sv_matmul`** 读 **`v_cache`**（**`block_table` → `pbid` → `cache_row0`** 与 **`qk_matmul`** 同源）与 **`softmax`** 产出的 **`all_exp_padded0/1`**；**`v_cache`** 的当步写入同样来自 **`rope_kv_cache`**。

**每个 `b` 仅一次 `rope_kv_cache`**；**同一 `b` 下所有 `gi` 的 `qk_matmul` / `sv_matmul` 实例**共享该次 RoPE/Q 与当步 KV 写。程序顺序上为 **`rope_kv_cache` → `qk_matmul` → `softmax` → `sv_matmul` → `online_softmax`**。

---

## 10. `qk` / `sv` 读行并集（与 `rope` 写行集合的包含关系）

对固定 **`b`**，所有 **`sb ∈ \{0,\ldots,\mathrm{ctx\_blocks}-1\}`** 与所有 **`kvh`** 上，**`k_tile`（`qk_matmul`）** 与 **`v_tile`（`sv_matmul`）** 使用**同一** \(\texttt{cache\_row0}(\texttt{kvh})=(\mathrm{pbid}\cdot H+\texttt{kvh})\cdot B\) 作为半开区间 \([\texttt{cache\_row0},\,\texttt{cache\_row0}+B)\) 的下标。将 **`k_cache` / `v_cache`** 的扁平行号视为同一整数轴（§1 的 \(\mathrm{row}(P,k,o)\)），则两路 matmul 在本步内**读到的行号集合**相同，其**并集**与**单集合**一致，记为：

\[
\mathcal{R}_{\mathrm{qk}}=\mathcal{R}_{\mathrm{sv}}
=\mathcal{R}
:= \bigcup_{\texttt{sb},\texttt{kvh}}
\bigl[\texttt{cache\_row0}(\texttt{kvh}),\,\texttt{cache\_row0}(\texttt{kvh})+B\bigr)\cap\mathbb{Z}.
\]

在 **§8** 的 **`pbid` 取值范围**（\(\mathrm{pbid}\in\{bM,\ldots,bM+\mathrm{ctx\_blocks}-1\}\)）下，上述并集的最小半开包络为

\[
\boxed{
\mathcal{R}\subseteq
\bigl[bMHB,\ (bM+\mathrm{ctx\_blocks})HB\bigr)
}
\]

（当 **`block_table`** 在 **`sb`** 上取满该区间内的物理块、且 **`kvh`** 遍历全部 KV 头时，常取等号 \(\mathcal{R}=[bMHB,(bM+\mathrm{ctx\_blocks})HB)\)；一般 paged 表下 \(\mathcal{R}\) 仍是该半开区间的子集。）

**本步 `rope_kv_cache` 的写行集合**（仅当前 token 的 \(H\) 行 KV，**`ki = 0,\ldots,H-1`**）：

\[
\mathcal{W}_{\mathrm{rope}}
:=\bigl\{\,\mathrm{cache\_row}(\mathrm{ki}) \;\big|\; \mathrm{ki}\in\{0,\ldots,H-1\}\,\bigr\},
\qquad
\mathrm{cache\_row}(\mathrm{ki})=(\mathrm{slot\_block}\cdot H+\mathrm{ki})\cdot B+\mathrm{slot\_offset}.
\]

这是 **\(H\) 个整数点**（行号两两相差 \(B\)，互不相同），**不是**连续区间。

**与 \(\mathcal{R}\) 的包含关系**：记 \(L=\lfloor\mathrm{pos}/B\rfloor\)，\(O=\mathrm{pos}\bmod B\)。有 \(\mathrm{slot\_block}=bM+L\)（见 **§2.2**），且 \(0\le L\le \mathrm{ctx\_blocks}-1\)。对任意 \(\mathrm{ki}\)，

\[
\mathrm{cache\_row}(\mathrm{ki})
= (bM+L)\cdot HB + \mathrm{ki}\cdot B + O.
\]

又 \(\mathrm{pbid}=bM+L\) 恰为 **`sb = L`** 时 **`block_table[bM+L]`** 在 golden（**§2.3 `arange`**）下的取值，故该写行落在 **`sb=L`**、**`kvh=\mathrm{ki}`** 所读 **`k_tile` / `v_tile`** 的半开区间 \([\texttt{cache\_row0}(\mathrm{ki}),\,\texttt{cache\_row0}(\mathrm{ki})+B)\) 内（见 **§7**）。因此

\[
\boxed{\mathcal{W}_{\mathrm{rope}}\subseteq \mathcal{R}}.
\]

**下界对比**：

\[
R_{\mathrm{rope,min}}:=\min \mathcal{W}_{\mathrm{rope}}
=\mathrm{slot\_block}\cdot HB+\mathrm{slot\_offset}
=bMHB+L\cdot HB+O.
\]

读并包络下界为 **\(bMHB\)**，故

\[
R_{\mathrm{rope,min}} - bMHB = L\cdot HB + O \ge 0,
\]

等号仅当 \(\mathrm{pos}=0\)（\(L=O=0\)）。因此 **\(\mathcal{W}_{\mathrm{rope}}\)** 整体落在 **\(\mathcal{R}\)**（及其包络区间）之内，但 **\(R_{\mathrm{rope,min}}\)** 一般**严格大于**包络下界 **\(bMHB\)**；「写集合 ⊂ 读并集」与「最小写行 ≠ 读区间的左端点」同时成立。

**`k_cache` 与 `v_cache`**：\(\mathcal{W}_{\mathrm{rope}}\) 在 **`k_cache`** 与 **`v_cache`** 上各写 **同一组行号**；\(\mathcal{R}\) 在 **`qk_matmul` / `sv_matmul`** 中分别体现为对 **`k_cache`**、**`v_cache`** 的读，行号集合仍记为 \(\mathcal{R}\)。

---