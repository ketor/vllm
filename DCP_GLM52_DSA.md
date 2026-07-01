# GLM-5.2 / DeepSeek Sparse Attention (DSA) — Decode Context Parallel (DCP) 改造

> 分支 `feat/glm52-dsa-dcp`，基于 vLLM **v0.23.0**（tag `v0.23.0`, commit 0fc695f）。
> 目的：让 **DSA 稀疏 MLA 模型（GLM-5.2 / DeepSeek-V3.2 系）在 stock vLLM 0.23 上支持 DCP**，从而在单机 8×H100 上把 1M 上下文的 MLA KV 分片装下（MLA KV 在 TP 下复制，唯 DCP 能分片）。

## 背景
- stock 0.23 开 DCP 直接报错：`FlashMLASparseImpl` 的 `can_return_lse_for_decode=False` → `DCP requires attention impl to return softmax LSE`。
- 稀疏 decode kernel（`flash_mla_sparse_fwd`）**stock 就返回 LSE**，所以**无需重编 CUDA kernel**，纯 Python 改造即可。
- 参考实现：openclaw 内部镜像 `vllm-openai:v0.22.0-dcp-260625` 的 6 文件 Python 补丁（在 DSA 上真跑 `-dcp8`）。本分支把它 **port 到 stock 0.23**。

## 改动文件（4 个；mla_attention 无需改）
| 文件 | 改动 | 来源 |
|---|---|---|
| `vllm/v1/attention/backends/mla/flashmla_sparse.py` | **本仓 0.23-base 精确 port**（下方 4 关键点） | 本次原创 |
| `vllm/v1/attention/backends/mla/sparse_utils.py` | `triton_convert_req_index_to_global_index` 加 DCP 参数（top-k 全局位置→本 rank 物理 slot 重映射，owner-rank 过滤） | openclaw 0.22 补丁 port |
| `vllm/v1/attention/backends/mla/indexer.py` | metadata 加 dcp 字段 / local vs global seq_lens | openclaw 0.22 补丁 port |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | `_dcp_allgather_indexer_logits`（各 rank 本地 logits 散射回全局位置 + all_reduce → 全局一致 top-k）+ prefill 重建 | openclaw 0.22 补丁 port |
| `vllm/model_executor/layers/attention/mla_attention.py` | **无需改** —— 0.23-stock 已含完整 DCP 框架（Q head all-gather + `attn_out,lse=forward_mqa` + `cp_lse_ag_out_rs`/`dcp_a2a_lse_reduce` 合并；bf16 下稀疏 assert 不触发） | — |

## flashmla_sparse.py 的 4 个关键点（bf16 decode 路径）
1. 类属性 `can_return_lse_for_decode: bool = True`。
2. `__init__` 接线 `dcp_world_size`/`dcp_rank` + **`dcp_interleave_size = parallel_config.cp_kv_cache_interleave_size`**（★读侧必须匹配 0.23 写侧 `block_table._compute_slot_mapping_kernel` 用的 `cp_kv_cache_interleave_size`，**不是** `--dcp-kv-cache-interleave-size` flag 设的 `dcp_kv_cache_interleave_size`，否则 slot 错位=输出垃圾）。
3. `_bf16_flash_mla_kernel`：用 `actual_num_heads = q.shape[1]`（DCP all-gather 后 num_heads×dcp）做 padding/切片，替 `self.num_heads`；`flash_mla_sparse_fwd(...)` 返回 `(output, max_logits, lse)`（stock 只取 `[0]` 丢了 lse），捕获并返回 `(output, lse)`；lse `[s_q,h_q]` 已是 [B,H] 布局无需 transpose。
4. `_forward_bf16_kv` 把 `dcp_world_size/dcp_rank/dcp_interleave_size` 传给 `triton_convert_req_index_to_global_index`；`forward_mqa` bf16 分支返回 `(attn_out, lse)`。

## 解题精髓
破"全局 top-2048 vs DCP 切片"冲突：**选择走全局复制**（all-reduce indexer logits → 所有 rank 选同一份 top-2048），**注意力走分片**（owner-rank 过滤 + 跨 rank LSE 合并）；两者用同一 round-robin 几何（粒度 S=interleave，全局位置 p 归 rank (p//S)%n）。

## 启动参数（GLM-5.2-Int4mix，单机 8×H100，1M）
```
-tp8 --decode-context-parallel-size 8 --dcp-kv-cache-interleave-size 64 \
--disable-hybrid-kv-cache-manager --kv-cache-dtype auto --block-size 64 \
--max-model-len 1048576
```
（`--disable-hybrid-kv-cache-manager` 必加：DCP 与 hybrid KV manager 不兼容。KV dtype 用 auto=bf16 求正确性；fp8 会让 DSA 长上下文更差。）

## 验证（hd04 CCI，GLM-5.2-Int4-Int8Mix，TP8+DCP8）
- 17K / **139K needle 命中（33.9s）**；1M KV 装下（1.15M token / 1.1× 并发 / 13.43GiB per-card，bf16+DCP8）。
- 叠 PD 分离 + dfkv（DfkvStoreConnector）跨实例 KV 传输跑通（dfkv 侧改动见 dingodb/dfkv#69、#70）。

## 注意
- `indexer.py` / `sparse_attn_indexer.py` 是 openclaw 0.22 补丁 port，diff 含少量 0.22-vs-0.23 base 差异，但已在 0.23 运行时验证（真机 needle 命中）。`flashmla_sparse.py` 是本仓 0.23-base 干净 port。
- 此改造只覆盖 **bf16 decode 路径**（DCP 用 bf16 KV）；fp8_ds_mla 路径的 DCP 未在本分支验证。
