# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


@triton.jit
def _dcp_scatter_indexer_logits_kernel(
    local_ptr,
    local_row_stride,
    global_ptr,
    global_row_stride,
    seq_lens_local_ptr,  # int32 [num_rows], per-row LOCAL context length
    N: tl.constexpr,  # dcp_world_size
    RANK: tl.constexpr,  # dcp_rank
    S: tl.constexpr,  # cp_kv_cache_interleave_size
    max_local_cols,  # cdiv(global_width, N)
    global_width,
    BLOCK: tl.constexpr,
):
    # Scatter this rank's local-order logits to their global positions.
    # The KV cache is sharded across `N` ranks by an interleaved round-robin at
    # granularity `S` (mirrors block_table._compute_slot_mapping_kernel). The
    # L-th local token on rank RANK lives in interleave-block ``L // S`` at
    # offset ``L % S``, whose global position is
    #     (L // S) * (N * S) + RANK * S + (L % S).
    # With S == 1 this reduces to ``L * N + RANK`` (per-token round-robin).
    row = tl.program_id(0)
    llen = tl.load(seq_lens_local_ptr + row)
    for i in range(0, max_local_cols, BLOCK):
        L = i + tl.arange(0, BLOCK)
        valid = L < llen
        val = tl.load(local_ptr + row * local_row_stride + L, mask=valid, other=0.0)
        gcol = (L // S) * (N * S) + RANK * S + (L % S)
        gvalid = valid & (gcol < global_width)
        tl.store(global_ptr + row * global_row_stride + gcol, val, mask=gvalid)


def _dcp_allgather_indexer_logits(
    local_logits: torch.Tensor,
    local_seq_lens: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    cp_interleave_size: int = 1,
) -> torch.Tensor:
    """Reconstruct full-sequence indexer logits from per-rank shards under DCP.

    Each rank computed ``local_logits`` over the KV tokens it physically holds
    (local order). We scatter them back to their global sequence positions into
    a zero-filled buffer (same shape/layout as the kernel output, so the
    downstream top-k kernels see the exact layout they expect) and SUM-reduce
    across the DCP group. Every global position is owned by exactly one rank
    (the round-robin partition), so the other ranks contribute 0 and the sum
    equals that rank's real logit; positions beyond the sequence are 0 on every
    rank and are masked out by the global seq_lens in the top-k. The reduce uses
    the DCP group coordinator (not a raw torch.distributed call) so it is issued
    on vLLM's collective stream and stays compatible with CUDA graph capture.
    """
    from vllm.distributed import get_dcp_group

    num_rows, width = local_logits.shape
    global_logits = torch.zeros_like(local_logits)
    seq_lens_local_flat = local_seq_lens.reshape(-1).to(torch.int32).contiguous()
    max_local_cols = (width + dcp_world_size - 1) // dcp_world_size
    _dcp_scatter_indexer_logits_kernel[(num_rows,)](
        local_logits,
        local_logits.stride(0),
        global_logits,
        global_logits.stride(0),
        seq_lens_local_flat,
        dcp_world_size,
        dcp_rank,
        cp_interleave_size,
        max_local_cols,
        width,
        BLOCK=1024,
    )
    return get_dcp_group().all_reduce(global_logits)


def _dcp_reconstruct_full_indexer_k(
    kv_cache: torch.Tensor,
    chunk,
    values_width: int,
    values_dtype: torch.dtype,
    scales_width: int,
    scales_dtype: torch.dtype,
    device: torch.device,
    cp_interleave_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the full prefill context index-K from per-rank DCP shards.

    Mirrors the dense MLA prefill path
    (mla_attention._context_parallel_compute_prefill_context): each rank gathers
    its LOCAL index-K (padded so every rank gathers the same number of tokens),
    we all-gather across the DCP group, then reorg the shards back into global
    token order. The KV cache is sharded by an interleaved round-robin at
    granularity ``S = cp_kv_cache_interleave_size``: global position ``p`` is held
    by rank ``(p // S) % n`` at local index ``(p // (n*S)) * S + (p % S)``. The
    reorg therefore regroups the all-gathered shards ``[n, padded_len, W]`` into
    ``[padded_len//S, n, S, W]`` (interleave-block major) before flattening to
    global order. With S == 1 this is the original transpose-reshape
    (rank-major -> per-token round-robin). Returns (k_quant, k_scale) over the
    full context, in the same layout cp_gather_indexer_k_quant_cache produces
    without DCP.
    """
    from vllm.distributed import get_dcp_group

    n = chunk.dcp_world_size
    s = cp_interleave_size
    sum_padded = int(chunk.local_cu_seq_lens[-1].item())
    local_k = torch.empty((sum_padded, values_width), dtype=values_dtype, device=device)
    local_scale = torch.empty(
        (sum_padded, scales_width), dtype=scales_dtype, device=device
    )
    # Local gather: padded local cu_seq_lens -> this rank's compact local tokens
    # (plus a little block padding) for each request.
    ops.cp_gather_indexer_k_quant_cache(
        kv_cache, local_k, local_scale, chunk.block_table, chunk.local_cu_seq_lens
    )
    # All-gather across the DCP group (rank-major concatenation along dim 0).
    ag_k = (
        get_dcp_group()
        .all_gather(local_k.view(torch.uint8), dim=0)
        .view(values_dtype)
        .view(n, sum_padded, values_width)
    )
    ag_scale = (
        get_dcp_group()
        .all_gather(local_scale, dim=0)
        .view(n, sum_padded, scales_width)
    )
    # Reorg per request back into global token order, trimmed to ctx.
    #   S == 1: [n, P, W] -> transpose -> [P, n, W] -> [P*n, W]  (per-token RR)
    #   S  > 1: [n, P, W] -> [n, P//S, S, W] -> [P//S, n, S, W] -> [P*n, W]
    # The interleave-block-major regroup makes global position
    #   (P//S-block) * (n*S) + rank*S + (offset in S) land contiguously.
    def _reorg(ag, width, padded_len, ctx_len):
        seg = ag[:, offset : offset + padded_len, :]
        if s == 1:
            return seg.transpose(0, 1).reshape(padded_len * n, width)[:ctx_len]
        assert padded_len % s == 0, (
            f"padded local seq len {padded_len} must be a multiple of "
            f"cp_kv_cache_interleave_size {s}"
        )
        return (
            seg.reshape(n, padded_len // s, s, width)
            .permute(1, 0, 2, 3)
            .reshape(padded_len * n, width)[:ctx_len]
        )

    k_segments = []
    s_segments = []
    offset = 0
    for padded_len, ctx_len in zip(
        chunk.padded_local_seq_lens, chunk.global_seq_lens_lst
    ):
        k_segments.append(_reorg(ag_k, values_width, padded_len, ctx_len))
        s_segments.append(_reorg(ag_scale, scales_width, padded_len, ctx_len))
        offset += padded_len
    k_full = torch.cat(k_segments, dim=0)
    s_full = torch.cat(s_segments, dim=0)
    return k_full, s_full


RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            if not chunk.skip_kv_gather:
                if chunk.dcp_world_size > 1:
                    # DCP: the index-K is sharded across ranks; gather each
                    # rank's local shard, all-gather, and reorg back into global
                    # token order so the logits below see the FULL context.
                    assert not use_fp4_cache, (
                        "DCP sparse indexer prefill does not support fp4 cache yet"
                    )
                    k_quant, k_scale = _dcp_reconstruct_full_indexer_k(
                        kv_cache,
                        chunk,
                        values_width=values_spec[0][1],
                        values_dtype=values_spec[1],
                        scales_width=scales_spec[0][1],
                        scales_dtype=scales_spec[1],
                        device=hidden_states.device,
                        cp_interleave_size=chunk.cp_interleave_size,
                    )
                else:
                    k_quant = k_quant_full[: chunk.total_seq_lens]
                    k_scale = k_scale_full[: chunk.total_seq_lens]
                    ops.cp_gather_indexer_k_quant_cache(
                        kv_cache,
                        k_quant,
                        k_scale,
                        chunk.block_table,
                        chunk.cu_seq_lens,
                    )
            # else: reuse k_quant / k_scale from the previous (gathered) chunk.

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            if current_platform.is_xpu():
                if q_scale_slice is not None:
                    raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                    q_slice_cast,
                    k_quant_cast,
                    k_scale_cast,
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
            else:
                logits = fp8_fp4_mqa_logits(
                    (q_slice_cast, q_scale_slice),
                    (k_quant_cast, k_scale_cast),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            num_rows = logits.shape[0]

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            ops.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        # Under DCP the kernel above read only this rank's KV shard (with local
        # seq_lens), producing per-rank logits in local order. Scatter them back
        # to global positions and all-reduce(MAX) so every rank has the full
        # logits and selects an identical GLOBAL top-k. Top-k then runs over the
        # reconstructed logits with the GLOBAL seq_lens; the resulting global
        # logical positions are remapped to this rank's local cache slots by the
        # sparse attention kernel (triton_convert_req_index_to_global_index).
        if decode_metadata.dcp_world_size > 1:
            assert decode_metadata.global_seq_lens is not None
            topk_seq_lens = decode_metadata.global_seq_lens[:batch_size]
            logits = _dcp_allgather_indexer_logits(
                logits,
                seq_lens,
                decode_metadata.dcp_world_size,
                decode_metadata.dcp_rank,
                decode_metadata.cp_interleave_size,
            )
        else:
            topk_seq_lens = seq_lens

        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                topk_seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                topk_seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
