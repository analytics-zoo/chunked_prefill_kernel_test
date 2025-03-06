import torch
import random
from typing import List, Optional, Tuple
# import intel_extension_for_pytorch as ipex  # noqa
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
import pytest

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = 1024
NUM_BLOCKS = 128  # Arbitrary values for testing
PARTITION_SIZE = 512

DTYPES = [torch.float16]
NUM_GEN_SEQS = [1]  # Arbitrary values for testing
NUM_HEADS = [1]
HEAD_SIZES = [64]
BLOCK_SIZES = [32]
USE_ALIBI = [False]
SEEDS = [0]


class TestChunkedPrefill(TestCase):
    def ref_masked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = query.float()
        key = key.float()
        value = value.float()

        attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key)
        if attn_mask is not None:
            attn_mask = attn_mask.float()
            attn_weights = attn_weights + attn_mask
        attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
        out = torch.einsum("hqk,khd->qhd", attn_weights, value)
        return out

    def ref_chunked_prefill_mine(
        self,
        output: torch.Tensor,
        query: torch.Tensor,  # (num_tokens, num_heads, head_size)
        num_queries_per_kv: int,
        # key_cache's new shape is (num_blocks, num_kv_heads, block_size, head_size)
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,  # (num_seqs, max_num_blocks_per_seq)
        cu_seqlen_q: torch.Tensor,  # (num_seqs + 1,)
        cu_seqlen_k: torch.Tensor,  # (num_seqs + 1,)
        max_seqlen_q: int,
        max_seqlen_k: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        causal: bool = True,
    ) -> None:
        # First calculate q and k...
        query = query.to("xpu")
        key_cache = key_cache.to("xpu")
        value_cache = value_cache.to("xpu")
        block_tables = block_tables.to("xpu")
        cu_seqlen_k = cu_seqlen_k.to("xpu")
        cu_seqlen_q = cu_seqlen_q.to("xpu")
        num_query_heads = query.shape[1]
        head_dim = value_cache.shape[3]
        num_kv_heads = value_cache.shape[2]
        block_size = value_cache.shape[1]
        num_batch = cu_seqlen_q.shape[0] - 1
        num_tokens = query.shape[0]
        max_num_blocks_per_seq = block_tables.shape[1]

        key_cache = key_cache.transpose(1, 2).contiguous()
        value_cache = value_cache.transpose(1, 2).contiguous()

        # (num_blocks, block_size, num_kv_heads, head_size)
        # We should get q, k, and v, q should be easy
        key = key_cache[block_tables].view(num_batch, )

    def ref_chunked_prefill(
        self,
        output: torch.Tensor,
        query: torch.Tensor,  # (num_tokens, num_heads, head_size)
        num_queries_per_kv: int,
        # key_cache's new shape is (num_blocks, num_kv_heads, block_size, head_size)
        key_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_size)
        value_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_size,)
        block_tables: torch.Tensor,  # (num_seqs, max_num_blocks_per_seq)
        cu_seqlen_q: torch.Tensor,  # (num_seqs + 1,)
        cu_seqlen_k: torch.Tensor,  # (num_seqs + 1,)
        max_seqlen_q: int,
        max_seqlen_k: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        causal: bool = True,
    ) -> None:
        query = query.to("xpu")
        key_cache = key_cache.to("xpu")
        value_cache = value_cache.to("xpu")
        block_tables = block_tables.to("xpu")
        cu_seqlen_k = cu_seqlen_k.to("xpu")
        cu_seqlen_q = cu_seqlen_q.to("xpu")
        num_query_heads = query.shape[1] 
        # (num_blocks, num_kv_heads, block_size, head_size) 
        head_dim = value_cache.shape[3]
        num_kv_heads = value_cache.shape[1]
        block_size = value_cache.shape[2]
        num_batch = cu_seqlen_q.shape[0] - 1
        num_tokens = query.shape[0]
        max_num_blocks_per_seq = block_tables.shape[1]

        # (num_blocks, num_kv_heads, block_size, head_size) 
        # -> (num_blocks, block_size, num_kv_heads, head_size)  
        key_cache = key_cache.transpose(1, 2).contiguous()
        value_cache = value_cache.transpose(1, 2).contiguous()

        # key_cache[block_tables] -> This will create (num_batch, max_num_blocks_per_seq, block_size, num_kv_heads, head_size)
        key = key_cache[block_tables].view(
            num_batch, max_num_blocks_per_seq * block_size, num_kv_heads, head_dim
        )

        value = value_cache[block_tables].view(
            num_batch, max_num_blocks_per_seq * block_size, num_kv_heads, head_dim
        )

        # num_batch, max_seqlen_k, num_kv_heads, head_dim
        key = key[:, :max_seqlen_k, :, :]
        value = value[:, :max_seqlen_k, :, :]

        # This is used for getting the seqlen_k for each batch -> in shape [batch_size]
        seqlen_k = cu_seqlen_k[1:] - cu_seqlen_k[:-1]
        # This is used for getting the seqlen_q for each batch -> in shape [batch_size]
        seqlen_q = cu_seqlen_q[1:] - cu_seqlen_q[:-1]

        # in shape: [batch_size, 1] -> Assuming this is: [4, 6]
        seqlen_q = seqlen_q.view(-1, 1)
        # in shape: [batch_size, 1] -> Assuming this is: [8, 10] -> this is total_seq_length
        seqlen_k = seqlen_k.view(-1, 1)
        # seqlen_diff is basically the same with context_len, which is (seq_len - query_len)
        seqlen_diff = seqlen_k - seqlen_q
        # Generate a sequence from [0, max_seqlen_q - 1] -> turns to two dimensional
        # Generate a tensor which dimension is [num_batch, max_seqlen_q]
        q_idx_mask = (
            torch.arange(0, max_seqlen_q, device="xpu").view(1, -1).repeat(num_batch, 1)
        )
        # shape [num_batch, max_seqlen_q]
        k_idx_mask = (
            torch.arange(0, max_seqlen_k, device="xpu").view(1, -1).repeat(num_batch, 1)
        )
        # q_mask = (num_batch, max_seqlen_q) < (num_batch, 1) -> This will get broadcast
        # q_mask: (num_batch, max_seqlen_q) selects all the elements that are valid in query
        # Indicates all the valid elements in the query in each batch
        q_mask = q_idx_mask < seqlen_q
        # k_mask: (num_batch, max_seqlen_k) selects all the elements that are valid in key
        k_mask = k_idx_mask < seqlen_k

        # Our purpose is to generate a mask with shape: [num_batch, max_seqlen_q, max_seqlen_k]
        # Within each batch, [max_seqlen_q, max_seqlen_k] will indicate whether the elements will be
        # counted or not.

        # causal_mask_idx: (num_batch, max_seqlen_q) each batch contain elements from [context_len, context_len + max_seqlen_q]
        # After q_mask selection, it will become a one dimensional tensor which only contains
        # query token indices within the total query.
        # It will contain num_tokens elements -> which contains all the index of the tokens.
        causal_mask_idx = (q_idx_mask + seqlen_diff)[q_mask]

        # generate causal mask [max_seqlen_q, max_seqlen_k]
        tril_mask = torch.tril(torch.ones(max_seqlen_k, max_seqlen_k, device="xpu"))
        tril_mask[tril_mask == 0] = float("-inf")
        tril_mask[tril_mask == 1] = 0
        causal_mask = tril_mask[causal_mask_idx]
        causal_mask_padding = torch.empty(
            [num_batch, max_seqlen_q, max_seqlen_k], device="xpu"
        ).fill_(float("-inf"))
        # num_batch, max_seqlen_q, max_seqlen_k with index[num_batch, max_seqlen_q]
        causal_mask_padding[q_mask] = causal_mask
        # to [batch, num_heads, max_seqlen_q, max_seqlen_k]
        causal_mask_padding = causal_mask_padding.unsqueeze(1)

        pad_q = torch.zeros(
            [num_batch, max_seqlen_q, num_query_heads, head_dim],
            device="xpu",
            dtype=query.dtype,
        )
        pad_k = torch.zeros(
            [num_batch, max_seqlen_k, num_kv_heads, head_dim],
            device="xpu",
            dtype=key.dtype,
        )
        pad_v = torch.zeros(
            [num_batch, max_seqlen_k, num_kv_heads, head_dim],
            device="xpu",
            dtype=value.dtype,
        )
        pad_q[q_mask] = query
        pad_k[k_mask] = key[k_mask]
        pad_v[k_mask] = value[k_mask]

        if num_query_heads > num_kv_heads:
            # TODO: check this part
            pad_k = pad_k.view([num_batch, max_seqlen_k, num_kv_heads, 1, head_dim])
            pad_k = pad_k.repeat(1, 1, 1, num_query_heads // num_kv_heads, 1).view(
                [num_batch, max_seqlen_k, num_query_heads, head_dim]
            )
            pad_v = pad_v.view([num_batch, max_seqlen_k, num_kv_heads, 1, head_dim])
            pad_v = pad_v.repeat(1, 1, 1, num_query_heads // num_kv_heads, 1).view(
                [num_batch, max_seqlen_k, num_query_heads, head_dim]
            )
        # permute to [b, h, seq_len, k]
        pad_q = pad_q.permute(0, 2, 1, 3)
        pad_k = pad_k.permute(0, 2, 1, 3)
        pad_v = pad_v.permute(0, 2, 1, 3)
        attn_mask = torch.empty([num_batch, 1, 1, max_seqlen_k], device="xpu").fill_(
            float("-inf")
        )
        attn_mask[:, :, :, :max_seqlen_k].masked_fill_(k_mask[:, None, None, :], 0)
        # [b, h, f, t]
        attn_weights = torch.einsum("bhqd,bhkd->bhqk", pad_q, pad_k)
        attn_weights *= scale
        attn_mask = attn_mask.float()
        attn_weights = attn_weights + attn_mask
        if causal:
            # Adding the causal mask
            attn_weights = attn_weights + causal_mask_padding

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, pad_v.float())
        attn_output = attn_output.permute(0, 2, 1, 3)

        attn_output = (
            attn_output[q_mask].view([-1, num_query_heads, head_dim]).to(output.dtype)
        )
        output.copy_(attn_output)
        return attn_output

    def create_q_buffer(
        self, cu_seqlen_q, num_query_heads, head_size, dtype, init_value=0
    ):
        num_tokens = cu_seqlen_q[-1]
        query = torch.empty(
            num_tokens, num_query_heads, head_size, dtype=dtype, device="cpu"
        )
        if not init_value:
            query.uniform_(-1, 1)
        else:
            query.fill_(init_value)
        return query

    def create_kv_caches(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
        seed: int,
        init_value=0,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        torch.random.manual_seed(seed)
        torch.manual_seed(seed)

        scale = head_size**-0.5
        # key_value_cache_shape = (num_blocks, block_size, num_heads, head_size)
        key_value_cache_shape = (num_blocks, num_heads, block_size, head_size)
        key_caches = []
        for _ in range(num_layers):
            key_cache = torch.empty(size=key_value_cache_shape, dtype=dtype)
            if not init_value:
                key_cache.uniform_(-scale, scale)
            else:
                key_cache.fill_(1)
            key_caches.append(key_cache)

        value_caches = []
        for _ in range(num_layers):
            value_cache = torch.empty(size=key_value_cache_shape, dtype=dtype)
            if not init_value:
                value_cache.uniform_(-scale, scale)
            else:
                value_cache.fill_(1)
            value_caches.append(value_cache)
        return key_caches, value_caches

    def chunk_prefill(
        self,
        num_seqs,
        max_seqlen,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        use_alibi,
        is_causal,
        version,
        dtype,
        seed,
    ) -> None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.manual_seed(seed)

        scale = float(1.0 / (head_size**0.5))
        # TODO(gc): check num_kv_heads
        num_query_heads, num_kv_heads = num_heads, num_kv_heads
        assert num_query_heads % num_kv_heads == 0
        num_queries_per_kv = num_query_heads // num_kv_heads
        alibi_slopes = None
        if use_alibi:
            alibi_slopes = torch.rand(
                num_seqs, max_seqlen, max_seqlen, device="cpu", dtype=dtype
            )
        # This context_lens just means seq_lens -> = context_len + query_len
        # All the keys should be stored within it
        context_lens = [random.randint(1, max_seqlen) for _ in range(num_seqs)]

        max_seqlen_k = max(context_lens)
        context_lens = [0] + context_lens
        context_lens = torch.tensor(context_lens, dtype=torch.int, device="cpu")

        # Create the block tables.NUM_PREFILL_SEQS
        max_num_blocks_per_seq = (max_seqlen_k + block_size - 1) // block_size
        block_tables = []
        for _ in range(num_seqs):
            block_table = [
                # comment(gc): here max_num_blocks_per_seq = num_blocks, check later when we create kv_cache
                random.randint(0, max_num_blocks_per_seq - 1)
                for i in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int, device="cpu")
        cu_seqlen_k = torch.cumsum(context_lens, 0)
        q_lens = context_lens[1:] if version == "chunked_prefill" else [1] * num_seqs
        # for each of the sequence, generate a randomm query_len from range (1, max_lens)
        # max_lens indicate that q_lens = max_lens
        q_lens = [random.randint(1, max_lens) for max_lens in q_lens]
        max_seqlen_q = max(q_lens)
        q_lens = [0] + q_lens
        q_lens_tensor = torch.tensor(q_lens, dtype=torch.int, device="cpu")
        cu_seqlen_q = torch.cumsum(q_lens_tensor, 0)

        query = self.create_q_buffer(cu_seqlen_q, num_query_heads, head_size, dtype)
        key_caches, value_caches = self.create_kv_caches(
            max_num_blocks_per_seq, block_size, 1, num_kv_heads, head_size, dtype, seed
        )
        key_cache, value_cache = key_caches[0], value_caches[0]
        # Call the paged attention kernel.
        output = torch.zeros_like(query)

        xpu_device = torch.device("xpu")
        cu_seqlen_q_xpu = cu_seqlen_q.to(xpu_device).int()
        cu_seqlen_k_xpu = cu_seqlen_k.to(xpu_device).int()
        output_xpu = output.to(xpu_device)
        query_xpu = query.to("xpu")
        key_cache_xpu = key_cache.to(xpu_device)
        value_cache_xpu = value_cache.to(xpu_device)
        block_tables_xpu = block_tables.to(xpu_device)

        alibi_slopes_xpu = None

        # execute ref path of chunked prefill
        output = output.to("xpu")
        self.ref_chunked_prefill(
            output,
            query,
            num_queries_per_kv,
            key_cache,
            value_cache,
            block_tables,
            cu_seqlen_q,
            cu_seqlen_k,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            alibi_slopes,
            is_causal,
        )

        import vllm._C.ops
        seq_lens_tensor_xpu = context_lens[1:].to("xpu")
        q_lens_tensor_xpu = q_lens_tensor[1:].to("xpu")
        context_lens_tensor_xpu = seq_lens_tensor_xpu - q_lens_tensor_xpu
        out = vllm._C.ops.context_attention_forward_v1(query_xpu, key_cache_xpu, value_cache_xpu, block_tables_xpu, cu_seqlen_q_xpu, seq_lens_tensor_xpu , context_lens_tensor_xpu, max_seqlen_k, torch.amax(context_lens_tensor_xpu).item())
        torch.testing.assert_close(output.cpu(), out.cpu(), atol=3e-3, rtol=1e-3)

    @parametrize("num_gen_seqs", [1, 3, 8, 13, 7])
    @parametrize("max_seqlen_k", [8, 1024, 2088])
    @parametrize("num_heads", [16])
    @parametrize("num_kv_heads", [8, 16])
    @parametrize("head_size", [64, 128])
    # @parametrize("head_size", [128])
    @parametrize("block_size", [8])
    @parametrize("use_alibi", [False])
    @parametrize("is_causal", [True])        # comment(gc): is_causal must be set to True
    @parametrize("dtype", [torch.float16])
    @parametrize("seed", [0, 22, 34, 44])
    def test_chunked_prefill(
        self,
        num_gen_seqs,
        max_seqlen_k,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        use_alibi,
        is_causal,
        dtype,
        seed,
    ):
        self.chunk_prefill(
            num_gen_seqs,
            max_seqlen_k,
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            use_alibi,
            is_causal,
            "chunked_prefill",
            dtype,
            seed,
        )

instantiate_parametrized_tests(TestChunkedPrefill)

if __name__ == "__main__":
    run_tests()
