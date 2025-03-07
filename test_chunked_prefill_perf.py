import torch
import random
import time
from typing import List, Optional, Tuple
import itertools

alibi_slopes = None
USE_ALIBI = False
WARMUP = 3
ITERATION_NUM = 500
NUM_HEADS = [32] # num_query_heads, num_kv_headss
NUM_KV_HEADS = [4,8,32]
HEAD_SIZE = [128]
BLOCK_SIZE = [8]
MAX_SEQ_LEN = [1024, 2048]
NUM_SEQS = [8]
DTYPE = torch.float16
SEED = 0
DEVICE = torch.device("xpu")

random.seed(SEED)
torch.random.manual_seed(SEED)
torch.manual_seed(SEED)


def create_q_buffer(
    cu_seqlen_q, num_query_heads, head_size, dtype, init_value=0
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


# Iterate over all the possible values
def random_test():
    for num_heads, num_kv_heads, head_size, block_size, max_seq_len, num_seqs in itertools.product(
        NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE, MAX_SEQ_LEN, NUM_SEQS):
        scale = float(1.0 / (head_size**0.5))
        num_query_heads = num_heads
        num_kv_heads = num_kv_heads
        assert num_query_heads % num_kv_heads == 0
        num_queries_per_kv = num_query_heads // num_kv_heads

        import vllm._C.ops
        # This context_lens just means seq_lens -> = context_len + query_len
            # All the keys should be stored within it
        context_lens = [random.randint(1, max_seq_len) for _ in range(num_seqs)]

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
        q_lens = context_lens[1:]
        q_lens = [random.randint(1, max_lens) for max_lens in q_lens]
        max_seqlen_q = max(q_lens)
        q_lens = [0] + q_lens
        q_lens_tensor = torch.tensor(q_lens, dtype=torch.int, device="cpu")
        cu_seqlen_q = torch.cumsum(q_lens_tensor, 0)
        query = create_q_buffer(cu_seqlen_q, num_query_heads, head_size, DTYPE)
        key_caches, value_caches = create_kv_caches(
            max_num_blocks_per_seq, block_size, 1, num_kv_heads, head_size, DTYPE, SEED)
        key_cache, value_cache = key_caches[0], value_caches[0]
        xpu_device = torch.device("xpu")
        cu_seqlen_q_xpu = cu_seqlen_q.to(xpu_device).int()
        query_xpu = query.to(xpu_device)
        key_cache_xpu = key_cache.to(xpu_device)
        value_cache_xpu = value_cache.to(xpu_device)
        block_tables_xpu = block_tables.to(xpu_device)
        output = torch.zeros_like(query)
        seq_lens_tensor_xpu = context_lens[1:].to("xpu")
        q_lens_tensor_xpu = q_lens_tensor[1:].to("xpu")
        context_lens_tensor_xpu = seq_lens_tensor_xpu - q_lens_tensor_xpu
        max_context_len = torch.amax(context_lens_tensor_xpu).item()
        # Will need to execute multiple times
        total_time = 0
        print(f"===================================Performing random test start ==================================")
        # TODO: add more info
        print(f"Benchmark info:")
        print(f"NUM_HEADS: {num_heads}, NUM_KV_HEADS: {num_kv_heads}, HEAD_SIZE: {head_size}, BLOCK_SIZE: {block_size}, MAX_SEQ_LEN: {max_seq_len}, NUM_SEQS: {num_seqs}")
        print(f"Seq lens: {seq_lens_tensor_xpu.tolist()}")
        print(f"Context lens: {context_lens_tensor_xpu.tolist()}")
        print(f"Query lens: {(seq_lens_tensor_xpu - context_lens_tensor_xpu).tolist()}")
        for i in range(WARMUP + ITERATION_NUM):
            torch.xpu.synchronize()
            st = time.time()
            out = vllm._C.ops.context_attention_forward_v1(query_xpu, key_cache_xpu, value_cache_xpu, block_tables_xpu, cu_seqlen_q_xpu, seq_lens_tensor_xpu , context_lens_tensor_xpu, max_seqlen_k, max_context_len)
            torch.xpu.synchronize()
            et = time.time()
            if i >= WARMUP:
                total_time += (et-st) * 1000
        torch.xpu.empty_cache()
        print(f"Chunked prefill kernel take time: {total_time/ITERATION_NUM:.2f} us")
        print(f"===================================Performing random test end ==================================\n")

if __name__ == "__main__":
    random_test()