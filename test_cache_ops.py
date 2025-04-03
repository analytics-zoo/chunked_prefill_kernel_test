import random
import torch
import pytest
from typing import List, Tuple
from enum import Enum

# COPYING_DIRECTION = [("xpu", "cpu"), ("xpu", "xpu"), ("cpu", "xpu")]
DTYPES = [torch.half]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112]
# HEAD_SIZES = [64]
BLOCK_SIZES = [16, 32]
# BLOCK_SIZES = [8]
NUM_BLOCKS = [1024, 3600]  # Arbitrary values for testing
# NUM_BLOCKS = [1024]
NUM_MAPPINGS = [256]  # Arbitrary values for testing
SEEDS = [0]


class KVCacheFormat(Enum):
    Paged = 0
    Chunked = 1


CACHE_FORMAT = [KVCacheFormat.Paged, KVCacheFormat.Chunked]


# If paged, the attention shape is:
#         key_cache_shape = (num_blocks, num_heads, head_size // 8, block_size, 8)
#         value_cache_shape = (num_blocks, num_heads, head_size, block_size)
# If chunked, the attention shape is:
#         key_value_cache_shape = (num_blocks, num_heads, block_size, head_size)
def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    cache_format: KVCacheFormat = KVCacheFormat.Paged,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    if cache_format == KVCacheFormat.Chunked:
        key_cache_shape = (num_blocks, num_heads, block_size, head_size)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=dtype)
        key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    if cache_format == KVCacheFormat.Chunked:
        value_cache_shape = (num_blocks, num_heads, block_size, head_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape, dtype=dtype)
        value_cache.uniform_(-scale, scale)
        value_caches.append(value_cache)
    return key_caches, value_caches


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cache_format", CACHE_FORMAT)
@torch.inference_mode()
def test_reshape_and_cache(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    cache_format: KVCacheFormat,
    device: str = "xpu",
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    # Create a random slot mapping.
    # Generate slot_mappings for each of the 
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long)

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        dtype,
        seed,
        device,
        cache_format,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Clone the KV caches.
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()

    cloned_key_cache_1 = key_cache.clone()
    cloned_value_cache_1 = value_cache.clone()

    # Call the reshape_and_cache kernel.
    import vllm._C.ops
    if cache_format == KVCacheFormat.Paged:
        vllm._C.cache_ops.reshape_and_cache(
            key, value, cloned_key_cache_1, cloned_value_cache_1, slot_mapping, "float16", 1.0
        )
        vllm._C.cache_ops.reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping.to(torch.int64), "float16", 1.0
        )
    else:
        vllm._C.cache_ops.reshape_and_cache_ipexllm(
            key, value, cloned_key_cache_1, cloned_value_cache_1, slot_mapping, "float16", 1.0
        )
        vllm._C.cache_ops.reshape_and_cache_ipexllm(
            key, value, key_cache, value_cache, slot_mapping.to(torch.int64), "float16", 1.0
        )

    # Run the reference implementation.
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    if cache_format == KVCacheFormat.Chunked:
        for i in range(num_tokens):
            # key_value_cache_shape = (num_blocks, num_heads, block_size, head_size)
            block_idx = block_indicies[i]
            block_offset = block_offsets[i]
            for head_idx in range(num_heads):
                cloned_key_cache[block_idx, head_idx, block_offset, :] = key[i, head_idx]
                cloned_value_cache[block_idx, head_idx, block_offset, :] = value[i, head_idx]
    else:
        # num_tokens, num_heads, head_size // 8, 8
        reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
        for i in range(num_tokens):
            block_idx = block_indicies[i]
            block_offset = block_offsets[i]
            # (num_blocks, num_heads, head_size // 8, block_size, 8)
            cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
            cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    assert torch.allclose(key_cache, cloned_key_cache, atol=1e-2, rtol=1e-2)
    assert torch.allclose(value_cache, cloned_value_cache, atol=1e-2, rtol=1e-2)
    assert torch.allclose(cloned_key_cache_1, cloned_key_cache, atol=1e-2, rtol=1e-2)
    assert torch.allclose(
        cloned_value_cache_1, cloned_value_cache, atol=1e-2, rtol=1e-2
    )
    torch.set_default_device("cpu")
    torch.xpu.empty_cache()

