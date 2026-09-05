# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

from tests.v1.kv_connector.unit.utils import create_model_runner_output
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)

from .utils import create_requests, create_scheduler

FAR_FUTURE_NS = 10**30


def test_popleft_n_where_skips_and_unlinks():
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    queue = FreeKVCacheBlockQueue(blocks)

    picked = queue.popleft_n_where(2, lambda b: b.block_id % 2 == 1)
    assert [b.block_id for b in picked] == [1, 3]
    assert queue.num_free_blocks == 4

    # Remaining blocks keep their relative order and the list stays linked.
    rest = queue.popleft_n_where(10, lambda _: True)
    assert [b.block_id for b in rest] == [0, 2, 4, 5]
    assert queue.num_free_blocks == 0
    assert queue.fake_free_list_head.next_free_block is queue.fake_free_list_tail


def test_popleft_n_where_returns_fewer_when_exhausted():
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    queue = FreeKVCacheBlockQueue(blocks)

    assert queue.popleft_n_where(5, lambda b: b.block_id == 1) == [blocks[1]]
    assert queue.num_free_blocks == 1
    assert queue.popleft_n_where(5, lambda b: b.block_id == 1) == []
    assert queue.num_free_blocks == 1


def _cache_all(pool: BlockPool, blocks: list[KVCacheBlock]) -> list:
    hashes = []
    for i, block in enumerate(blocks):
        block_hash = make_block_hash_with_group_id(BlockHash(f"h{i}".encode()), 0)
        block.set_block_hash(block_hash)
        pool.cached_block_hash_to_block.insert(block_hash, block)
        hashes.append(block_hash)
    return hashes


def test_protected_blocks_are_evicted_last():
    # Block 0 is the pool's null block; 4 real blocks remain.
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=16)
    blocks = pool.get_new_blocks(4)
    hashes = _cache_all(pool, blocks)
    pool.free_blocks(blocks)

    # The natural eviction order is queue order (b1, b2, ...). Protect the
    # first two victims; allocation must skip them while others remain.
    blocks[0].protected_until_ns = FAR_FUTURE_NS
    blocks[1].protected_until_ns = FAR_FUTURE_NS

    picked = pool.get_new_blocks(2)
    assert picked == blocks[2:]
    # Protected blocks kept their cache entries; unprotected were evicted.
    assert pool.cached_block_hash_to_block.get_one_block(hashes[0]) is blocks[0]
    assert pool.cached_block_hash_to_block.get_one_block(hashes[1]) is blocks[1]
    assert pool.cached_block_hash_to_block.get_one_block(hashes[2]) is None
    assert pool.cached_block_hash_to_block.get_one_block(hashes[3]) is None
    # Allocated blocks start unprotected for their new owner.
    assert all(b.protected_until_ns == 0 for b in picked)

    # Only protected blocks remain; allocation still makes progress and
    # clears the protection for the new owner.
    picked_protected = pool.get_new_blocks(2)
    assert picked_protected == blocks[:2]
    assert all(b.protected_until_ns == 0 for b in picked_protected)


def test_expire_skips_protected_blocks(monkeypatch):
    now_ns = [10_000_000_000]
    monkeypatch.setattr(
        "vllm.v1.core.block_pool.time.monotonic_ns",
        lambda: now_ns[0],
    )
    pool = BlockPool(
        num_gpu_blocks=2,
        enable_caching=True,
        hash_block_size=16,
        prefix_cache_idle_timeout_seconds=5,
    )
    block = pool.get_new_blocks(1)[0]
    block_hash = make_block_hash_with_group_id(BlockHash(b"ttl"), 0)
    block.set_block_hash(block_hash)
    pool.cached_block_hash_to_block.insert(block_hash, block)
    pool.free_blocks([block])

    # Protection outlasts the idle TTL: the block must not expire.
    block.protected_until_ns = now_ns[0] + 10_000_000_000
    now_ns[0] += 6_000_000_000
    assert pool.expire_cached_blocks() == 0
    assert pool.cached_block_hash_to_block.get_one_block(block_hash) is block

    # Once protection lapses, the idle timeout applies from the free time.
    now_ns[0] += 5_000_000_000
    assert pool.expire_cached_blocks() == 1
    assert pool.cached_block_hash_to_block.get_one_block(block_hash) is None


def test_session_request_blocks_protected_on_finish():
    scheduler = create_scheduler(enable_prefix_caching=True)

    session_req, cold_req = create_requests(
        2, num_tokens=32, req_ids=["warm", "cold"]
    )
    session_req.session_id = "session-1"
    scheduler.add_request(session_req)
    scheduler.add_request(cold_req)

    output = scheduler.schedule()
    (session_block_ids,) = scheduler.kv_cache_manager.get_block_ids(
        session_req.request_id
    )
    (cold_block_ids,) = scheduler.kv_cache_manager.get_block_ids(
        cold_req.request_id
    )
    assert session_block_ids and cold_block_ids

    scheduler.update_from_output(
        output,
        create_model_runner_output([session_req, cold_req], use_eos=True),
    )

    pool = scheduler.kv_cache_manager.block_pool
    now_ns = time.monotonic_ns()
    for block_id in session_block_ids:
        block = pool.blocks[block_id]
        assert block.ref_cnt == 0
        assert block.protected_until_ns > now_ns
    for block_id in cold_block_ids:
        block = pool.blocks[block_id]
        assert block.ref_cnt == 0
        assert block.protected_until_ns == 0
