# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side repair for DFlash/DSpark prefix-cache hits."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

from .utils import create_requests, create_scheduler

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_no_hit_is_unchanged(method: str, monkeypatch: pytest.MonkeyPatch):
    scheduler = _make_fake_scheduler(method, block_size=16)
    calls = _record_truncation(scheduler, monkeypatch)
    blocks = scheduler.kv_cache_manager.empty_kv_cache_blocks

    repaired, count = Scheduler._repair_unaligned_dspark_prefix(
        scheduler, blocks, 0
    )

    assert repaired is blocks
    assert count == 0
    assert calls == []


def test_prefix_disabled_has_no_repair(monkeypatch: pytest.MonkeyPatch):
    scheduler = _make_fake_scheduler(
        "dspark", block_size=16, enable_prefix_caching=False
    )
    calls = _record_truncation(scheduler, monkeypatch)
    blocks = scheduler.kv_cache_manager.empty_kv_cache_blocks

    repaired, repaired_count = Scheduler._repair_unaligned_dspark_prefix(
        scheduler, blocks, 33
    )

    assert repaired is blocks
    assert repaired_count == 33
    assert calls == []


def test_aligned_256_token_hit_is_unchanged(monkeypatch: pytest.MonkeyPatch):
    scheduler = _make_fake_scheduler("dspark", block_size=256)
    calls = _record_truncation(scheduler, monkeypatch)
    blocks = scheduler.kv_cache_manager.empty_kv_cache_blocks

    repaired, count = Scheduler._repair_unaligned_dspark_prefix(
        scheduler, blocks, 2 * 256
    )

    assert repaired is blocks
    assert count == 2 * 256
    assert calls == []


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_unaligned_hit_clamps_to_scheduler_block(
    method: str, monkeypatch: pytest.MonkeyPatch
):
    scheduler = _make_fake_scheduler(method, block_size=256)
    calls = _record_truncation(scheduler, monkeypatch)
    blocks = scheduler.kv_cache_manager.empty_kv_cache_blocks

    repaired, count = Scheduler._repair_unaligned_dspark_prefix(
        scheduler, blocks, 257
    )

    assert repaired is blocks
    assert count == 256
    assert calls == [256]


def test_unaligned_and_aligned_requests_both_keep_drafting_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler = _make_scheduler("dspark", block_size=256, num_blocks=16)
    requests = create_requests(
        num_requests=2,
        num_tokens=300,
        block_size=256,
        req_ids=["unaligned", "aligned"],
    )
    for request in requests:
        scheduler.add_request(request)

    pool = scheduler.kv_cache_manager.block_pool
    blocks_by_id = {
        request.request_id: pool.get_new_blocks(2) for request in requests
    }
    counts = {"unaligned": 257, "aligned": 256}

    def fake_local_hit(request):
        blocks = scheduler.kv_cache_manager.create_kv_cache_blocks(
            (blocks_by_id[request.request_id],)
        )
        return blocks, counts[request.request_id], 0, False

    monkeypatch.setattr(scheduler, "_get_local_prefix_cache_hit", fake_local_hit)
    output = scheduler.schedule()

    by_id = {request.req_id: request for request in output.scheduled_new_reqs}
    assert by_id["unaligned"].num_computed_tokens == 256
    assert by_id["aligned"].num_computed_tokens == 256
    assert output.num_scheduled_tokens["unaligned"] == 44
    assert output.num_scheduled_tokens["aligned"] == 44


def test_repair_does_not_retain_evicted_tail_on_block_reuse():
    scheduler = _make_cache_scheduler("dspark", block_size=16, num_blocks=8)
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    blocks = pool.get_new_blocks(3)
    original_ids = [block.block_id for block in blocks]
    wrapped = manager.create_kv_cache_blocks((blocks,))

    repaired, count = Scheduler._repair_unaligned_dspark_prefix(
        scheduler, wrapped, 33
    )

    assert count == 32
    assert repaired.get_block_ids() == ([original_ids[0], original_ids[1]],)

    pool.free_blocks([blocks[2]])
    reused = pool.get_new_blocks(1)[0]
    assert reused.block_id == original_ids[2]
    assert repaired.get_block_ids() == ([original_ids[0], original_ids[1]],)


@pytest.mark.parametrize(
    "unsupported",
    ["connector", "partial_hash", "mamba_align", "dcp"],
)
def test_repair_is_not_used_for_unsupported_cache_topologies(
    unsupported: str, monkeypatch: pytest.MonkeyPatch
):
    scheduler = _make_fake_scheduler("dspark", block_size=16)
    calls = _record_truncation(scheduler, monkeypatch)
    blocks = scheduler.kv_cache_manager.empty_kv_cache_blocks

    if unsupported == "connector":
        scheduler.connector = SimpleNamespace()
    elif unsupported == "partial_hash":
        scheduler.kv_cache_manager.coordinator.enable_partial_hash_hits = True
    elif unsupported == "mamba_align":
        scheduler.need_mamba_block_aligned_split = True
    else:
        scheduler.dcp_world_size = 2

    repaired, count = Scheduler._repair_unaligned_dspark_prefix(
        scheduler, blocks, 33
    )

    assert repaired is blocks
    assert count == 33
    assert calls == []


def test_second_warm_turn_reuses_aligned_boundary():
    scheduler = _make_cache_scheduler("dspark", block_size=16, num_blocks=8)
    manager = scheduler.kv_cache_manager
    blocks = manager.block_pool.get_new_blocks(3)
    wrapped = manager.create_kv_cache_blocks((blocks,))

    first_blocks, first_count = Scheduler._repair_unaligned_dspark_prefix(
        scheduler, wrapped, 33
    )
    second_blocks, second_count = Scheduler._repair_unaligned_dspark_prefix(
        scheduler, first_blocks, first_count
    )

    assert first_count == second_count == 32
    assert second_blocks is first_blocks
    assert second_blocks.get_block_ids() == (
        [blocks[0].block_id, blocks[1].block_id],
    )




def _make_fake_scheduler(
    method: str | None,
    *,
    block_size: int,
    enable_prefix_caching: bool = True,
):
    group = SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        )
    )
    manager = SimpleNamespace(
        coordinator=SimpleNamespace(
            single_type_managers=(SimpleNamespace(block_size=block_size),),
            enable_partial_hash_hits=False,
        ),
        empty_kv_cache_blocks=KVCacheBlocks(((),)),
        truncate_computed_blocks=lambda blocks, count: blocks,
    )
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=(
                SimpleNamespace(method=method) if method is not None else None
            )
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
        connector=None,
        dcp_world_size=1,
        need_mamba_block_aligned_split=False,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[group]),
        kv_cache_manager=manager,
        block_size=block_size,
    )
    scheduler._repair_unaligned_dspark_prefix_enabled = MethodType(
        Scheduler._repair_unaligned_dspark_prefix_enabled, scheduler
    )
    scheduler._repair_unaligned_dspark_prefix = MethodType(
        Scheduler._repair_unaligned_dspark_prefix, scheduler
    )
    return scheduler


def _make_cache_scheduler(
    method: str,
    *,
    block_size: int,
    num_blocks: int,
):
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    manager = KVCacheManager(
        kv_cache_config,
        max_model_len=128,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
    )
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(method=method)
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        connector=None,
        dcp_world_size=1,
        need_mamba_block_aligned_split=False,
        kv_cache_config=kv_cache_config,
        kv_cache_manager=manager,
        block_size=block_size,
    )
    scheduler._repair_unaligned_dspark_prefix_enabled = MethodType(
        Scheduler._repair_unaligned_dspark_prefix_enabled, scheduler
    )
    scheduler._repair_unaligned_dspark_prefix = MethodType(
        Scheduler._repair_unaligned_dspark_prefix, scheduler
    )
    return scheduler


def _make_scheduler(
    method: str | None,
    *,
    block_size: int,
    enable_prefix_caching: bool = True,
    num_blocks: int = 32,
):
    scheduler = create_scheduler(
        enable_prefix_caching=enable_prefix_caching,
        block_size=block_size,
        num_blocks=num_blocks,
    )
    if method is not None:
        scheduler.vllm_config.speculative_config = SimpleNamespace(method=method)
    return scheduler


def _record_truncation(scheduler, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = scheduler.kv_cache_manager.truncate_computed_blocks

    def record(blocks: KVCacheBlocks, count: int) -> KVCacheBlocks:
        calls.append(count)
        return original(blocks, count)

    monkeypatch.setattr(scheduler.kv_cache_manager, "truncate_computed_blocks", record)
    return calls
