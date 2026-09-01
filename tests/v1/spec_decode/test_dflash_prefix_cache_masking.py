# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash/DSpark draft context masking under prefix caching.

Cache-restored tokens never flow through the target forward, so the draft's
context KV is never written for them. shift_draft_block_tables hides those
slots from the draft's attention by left-shifting each request's block-table
row by the restored whole blocks (seq_lens is shortened to match by
_prepare_dflash_inputs_kernel).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    shift_draft_block_tables,
)

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

DEVICE = "cuda"
BLOCK_SIZE = 16
MAX_BLOCKS = 64
MAX_NUM_REQS = 8


def test_unaligned_cached_prefix_detection():
    speculator = SimpleNamespace(
        num_cached_tokens_np=np.array([32, 35, 64], dtype=np.int32),
        block_tables=SimpleNamespace(kernel_block_sizes=[16]),
    )

    aligned = SimpleNamespace(
        idx_mapping_np=np.array([0, 2], dtype=np.int32),
        num_reqs=2,
    )
    unaligned = SimpleNamespace(
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
    )

    assert not DFlashSpeculator._has_unaligned_cached_prefix(speculator, aligned)
    assert DFlashSpeculator._has_unaligned_cached_prefix(speculator, unaligned)


def _make_block_table(num_reqs: int) -> torch.Tensor:
    # Distinct block ids per (request, slot) so shifts are detectable.
    table = torch.arange(
        MAX_NUM_REQS * MAX_BLOCKS, dtype=torch.int32, device=DEVICE
    ).view(MAX_NUM_REQS, MAX_BLOCKS)
    return table[:num_reqs].contiguous()


@pytest.mark.parametrize(
    "num_cached,expected_shift",
    [
        (0, 0),  # no cache hit: no-op
        (BLOCK_SIZE * 3, 3),  # block-aligned hit (the common APC case)
        (BLOCK_SIZE * 3 + 5, 3),  # unaligned: floor to whole blocks
        (BLOCK_SIZE - 1, 0),  # less than one block: no-op
    ],
)
def test_shift_single_request(num_cached: int, expected_shift: int):
    block_table = _make_block_table(1)
    original = block_table.clone()
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    num_cached_tokens = torch.full(
        (MAX_NUM_REQS,), num_cached, dtype=torch.int32, device=DEVICE
    )

    seq_lens = torch.full(
        (idx_mapping.shape[0],),
        MAX_BLOCKS * BLOCK_SIZE,
        dtype=torch.int32,
        device=DEVICE,
    )
    shift_draft_block_tables(
        block_table, idx_mapping, num_cached_tokens, seq_lens, BLOCK_SIZE
    )

    kept = MAX_BLOCKS - expected_shift
    torch.testing.assert_close(block_table[0, :kept], original[0, expected_shift:])


def test_shift_per_request_and_idx_mapping():
    # Requests in batch order 0..3 map to request-state slots 3..0, with a
    # different cached count per slot. Each row must shift by its own count.
    num_reqs = 4
    block_table = _make_block_table(num_reqs)
    original = block_table.clone()
    idx_mapping = torch.tensor([3, 2, 1, 0], dtype=torch.int32, device=DEVICE)
    # Slot i has i whole cached blocks.
    num_cached_tokens = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE)
    num_cached_tokens[:4] = (
        torch.arange(4, dtype=torch.int32, device=DEVICE) * BLOCK_SIZE
    )

    seq_lens = torch.full(
        (idx_mapping.shape[0],),
        MAX_BLOCKS * BLOCK_SIZE,
        dtype=torch.int32,
        device=DEVICE,
    )
    shift_draft_block_tables(
        block_table, idx_mapping, num_cached_tokens, seq_lens, BLOCK_SIZE
    )

    for batch_idx in range(num_reqs):
        shift = int(idx_mapping[batch_idx])  # slot id == cached blocks
        kept = MAX_BLOCKS - shift
        torch.testing.assert_close(
            block_table[batch_idx, :kept],
            original[batch_idx, shift:],
            msg=f"batch row {batch_idx} (slot {shift})",
        )


def test_shift_large_row_in_place_overlap():
    # Shift smaller than the copy chunk (1024) exercises the overlapping
    # in-place load-before-store path on a long row.
    max_blocks = 4096
    block_table = (
        torch.arange(max_blocks, dtype=torch.int32, device=DEVICE)
        .unsqueeze(0)
        .contiguous()
    )
    original = block_table.clone()
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    num_cached_tokens = torch.full(
        (1,), 7 * BLOCK_SIZE, dtype=torch.int32, device=DEVICE
    )

    seq_lens = torch.full(
        (idx_mapping.shape[0],),
        max_blocks * BLOCK_SIZE,
        dtype=torch.int32,
        device=DEVICE,
    )
    shift_draft_block_tables(
        block_table, idx_mapping, num_cached_tokens, seq_lens, BLOCK_SIZE
    )

    torch.testing.assert_close(block_table[0, : max_blocks - 7], original[0, 7:])


def test_shift_copy_bounded_by_seq_len():
    # Only the blocks referenced by the shifted sequence move; the tail of the
    # row must stay untouched (perf guard for long-context block tables).
    block_table = _make_block_table(1)
    original = block_table.clone()
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    num_cached_tokens = torch.full(
        (MAX_NUM_REQS,), 4 * BLOCK_SIZE, dtype=torch.int32, device=DEVICE
    )
    # Shifted draft length of 3.5 blocks -> exactly 4 blocks copied.
    seq_lens = torch.full(
        (1,), 3 * BLOCK_SIZE + BLOCK_SIZE // 2, dtype=torch.int32, device=DEVICE
    )

    shift_draft_block_tables(
        block_table, idx_mapping, num_cached_tokens, seq_lens, BLOCK_SIZE
    )

    torch.testing.assert_close(block_table[0, :4], original[0, 4:8])
    torch.testing.assert_close(block_table[0, 4:], original[0, 4:])


def test_window_shift_updates_shared_sequence_length_once_for_two_groups():
    tables = [_make_block_table(1), _make_block_table(1)]
    originals = [table.clone() for table in tables]
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    num_cached_tokens = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    seq_lens = torch.full((1,), 8 * BLOCK_SIZE, dtype=torch.int32, device=DEVICE)
    draft_kv_window = 4 * BLOCK_SIZE

    shift_draft_block_tables(
        tables[0],
        idx_mapping,
        num_cached_tokens,
        seq_lens,
        BLOCK_SIZE,
        draft_kv_window,
        update_seq_lens=False,
    )
    shift_draft_block_tables(
        tables[1],
        idx_mapping,
        num_cached_tokens,
        seq_lens,
        BLOCK_SIZE,
        draft_kv_window,
        update_seq_lens=True,
    )

    assert seq_lens.item() == draft_kv_window
    for table, original in zip(tables, originals, strict=True):
        torch.testing.assert_close(table[0, :4], original[0, 4:8])


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_aligned_prefix_metadata_matches_shifted_draft_table():
    """Restored whole blocks are removed without changing logical positions."""
    device = torch.device("cuda")
    max_num_reqs = 1
    max_num_tokens = 8
    num_speculative_steps = 2
    num_query_per_req = 3
    num_context_tokens = 2
    padded_context_tokens = 4

    input_buffers = SimpleNamespace(
        input_ids=torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        positions=torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=np.array([num_context_tokens], dtype=np.int32),
        positions=torch.tensor(
            [BLOCK_SIZE, BLOCK_SIZE + 1], dtype=torch.int64, device=device
        ),
        query_start_loc=torch.tensor(
            [0, num_context_tokens], dtype=torch.int32, device=device
        ),
        idx_mapping=torch.zeros(1, dtype=torch.int32, device=device),
    )
    block_table = torch.tensor([[11, 12, 13, 14]], dtype=torch.int32, device=device)
    num_cached_tokens = torch.tensor([BLOCK_SIZE], dtype=torch.int32, device=device)
    query_slots = torch.full(
        (max_num_tokens,), -1, dtype=torch.int64, device=device
    )
    context_positions = torch.full_like(query_slots, -1)
    context_slots = torch.full_like(query_slots, -1)
    sample_rows = max_num_reqs * num_speculative_steps

    prepare_dflash_inputs(
        input_buffers,
        query_slots,
        context_positions,
        context_slots,
        torch.zeros(sample_rows, dtype=torch.int64, device=device),
        torch.zeros(sample_rows, dtype=torch.int64, device=device),
        torch.full((sample_rows,), -1, dtype=torch.int32, device=device),
        input_batch,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
        torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
        block_table,
        BLOCK_SIZE,
        num_cached_tokens,
        1,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        1024,
        context_num_tokens_padded=padded_context_tokens,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(
        context_positions[:num_context_tokens],
        torch.tensor(
            [BLOCK_SIZE, BLOCK_SIZE + 1], dtype=torch.int64, device=device
        ),
    )
    torch.testing.assert_close(
        context_slots[:num_context_tokens],
        torch.tensor(
            [12 * BLOCK_SIZE, 12 * BLOCK_SIZE + 1],
            dtype=torch.int64,
            device=device,
        ),
    )
    torch.testing.assert_close(
        context_positions[num_context_tokens:padded_context_tokens],
        torch.zeros(
            padded_context_tokens - num_context_tokens,
            dtype=torch.int64,
            device=device,
        ),
    )
    torch.testing.assert_close(
        context_slots[num_context_tokens:padded_context_tokens],
        torch.full(
            (padded_context_tokens - num_context_tokens,),
            PAD_SLOT_ID,
            dtype=torch.int64,
            device=device,
        ),
    )
    torch.testing.assert_close(
        input_buffers.positions[:num_query_per_req],
        torch.tensor(
            [BLOCK_SIZE + 2, BLOCK_SIZE + 3, BLOCK_SIZE + 4],
            dtype=torch.int64,
            device=device,
        ),
    )
    torch.testing.assert_close(
        query_slots[:num_query_per_req],
        torch.tensor(
            [12 * BLOCK_SIZE + 2, 12 * BLOCK_SIZE + 3, 12 * BLOCK_SIZE + 4],
            dtype=torch.int64,
            device=device,
        ),
    )
    torch.testing.assert_close(
        query_slots[num_query_per_req:],
        torch.full(
            (max_num_tokens - num_query_per_req,),
            PAD_SLOT_ID,
            dtype=torch.int64,
            device=device,
        ),
    )
    assert input_buffers.seq_lens[0].item() == (
        num_query_per_req + num_context_tokens
    )

    shift_draft_block_tables(
        block_table,
        input_batch.idx_mapping,
        num_cached_tokens,
        input_buffers.seq_lens,
        BLOCK_SIZE,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(
        block_table[0, :3],
        torch.tensor([12, 13, 14], dtype=torch.int32, device=device),
    )
    assert input_buffers.seq_lens[0].item() == (
        num_query_per_req + num_context_tokens
    )
