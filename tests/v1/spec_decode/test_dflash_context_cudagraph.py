# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
    DFlashContextCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    _DFlashInputBatch,
    prepare_dflash_inputs,
)


def _make_dispatch_manager(*, captured: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
            max_cudagraph_capture_size=8,
        ),
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        max_num_context_tokens=12,
        _capture_descs={},
        _graphs_captured=captured,
    )


def test_context_graph_dispatch_uses_smallest_capture_bucket():
    manager = _make_dispatch_manager()

    DFlashContextCudaGraphManager._init_candidates(manager)

    assert [d.num_tokens for d in manager._capture_descs[CUDAGraphMode.FULL]] == [
        8,
        4,
        2,
        1,
    ]
    assert DFlashContextCudaGraphManager.dispatch_context(manager, 1).num_tokens == 1
    assert DFlashContextCudaGraphManager.dispatch_context(manager, 3).num_tokens == 4
    assert DFlashContextCudaGraphManager.dispatch_context(manager, 7).num_tokens == 8
    assert (
        DFlashContextCudaGraphManager.dispatch_context(manager, 9).cg_mode
        == CUDAGraphMode.NONE
    )


def test_context_graph_dispatch_falls_back_before_capture():
    manager = _make_dispatch_manager(captured=False)
    DFlashContextCudaGraphManager._init_candidates(manager)

    desc = DFlashContextCudaGraphManager.dispatch_context(manager, 3)

    assert desc.cg_mode == CUDAGraphMode.NONE
    assert desc.num_tokens == 3


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_prepare_dflash_inputs_makes_context_graph_padding_inert():
    device = torch.device("cuda")
    max_num_reqs = 4
    max_num_tokens = 16
    num_reqs = 2
    num_speculative_steps = 2
    num_query_per_req = 3
    num_context_tokens = 3
    padded_context_tokens = 8

    input_buffers = SimpleNamespace(
        input_ids=torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        positions=torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
    )
    input_batch = _DFlashInputBatch(
        num_reqs=num_reqs,
        num_scheduled_tokens=np.array([2, 1], dtype=np.int32),
        positions=torch.arange(num_context_tokens, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32, device=device),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device),
    )

    query_slots = torch.full((max_num_tokens,), 777, dtype=torch.int64, device=device)
    context_positions = torch.full_like(query_slots, 777)
    context_slots = torch.full_like(query_slots, 777)
    sample_rows = max_num_reqs * num_speculative_steps
    block_table = torch.arange(
        1,
        1 + max_num_reqs * 4,
        dtype=torch.int32,
        device=device,
    ).view(max_num_reqs, 4)

    prepare_dflash_inputs(
        input_buffers,
        query_slots,
        context_positions,
        context_slots,
        torch.zeros(sample_rows, dtype=torch.int64, device=device),
        torch.zeros(sample_rows, dtype=torch.int64, device=device),
        torch.full((sample_rows,), -1, dtype=torch.int32, device=device),
        torch.zeros(max_num_reqs, dtype=torch.float32, device=device),
        torch.zeros(max_num_reqs, dtype=torch.int64, device=device),
        input_batch,
        torch.zeros(num_reqs, dtype=torch.int32, device=device),
        torch.zeros(num_reqs, dtype=torch.int32, device=device),
        torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
        torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
        torch.zeros(max_num_reqs, dtype=torch.float32, device=device),
        torch.zeros(max_num_reqs, dtype=torch.int64, device=device),
        block_table,
        16,
        torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
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
        torch.arange(num_context_tokens, dtype=torch.int64, device=device),
    )
    torch.testing.assert_close(
        context_slots[:num_context_tokens],
        torch.tensor([16, 17, 82], dtype=torch.int64, device=device),
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


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_dspark_request_row_condensing_keeps_paged_context_identity(monkeypatch):
    """A surviving request keeps its paged draft KV after its batch row moves."""
    from vllm.models.deepseek_v4.nvidia import dspark as dspark_model

    device = torch.device("cuda")
    block_size = 16
    max_num_reqs = 4
    max_num_tokens = 16
    num_speculative_steps = 2
    num_query_per_req = num_speculative_steps

    # Persistent block tables are keyed by request-state slot, not batch row.
    block_tables = BlockTables(
        block_sizes=[block_size],
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=max_num_tokens,
        max_num_blocks_per_group=[4],
        device=device,
        kernel_block_sizes=[block_size],
    )
    # Request A owns stable slot 3 and request B owns stable slot 1.
    block_tables.append_block_ids(3, ([31, 32],), overwrite=True)
    block_tables.append_block_ids(1, ([11, 12],), overwrite=True)
    block_tables.apply_staged_writes()

    last_sampled = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
    last_sampled[3] = 303
    last_sampled[1] = 101
    next_prefill = torch.zeros_like(last_sampled)
    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
    seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
    num_cached_tokens = torch.zeros(
        max_num_reqs, dtype=torch.int32, device=device
    )

    def prepare(
        idx_mapping: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> tuple[
        SimpleNamespace,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        num_reqs = idx_mapping.shape[0]
        input_buffers = SimpleNamespace(
            input_ids=torch.zeros(
                max_num_tokens, dtype=torch.int64, device=device
            ),
            positions=torch.zeros(
                max_num_tokens, dtype=torch.int64, device=device
            ),
            query_start_loc=torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            ),
            seq_lens=torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            ),
        )
        input_batch = _DFlashInputBatch(
            num_reqs=num_reqs,
            num_scheduled_tokens=np.diff(query_start_loc.cpu().numpy()),
            positions=positions,
            query_start_loc=query_start_loc,
            idx_mapping=idx_mapping,
        )
        block_table = block_tables.gather_block_tables(
            idx_mapping, num_reqs_padded=num_reqs
        )[0]
        query_slots = torch.full(
            (max_num_tokens,), PAD_SLOT_ID, dtype=torch.int64, device=device
        )
        context_positions = torch.full_like(query_slots, -1)
        context_slots = torch.full_like(query_slots, -1)
        sample_rows = max_num_reqs * num_speculative_steps
        sample_indices = torch.zeros(
            sample_rows, dtype=torch.int64, device=device
        )
        sample_pos = torch.zeros(sample_rows, dtype=torch.int64, device=device)
        sample_idx_mapping = torch.full(
            (sample_rows,), -1, dtype=torch.int32, device=device
        )
        output_temperature = torch.zeros_like(temperature)
        output_seeds = torch.zeros_like(seeds)

        prepare_dflash_inputs(
            input_buffers,
            query_slots,
            context_positions,
            context_slots,
            sample_indices,
            sample_pos,
            sample_idx_mapping,
            output_temperature,
            output_seeds,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill,
            temperature,
            seeds,
            block_table,
            block_size,
            num_cached_tokens,
            999,
            num_query_per_req,
            num_speculative_steps,
            max_num_reqs,
            max_num_tokens,
            1024,
            sample_from_anchor=True,
        )
        torch.accelerator.synchronize()
        return (
            input_buffers,
            query_slots,
            context_positions,
            context_slots,
            sample_idx_mapping,
        )

    # A and B first occupy batch rows 0 and 1. Their target contexts are
    # ragged, and rejected target rows must not be inserted into draft KV.
    first = prepare(
        torch.tensor([3, 1], dtype=torch.int32, device=device),
        torch.tensor(
            [0, 1, 2, 3, 16, 17, 18], dtype=torch.int64, device=device
        ),
        torch.tensor([0, 4, 7], dtype=torch.int32, device=device),
        torch.tensor([1, 1], dtype=torch.int32, device=device),
        torch.tensor([2, 1], dtype=torch.int32, device=device),
    )
    input_buffers, query_slots, _, context_slots, sample_idx_mapping = first
    assert input_buffers.input_ids[:4].tolist() == [303, 999, 101, 999]
    assert sample_idx_mapping[:4].tolist() == [3, 3, 1, 1]
    assert context_slots[:7].tolist() == [
        31 * block_size,
        31 * block_size + 1,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        12 * block_size,
        12 * block_size + 1,
        PAD_SLOT_ID,
    ]
    assert query_slots[:4].tolist() == [
        31 * block_size + 2,
        31 * block_size + 3,
        12 * block_size + 2,
        12 * block_size + 3,
    ]

    # A finishes. B condenses from batch row 1 to row 0, but its request-state
    # slot remains 1, so both its anchor and paged KV addresses stay B's.
    last_sampled[1] = 111
    condensed = prepare(
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([20, 21, 22], dtype=torch.int64, device=device),
        torch.tensor([0, 3], dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
    )
    (
        input_buffers,
        query_slots,
        context_positions,
        context_slots,
        sample_idx_mapping,
    ) = condensed
    assert input_buffers.input_ids[:2].tolist() == [111, 999]
    assert sample_idx_mapping[:2].tolist() == [1, 1]
    assert context_slots[:3].tolist() == [
        12 * block_size + 4,
        12 * block_size + 5,
        PAD_SLOT_ID,
    ]
    assert query_slots[:2].tolist() == [
        12 * block_size + 6,
        12 * block_size + 7,
    ]

    # The DSpark model forwards these external paged slots directly to the
    # insertion op, including the PAD slot masking the rejected target row.
    inserted: list[tuple[torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(
        dspark_model,
        "_insert_context_kv",
        lambda _attn, _kv, pos, slots: inserted.append((pos, slots)),
    )
    fake_attn = SimpleNamespace(
        q_lora_rank=1,
        fused_wqa_wkv=lambda x: (
            torch.cat((torch.zeros_like(x[:, :1]), x), dim=-1),
            None,
        ),
        kv_norm=lambda x: x,
    )
    fake_model = SimpleNamespace(layers=[SimpleNamespace(attn=fake_attn)])
    dspark_model.DSparkDeepseekV4Model.precompute_and_store_context_kv(
        fake_model,
        torch.ones((3, 2), dtype=torch.float32, device=device),
        context_positions[:3],
        [context_slots[:3]],
    )

    assert len(inserted) == 1
    assert inserted[0][0].tolist() == [20, 21, 22]
    assert inserted[0][1].tolist() == [
        12 * block_size + 4,
        12 * block_size + 5,
        PAD_SLOT_ID,
    ]
