# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
)


def _make_get_num_new_matched_tokens(
    req_num_new_matched_tokens: dict[str, int],
    async_load,
) -> Callable[[Request, int], tuple[int, bool]]:
    def get_num_new_matched_tokens(request: Request, _: int) -> tuple[int, bool]:
        value = req_num_new_matched_tokens.get(request.request_id, 0)
        return value, async_load

    return get_num_new_matched_tokens


@pytest.fixture
def scheduler():
    vllm_config = create_vllm_config(kv_load_failure_policy="recompute")
    return create_scheduler(vllm_config)


@pytest.mark.parametrize(
    "num_prompt_blocks,num_external_computed_blocks,invalid_block_idxs",
    [
        (100, 99, {0, 98}),
        (100, 99, {50, 98}),
        (100, 99, {98}),
    ],
)
def test_async_load_failure(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request1 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request1)
    request2 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request2)
    request3 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request3)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=True)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 3
    for request in scheduler.skipped_waiting:
        assert request.num_computed_tokens == num_external_computed_tokens
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3

    # Simulate a failure in loading some of request2 blocks.
    (req2_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request2.request_id)
    invalid_block_ids = {req2_block_ids[i] for i in invalid_block_idxs}
    model_runner_output = create_model_runner_output(
        reqs=[],
        finished_recving={request1.request_id, request3.request_id},
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    min_invalid_block_idx = min(invalid_block_idxs)

    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 3
    for request in scheduler.skipped_waiting:
        if request.request_id == request2.request_id:
            assert request.num_computed_tokens == (
                min_invalid_block_idx * scheduler.block_size
            )
        else:
            assert request.num_computed_tokens == num_external_computed_tokens
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.failed_recving_kv_req_ids == {request2.request_id}
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3


@pytest.mark.parametrize(
    "num_prompt_blocks,num_external_computed_blocks,invalid_block_idxs",
    [
        (100, 99, {0, 98}),
        (100, 99, {50, 98}),
        (100, 99, {98}),
    ],
)
def test_sync_load_failure(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request1 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request1)
    request2 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request2)
    request3 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request3)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=False)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    # req_id -> num_computed_tokens
    expected_computed_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    assert len(scheduler.running) == 3
    assert len(scheduler_output.scheduled_new_reqs) == 3
    for request in scheduler_output.scheduled_new_reqs:
        assert request.num_computed_tokens == expected_computed_tokens[request.req_id]
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3

    # Simulate a failure in loading some of request2 blocks.
    req2_block_ids = scheduler_output.scheduled_new_reqs[1].block_ids[0]
    invalid_block_ids = {req2_block_ids[i] for i in invalid_block_idxs}
    model_runner_output = create_model_runner_output(
        [request1, request2, request3],
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == request2.request_id
    assert scheduler.running[0].num_computed_tokens == (
        min(invalid_block_idxs) * scheduler.block_size
    )
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3
    assert scheduler.connector.request_finished.call_count == 2


@pytest.mark.parametrize(
    "num_prompt_blocks,"
    "num_external_computed_blocks,"
    "num_common_prefix_blocks,"
    "invalid_block_idxs",
    [
        (100, 99, 50, {0, 49}),
        (100, 99, 50, {25, 49}),
        (100, 99, 50, {49}),
    ],
)
def test_sync_load_failure_with_shared_blocks(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    num_common_prefix_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks >= num_common_prefix_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size
    common_prefix_len = num_common_prefix_blocks * scheduler.block_size

    request1 = create_request(
        num_tokens=num_prompt_tokens, common_prefix_len=common_prefix_len
    )
    scheduler.add_request(request=request1)
    request2 = create_request(
        num_tokens=num_prompt_tokens, common_prefix_len=common_prefix_len
    )
    scheduler.add_request(request=request2)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=False)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    # req_id -> num_computed_tokens
    expected_computed_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: common_prefix_len,
    }

    assert len(scheduler.running) == 2
    assert len(scheduler_output.scheduled_new_reqs) == 2
    for request in scheduler_output.scheduled_new_reqs:
        assert request.num_computed_tokens == expected_computed_tokens[request.req_id]
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2

    # Simulate a failure in loading some of the shared blocks.
    req1_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    invalid_block_ids = {req1_block_ids[i] for i in invalid_block_idxs}
    model_runner_output = create_model_runner_output(
        [request1, request2], invalid_block_ids=invalid_block_ids, use_eos=True
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    # req_id -> num_computed_tokens
    # all the common prefix blocks will be computed by request1
    expected_computed_tokens = {
        request1.request_id: min(invalid_block_idxs) * scheduler.block_size,
        request2.request_id: common_prefix_len,
    }

    assert len(scheduler.running) == 2
    for request in scheduler.running:
        assert (
            request.num_computed_tokens == expected_computed_tokens[request.request_id]
        )
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2


@pytest.mark.parametrize(
    "num_prompt_blocks,num_external_computed_blocks,invalid_block_idxs",
    [
        (100, 99, {0, 50, 98}),
        (100, 99, {98, 50, 0}),
    ],
)
def test_async_progressive_load_failure(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=True)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 1
    assert scheduler.skipped_waiting.peek_request().request_id == request.request_id
    assert request.num_computed_tokens == num_external_computed_tokens
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 1

    min_invalid_block_idx = max(invalid_block_idxs) + 1
    # Simulate failures when progressively loading request blocks.
    for invalid_block_idx in invalid_block_idxs:
        (req_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
        invalid_block_ids = {req_block_ids[invalid_block_idx]}
        model_runner_output = create_model_runner_output(
            reqs=[],
            finished_recving=set(),
            invalid_block_ids=invalid_block_ids,
            use_eos=True,
        )

        scheduler.update_from_output(scheduler_output, model_runner_output)

        min_invalid_block_idx = min(min_invalid_block_idx, invalid_block_idx)

        assert len(scheduler.waiting) == 0
        assert len(scheduler.skipped_waiting) == 1
        assert scheduler.skipped_waiting.peek_request().request_id == request.request_id
        assert request.num_computed_tokens == (
            min_invalid_block_idx * scheduler.block_size
        )
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert scheduler.failed_recving_kv_req_ids == {request.request_id}
        assert scheduler.connector.get_num_new_matched_tokens.call_count == 1


def test_async_load_failure_retries_before_recompute(scheduler: Scheduler):
    """A transient async-load failure retries the external lookup instead of
    recomputing, when the connector accepts a load-state reset."""
    num_prompt_blocks, num_external_computed_blocks = 100, 99
    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: num_external_computed_tokens}, async_load=True
        )
    )
    scheduler.connector.reset_load_state.return_value = True
    scheduler.connector.take_events.return_value = ()
    scheduler.kv_load_retry_backoff_seconds = 0

    scheduler_output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 1

    # First load fails at block 50; the worker then reports the transfer done.
    (req_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[],
            finished_recving=set(),
            invalid_block_ids={req_block_ids[50]},
            use_eos=True,
        ),
    )
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[],
            finished_recving={request.request_id},
            invalid_block_ids=None,
            use_eos=True,
        ),
    )

    # Promotion schedules a retry instead of truncating + recomputing. The
    # request moves to the skipped queue and is re-examined next step.
    scheduler.schedule()
    scheduler.connector.reset_load_state.assert_called_once_with(request)
    assert request.kv_load_retry_count == 1
    assert request.request_id not in scheduler.failed_recving_kv_req_ids
    assert request.request_id not in scheduler.finished_recving_kv_req_ids

    # backoff=0: the next step promotes the request and re-runs the external
    # lookup, starting a fresh async load with new blocks.
    scheduler.schedule()
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == num_external_computed_tokens

    # The retried load succeeds; the request resumes without any recompute.
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[],
            finished_recving={request.request_id},
            invalid_block_ids=None,
            use_eos=True,
        ),
    )
    scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    # The resumed request locally computes the remaining tail; no recompute of
    # the externally loaded prefix occurred.
    assert request.num_computed_tokens >= num_external_computed_tokens
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2


def test_async_load_failure_retry_exhausted_recomputes(scheduler: Scheduler):
    """Once the retry budget is spent, the failure falls back to the
    configured recompute policy."""
    scheduler.max_kv_load_retries = 1
    num_prompt_blocks, num_external_computed_blocks = 100, 99
    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: num_external_computed_tokens}, async_load=True
        )
    )
    scheduler.connector.reset_load_state.return_value = True
    scheduler.connector.take_events.return_value = ()
    scheduler.kv_load_retry_backoff_seconds = 0

    scheduler_output = scheduler.schedule()

    def fail_load_at(scheduler_output, block_idx: int) -> None:
        (block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
        scheduler.update_from_output(
            scheduler_output,
            create_model_runner_output(
                reqs=[],
                finished_recving=set(),
                invalid_block_ids={block_ids[block_idx]},
                use_eos=True,
            ),
        )
        scheduler.update_from_output(
            scheduler_output,
            create_model_runner_output(
                reqs=[],
                finished_recving={request.request_id},
                invalid_block_ids=None,
                use_eos=True,
            ),
        )

    # First failure: retry is scheduled now; re-admission happens next step.
    fail_load_at(scheduler_output, 50)
    scheduler.schedule()
    assert request.kv_load_retry_count == 1
    scheduler_output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # Second failure: budget spent -> truncate to the valid prefix and
    # recompute locally, without another connector reset.
    fail_load_at(scheduler_output, 25)
    scheduler.schedule()
    assert scheduler.connector.reset_load_state.call_count == 1
    assert request.kv_load_retry_count == 1
    # Truncated to the valid prefix; async scheduling may already count the
    # first locally recomputed chunk on top of it.
    assert 25 * scheduler.block_size <= request.num_computed_tokens < (
        num_external_computed_tokens
    )


def test_async_load_failure_retry_backoff_gates_promotion(scheduler: Scheduler):
    """A pending retry stays blocked until its backoff deadline passes."""
    num_prompt_blocks, num_external_computed_blocks = 100, 99
    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: num_external_computed_tokens}, async_load=True
        )
    )
    scheduler.connector.reset_load_state.return_value = True
    scheduler.connector.take_events.return_value = ()
    scheduler.kv_load_retry_backoff_seconds = 1000

    scheduler_output = scheduler.schedule()
    (req_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[],
            finished_recving=set(),
            invalid_block_ids={req_block_ids[50]},
            use_eos=True,
        ),
    )
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[],
            finished_recving={request.request_id},
            invalid_block_ids=None,
            use_eos=True,
        ),
    )

    # Retry is scheduled, but the 1000s backoff keeps the request blocked.
    scheduler.schedule()
    assert request.kv_load_retry_pending
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 1

    # Deadline reached: the gate opens and admission re-runs the lookup.
    request.kv_load_retry_not_before = 0.0
    scheduler.schedule()
    assert not request.kv_load_retry_pending
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_async_load_failure_multi_group_truncates_without_crash():
    """Hybrid (multi-group) KV cache: invalid blocks must truncate the
    computed prefix at the earliest invalid position across groups instead
    of crashing on the single-group unpack."""
    vllm_config = create_vllm_config(kv_load_failure_policy="recompute")
    kv_cache_config = make_kv_cache_config(block_size=16, swa_enabled=True, num_blocks=1000)
    scheduler = create_scheduler(vllm_config, kv_cache_config=kv_cache_config)

    num_prompt_blocks, num_external_computed_blocks = 100, 99
    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: num_external_computed_tokens}, async_load=True
        )
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # Fail block 50 of the full-attention group. The sliding-window
    # group's entries are null-block placeholders during the async load;
    # including one proves null blocks cannot force a full truncation.
    group_block_ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    assert len(group_block_ids) == 2
    invalid_block_ids = {group_block_ids[0][50], group_block_ids[1][10]}
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[],
            finished_recving=set(),
            invalid_block_ids=invalid_block_ids,
            use_eos=True,
        ),
    )
    assert request.num_computed_tokens == 50 * scheduler.block_size
    assert scheduler.failed_recving_kv_req_ids == {request.request_id}

    # The transfer then completes; the request recovers via recompute
    # without any engine crash.
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[],
            finished_recving={request.request_id},
            invalid_block_ids=None,
            use_eos=True,
        ),
    )
    scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens >= 50 * scheduler.block_size
