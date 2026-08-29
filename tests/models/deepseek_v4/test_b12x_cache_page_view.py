# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.b12x import _b12x_cache_page_view

_PAGE_SIZE = 64
_PAYLOAD_BYTES = _PAGE_SIZE * 584
_PADDED_PAGE_BYTES = 37_440


def test_b12x_cache_page_view_accepts_contiguous_payload_pages() -> None:
    cache = torch.empty((3, _PAGE_SIZE, 584), dtype=torch.uint8)

    view = _b12x_cache_page_view(cache, _PAGE_SIZE, "cache", "fp8_ds_mla")

    assert view.shape == (3, _PAYLOAD_BYTES)
    assert view.stride() == (_PAYLOAD_BYTES, 1)


def test_b12x_cache_page_view_preserves_padded_physical_stride() -> None:
    storage = torch.empty(3 * _PADDED_PAGE_BYTES, dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(3, _PAGE_SIZE, 584),
        stride=(_PADDED_PAGE_BYTES, 584, 1),
    )

    view = _b12x_cache_page_view(cache, _PAGE_SIZE, "cache", "fp8_ds_mla")

    assert view.shape == (3, _PAYLOAD_BYTES)
    assert view.stride() == (_PADDED_PAGE_BYTES, 1)


def test_b12x_cache_page_view_rejects_short_physical_stride() -> None:
    storage = torch.empty(2 * _PAYLOAD_BYTES, dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(2, _PAGE_SIZE, 584),
        stride=(_PAYLOAD_BYTES - 1, 584, 1),
    )

    with pytest.raises(RuntimeError, match=r"page stride .* is smaller"):
        _b12x_cache_page_view(cache, _PAGE_SIZE, "cache", "fp8_ds_mla")


def test_b12x_cache_page_view_rejects_overlapping_2d_pages() -> None:
    storage = torch.empty(2 * _PAYLOAD_BYTES, dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(2, _PAYLOAD_BYTES),
        stride=(_PAYLOAD_BYTES - 1, 1),
    )

    with pytest.raises(RuntimeError, match=r"page stride .* is smaller"):
        _b12x_cache_page_view(cache, _PAGE_SIZE, "cache", "fp8_ds_mla")


def test_b12x_cache_page_view_rejects_gapped_page_payload() -> None:
    storage = torch.empty(2 * _PAYLOAD_BYTES + _PAGE_SIZE, dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(2, _PAGE_SIZE, 584),
        stride=(_PAYLOAD_BYTES, 585, 1),
    )

    with pytest.raises(RuntimeError, match=r"page payload must be contiguous"):
        _b12x_cache_page_view(cache, _PAGE_SIZE, "cache", "fp8_ds_mla")


def test_b12x_cache_page_view_native_432_geometry() -> None:
    page_size = 64
    cache = torch.empty((3, page_size, 432), dtype=torch.uint8)
    view = _b12x_cache_page_view(cache, page_size, "native", "nvfp4_ds_mla")
    assert view.shape == (3, page_size * 432)
    assert view.stride() == (page_size * 432, 1)


def test_b12x_cache_page_view_native_preserves_padded_stride() -> None:
    page_size = 2
    payload_bytes = page_size * 432
    padded_bytes = 1152
    storage = torch.empty(3 * padded_bytes, dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(3, page_size, 432),
        stride=(padded_bytes, 432, 1),
    )
    view = _b12x_cache_page_view(cache, page_size, "native", "nvfp4_ds_mla")
    assert view.stride() == (padded_bytes, 1)


def test_dspark_context_insert_passes_native_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from vllm.models.deepseek_v4.nvidia import dspark

    calls: list[tuple[object, ...]] = []

    def fake_insert(*args):
        calls.append(args)

    class _Ops:
        fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert = staticmethod(fake_insert)

    monkeypatch.setattr(torch.ops, "_C", _Ops, raising=False)
    attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.empty((1, 64, 432), dtype=torch.uint8),
            block_size=64,
        ),
        rotary_emb=SimpleNamespace(
            cos_sin_cache=torch.empty((1, 64), dtype=torch.float32),
        ),
        n_local_heads=2,
        head_dim=512,
        padded_heads=2,
        eps=1e-6,
        kv_cache_dtype="nvfp4_ds_mla",
    )
    kv = torch.zeros((1, 512), dtype=torch.bfloat16)
    dspark._insert_context_kv(
        attn,
        kv,
        torch.zeros((1,), dtype=torch.int64),
        torch.zeros((1,), dtype=torch.int64),
    )
    assert len(calls) == 1
    assert calls[0][-1] is True


def test_hybrid_swa_contract_keeps_format_explicit() -> None:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend

    assert DeepseekSparseSWABackend.get_kv_cache_shape(
        1, 64, 1, 512, "fp8_ds_mla"
    ) == (1, 64, 584)
    assert DeepseekSparseSWABackend.get_kv_cache_shape(
        1, 64, 1, 512, "nvfp4_ds_mla"
    ) == (1, 64, 432)


def test_dsv4_resolver_fails_closed_without_native_b12x_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import b12x.attention.compressed_mla.api as b12x_api
    from vllm.models.deepseek_v4.attention import _resolve_dsv4_kv_cache_dtype

    monkeypatch.setattr(b12x_api, "supports_dsv4_nvfp4", lambda *args, **kwargs: False)
    with pytest.raises(ValueError, match="native DSV4 nvfp4_ds_mla reader"):
        _resolve_dsv4_kv_cache_dtype(
            True,
            "nvfp4_ds_mla",
            SimpleNamespace(cache_dtype="nvfp4_ds_mla"),
        )
