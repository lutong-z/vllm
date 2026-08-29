# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-static NVFP4 MLA cache-format configuration and ABI identity."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

NVFP4_MLA_SCALES_ENV = "VLLM_NVFP4_MLA_SCALES_FILE"
NVFP4_MLA_DYNAMIC_SCALE_ENV = "VLLM_NVFP4_MLA_DYNAMIC_SCALE"
KV_FP8_ROPE_ENV = "KV_FP8_ROPE"


@dataclass(frozen=True)
class Nvfp4MlaCacheFormat:
    """Immutable writer/reader configuration captured at process import."""

    dynamic_scale: bool
    fp8_rope: bool
    scales_file: str

    @classmethod
    def from_env(cls) -> Nvfp4MlaCacheFormat:
        return cls(
            dynamic_scale=os.getenv(NVFP4_MLA_DYNAMIC_SCALE_ENV, "0") == "1",
            fp8_rope=os.getenv(KV_FP8_ROPE_ENV, "0") == "1",
            scales_file=os.getenv(NVFP4_MLA_SCALES_ENV, "").strip(),
        )

    def validate(self) -> None:
        if self.dynamic_scale and self.scales_file:
            raise ValueError(
                f"{NVFP4_MLA_SCALES_ENV} and "
                f"{NVFP4_MLA_DYNAMIC_SCALE_ENV}=1 are mutually exclusive"
            )
        if self.dynamic_scale and not self.fp8_rope:
            raise ValueError(
                f"{NVFP4_MLA_DYNAMIC_SCALE_ENV}=1 requires {KV_FP8_ROPE_ENV}=1"
            )

    def validate_dsv4_native(self, *, outer_scale: float = 1.0) -> None:
        """Reject native DSV4 modes not implemented by the 432-byte ABI."""
        if self.dynamic_scale or self.scales_file or self.fp8_rope:
            raise ValueError(
                "native DSV4 nvfp4_ds_mla requires dynamic_scale=false, "
                "no scales file, and fp8_rope=false"
            )
        self.validate()
        if float(outer_scale) != 1.0:
            raise ValueError(
                "native DSV4 nvfp4_ds_mla requires identity outer scale 1.0, "
                f"got {outer_scale!r}"
            )

    def validate_dsv4_native_across_ranks(
        self,
        *,
        outer_scale: float = 1.0,
        identity: str | None = None,
    ) -> None:
        """Collect identity state before any rank raises for native DSV4."""
        local = (
            self.dynamic_scale,
            self.fp8_rope,
            self.scales_file,
            float(outer_scale),
            identity,
        )
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                gathered: list[tuple[object, ...] | None] = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, local)
                normalized = [value for value in gathered if value is not None]
                if not normalized or any(value != normalized[0] for value in normalized[1:]):
                    details = "; ".join(
                        f"rank={rank}:{value!r}"
                        for rank, value in enumerate(gathered)
                    )
                    raise ValueError(
                        "native DSV4 NVFP4 ABI configuration differs across ranks: "
                        f"{details}"
                    )
        except (RuntimeError, ValueError):
            raise
        self.validate_dsv4_native(outer_scale=outer_scale)

    def record_abi(self, cache_dtype: str) -> str:
        """Return an identity suitable for persistent external-cache keys."""
        normalized_dtype = str(cache_dtype).replace("torch.", "")
        if normalized_dtype != "nvfp4_ds_mla":
            return "vllm-default-v1"

        self.validate()
        if not self.dynamic_scale and not self.scales_file:
            # Native 432-byte records must never share a prefix namespace with
            # the historical default or the 584-byte FP8 alias.
            layout = "fp8-rope-368" if self.fp8_rope else "bf16-rope-432"
            return f"nvfp4_ds_mla:{layout}:identity-v1"

        layout = "fp8-rope-368" if self.fp8_rope else "bf16-rope-432"
        if self.dynamic_scale:
            scale_mode = "dynamic-token-v1"
        else:
            try:
                scale_digest = hashlib.sha256(
                    Path(self.scales_file).read_bytes()
                ).hexdigest()
            except OSError as exc:
                raise ValueError(
                    f"Cannot fingerprint {NVFP4_MLA_SCALES_ENV}={self.scales_file!r}"
                ) from exc
            scale_mode = f"static-calibrated-v1:{scale_digest}"
        return f"nvfp4_ds_mla:{layout}:{scale_mode}"


# All consumers import this one frozen value, so a process cannot configure
# the writer, readers, and external-cache namespace from different env reads.
NVFP4_MLA_CACHE_FORMAT = Nvfp4MlaCacheFormat.from_env()
