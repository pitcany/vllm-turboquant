"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(
    params=[
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            id="cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
        ),
    ]
)
def device(request) -> torch.device:
    return torch.device(request.param)
