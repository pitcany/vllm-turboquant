"""Codebook JSON sanity checks.

We don't recompute Lloyd-Max from scratch in CI (the SciPy quad calls take
several seconds per codebook). These tests verify that every JSON shipped in
``turboquant/codebooks/`` is well-formed and consistent with its filename, and
that ``get_codebook`` returns the same data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from turboquant.codebook import get_codebook

CODEBOOK_DIR = Path(__file__).resolve().parents[1] / "turboquant" / "codebooks"
_FILENAME_RE = re.compile(r"codebook_d(\d+)_b(\d+)\.json")


def _shipped_codebooks() -> list[tuple[int, int, Path]]:
    out: list[tuple[int, int, Path]] = []
    for path in sorted(CODEBOOK_DIR.glob("codebook_d*_b*.json")):
        m = _FILENAME_RE.match(path.name)
        assert m, f"unexpected codebook filename: {path.name}"
        out.append((int(m.group(1)), int(m.group(2)), path))
    return out


SHIPPED = _shipped_codebooks()


@pytest.mark.unit
def test_codebooks_directory_not_empty() -> None:
    assert SHIPPED, f"no codebooks found in {CODEBOOK_DIR}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "d,bits,path",
    SHIPPED,
    ids=[f"d{d}_b{b}" for d, b, _ in SHIPPED],
)
def test_codebook_well_formed(d: int, bits: int, path: Path) -> None:
    cb = json.loads(path.read_text())
    assert cb["d"] == d
    assert cb["bits"] == bits

    n_clusters = 1 << bits
    centroids = cb["centroids"]
    boundaries = cb["boundaries"]
    assert len(centroids) == n_clusters, f"{path.name} centroids length"
    assert len(boundaries) == n_clusters + 1, f"{path.name} boundaries length"

    # centroids must be strictly increasing
    assert all(a < b for a, b in zip(centroids, centroids[1:])), (
        f"{path.name} centroids not strictly increasing"
    )

    # boundaries must be sorted, start at -1, end at +1
    assert boundaries[0] == pytest.approx(-1.0)
    assert boundaries[-1] == pytest.approx(1.0)
    assert all(a <= b for a, b in zip(boundaries, boundaries[1:])), (
        f"{path.name} boundaries not sorted"
    )

    # each centroid lies inside its bucket
    for i, c in enumerate(centroids):
        assert boundaries[i] <= c <= boundaries[i + 1], (
            f"{path.name}: centroid {i} ({c}) outside its bucket"
        )

    assert cb["mse_per_coord"] > 0


@pytest.mark.unit
def test_get_codebook_matches_disk() -> None:
    """Loader returns exactly what's on disk for a small case."""
    d, bits, path = next(((d, b, p) for d, b, p in SHIPPED if d == 64 and b == 2), (None,) * 3)
    if d is None:
        pytest.skip("no d=64 b=2 codebook shipped")
    on_disk = json.loads(path.read_text())
    loaded = get_codebook(d, bits)
    assert loaded["centroids"] == on_disk["centroids"]
    assert loaded["boundaries"] == on_disk["boundaries"]
