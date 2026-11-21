"""Silence-aware chunk boundary selection for long-form audio."""
from __future__ import annotations

import numpy as np


def split_at_low_energy(
    samples: np.ndarray,
    sample_rate: int,
    target_seconds: float,
    search_window_seconds: float = 2.0,
    frame_seconds: float = 0.02,
    min_chunk_seconds: float = 3.0,
) -> tuple[list[np.ndarray], list[float]]:
    """Split into contiguous chunks, moving each target boundary to the quietest nearby frame.

    The algorithm never drops or overlaps samples. If no eligible low-energy
    frame exists in the search window, it falls back to the nominal boundary.
    """
    x = np.asarray(samples)
    if x.ndim != 1:
        raise ValueError("samples must be mono (1-D)")
    if sample_rate <= 0 or target_seconds <= 0 or frame_seconds <= 0:
        raise ValueError("sample_rate, target_seconds, and frame_seconds must be positive")
    if search_window_seconds < 0 or min_chunk_seconds <= 0:
        raise ValueError("search_window_seconds must be >= 0 and min_chunk_seconds > 0")
    n = len(x)
    if n == 0:
        return [], []
    target = max(1, int(round(target_seconds * sample_rate)))
    min_size = max(1, int(round(min_chunk_seconds * sample_rate)))
    window = int(round(search_window_seconds * sample_rate))
    frame = max(1, int(round(frame_seconds * sample_rate)))
    boundaries: list[int] = []
    start = 0
    while n - start > target + min_size // 2:
        nominal = start + target
        lo = max(start + min_size, nominal - window)
        hi = min(n - min_size, nominal + window)
        candidates = range(((lo + frame - 1) // frame) * frame, hi + 1, frame)
        best = nominal
        best_energy = float("inf")
        for point in candidates:
            left = max(start, point - frame // 2)
            right = min(n, point + frame // 2)
            if right <= left:
                continue
            energy = float(np.mean(np.square(x[left:right], dtype=np.float64)))
            if energy < best_energy:
                best, best_energy = point, energy
        if best <= start or best >= n:
            best = nominal
        boundaries.append(best)
        start = best
    edges = [0, *boundaries, n]
    chunks = [x[a:b] for a, b in zip(edges, edges[1:])]
    return chunks, [b / sample_rate for b in boundaries]
