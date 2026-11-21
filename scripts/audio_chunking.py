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


def split_with_overlap(samples: np.ndarray, sample_rate: int, chunk_seconds: float, overlap_seconds: float):
    """Return fixed chunks with shared waveform context and their start times."""
    x = np.asarray(samples)
    if x.ndim != 1:
        raise ValueError("samples must be mono (1-D)")
    if sample_rate <= 0 or chunk_seconds <= 0:
        raise ValueError("sample_rate and chunk_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be >= 0 and smaller than chunk_seconds")
    if not x.size:
        return [], []
    size = max(1, int(round(chunk_seconds * sample_rate)))
    stride = size - int(round(overlap_seconds * sample_rate))
    starts = []
    start = 0
    while start < len(x):
        starts.append(start)
        if start + size >= len(x):
            break
        start += stride
    chunks = [x[start:min(start + size, len(x))] for start in starts]
    return chunks, [start / sample_rate for start in starts[1:]]



def merge_timestamped_chunks(
    chunk_results: list[dict],
    chunk_start_seconds: list[float],
    chunk_durations_seconds: list[float],
) -> tuple[str, str]:
    """Merge overlap chunks by token start times, falling back when alignment is unusable.

    Sherpa token timestamps are chunk-local. Shared audio is divided at the
    midpoint of each overlap so only one chunk owns each side of the seam.
    Token text must reconstruct each recognizer text exactly; otherwise the
    timestamp path is rejected rather than risking silent token loss.
    """
    count = len(chunk_results)
    if count == 0:
        return "", "empty"
    if not (len(chunk_start_seconds) == len(chunk_durations_seconds) == count):
        return merge_overlapping_text([str(item.get("text", "")) for item in chunk_results]), "text_fallback"
    normalized = []
    for index, item in enumerate(chunk_results):
        text = str(item.get("text", "")).strip()
        tokens = list(item.get("tokens") or [])
        timestamps = list(item.get("timestamps") or [])
        if not tokens or len(tokens) != len(timestamps):
            return merge_overlapping_text([str(x.get("text", "")) for x in chunk_results]), "text_fallback"
        try:
            times = [float(value) for value in timestamps]
        except (TypeError, ValueError):
            return merge_overlapping_text([str(x.get("text", "")) for x in chunk_results]), "text_fallback"
        if any(value < 0 or value > chunk_durations_seconds[index] + 0.25 for value in times):
            return "".join(str(x.get("text", "")).strip() for x in chunk_results), "text_fallback"
        if any(right < left for left, right in zip(times, times[1:])):
            return "".join(str(x.get("text", "")).strip() for x in chunk_results), "text_fallback"
        rendered = "".join(str(token).replace("▁", " ") for token in tokens)
        if rendered.strip() != text:
            return "".join(str(x.get("text", "")).strip() for x in chunk_results), "text_fallback"
        normalized.append((tokens, times))

    lower_bounds = [float("-inf")] * count
    upper_bounds = [float("inf")] * count
    for index in range(count - 1):
        previous_end = chunk_start_seconds[index] + chunk_durations_seconds[index]
        next_start = chunk_start_seconds[index + 1]
        if next_start >= previous_end:
            continue
        seam = (next_start + previous_end) / 2.0
        upper_bounds[index] = seam
        lower_bounds[index + 1] = seam

    merged_tokens = []
    for index, (tokens, times) in enumerate(normalized):
        start = chunk_start_seconds[index]
        for token, timestamp in zip(tokens, times):
            absolute_time = start + timestamp
            if lower_bounds[index] <= absolute_time < upper_bounds[index]:
                merged_tokens.append(str(token).replace("▁", " "))
    return "".join(merged_tokens).strip(), "timestamp"

def merge_overlapping_text(texts: list[str], min_overlap_chars: int = 4) -> str:
    """Remove only substantial exact suffix/prefix duplicates between adjacent chunks.

    Short matches are intentionally preserved: in Japanese, two-character
    coincidences are common and are not reliable evidence of repeated speech.
    This remains a conservative text-only heuristic; timestamps or token-level
    alignment are preferable when the recognizer exposes them.
    """
    if min_overlap_chars < 1:
        raise ValueError("min_overlap_chars must be positive")
    merged = ""
    for raw in texts:
        part = raw.strip()
        if not part:
            continue
        overlap = 0
        for size in range(min(len(merged), len(part)), min_overlap_chars - 1, -1):
            if merged.endswith(part[:size]):
                overlap = size
                break
        merged += part[overlap:]
    return merged
