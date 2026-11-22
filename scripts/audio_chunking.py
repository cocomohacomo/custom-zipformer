"""Silence-aware chunk boundary selection for long-form audio."""
from __future__ import annotations

import math
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
    if np.issubdtype(x.dtype, np.inexact) and not np.isfinite(x).all():
        raise ValueError("samples must contain only finite values")
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
    if np.issubdtype(x.dtype, np.inexact) and not np.isfinite(x).all():
        raise ValueError("samples must contain only finite values")
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
        return _concatenate_chunk_text([str(item.get("text", "")) for item in chunk_results]), "text_fallback"
    # Invalid timing metadata must never silently discard tokens at a seam.
    try:
        starts = [float(value) for value in chunk_start_seconds]
        durations = [float(value) for value in chunk_durations_seconds]
    except (TypeError, ValueError):
        return _concatenate_chunk_text([str(item.get("text", "")) for item in chunk_results]), "text_fallback"
    if any(not math.isfinite(value) for value in starts + durations):
        return _concatenate_chunk_text([str(item.get("text", "")) for item in chunk_results]), "text_fallback"
    if any(value <= 0 for value in durations):
        return _concatenate_chunk_text([str(item.get("text", "")) for item in chunk_results]), "text_fallback"
    if starts[0] < 0 or any(right <= left for left, right in zip(starts, starts[1:])):
        return _concatenate_chunk_text([str(item.get("text", "")) for item in chunk_results]), "text_fallback"
    normalized = []
    for index, item in enumerate(chunk_results):
        text = str(item.get("text", "")).strip()
        tokens = list(item.get("tokens") or [])
        timestamps = list(item.get("timestamps") or [])
        if not tokens or len(tokens) != len(timestamps):
            return _concatenate_chunk_text([str(x.get("text", "")) for x in chunk_results]), "text_fallback"
        try:
            times = [float(value) for value in timestamps]
        except (TypeError, ValueError):
            return _concatenate_chunk_text([str(x.get("text", "")) for x in chunk_results]), "text_fallback"
        if any(not math.isfinite(value) or value < 0 or value > durations[index] + 0.25 for value in times):
            return _concatenate_chunk_text([str(x.get("text", "")) for x in chunk_results]), "text_fallback"
        if any(right < left for left, right in zip(times, times[1:])):
            return _concatenate_chunk_text([str(x.get("text", "")) for x in chunk_results]), "text_fallback"
        rendered = "".join(str(token).replace("▁", " ") for token in tokens)
        if rendered.strip() != text:
            return _concatenate_chunk_text([str(x.get("text", "")) for x in chunk_results]), "text_fallback"
        normalized.append((tokens, times))

    lower_bounds = [float("-inf")] * count
    upper_bounds = [float("inf")] * count
    for index in range(count - 1):
        previous_end = starts[index] + durations[index]
        next_start = starts[index + 1]
        if next_start >= previous_end:
            continue
        seam = (next_start + previous_end) / 2.0
        upper_bounds[index] = seam
        lower_bounds[index + 1] = seam

    merged_tokens = []
    for index, (tokens, times) in enumerate(normalized):
        start = starts[index]
        for token, timestamp in zip(tokens, times):
            absolute_time = start + timestamp
            if lower_bounds[index] <= absolute_time < upper_bounds[index]:
                merged_tokens.append(str(token).replace("▁", " "))
    return "".join(merged_tokens).strip(), "timestamp"

def _concatenate_chunk_text(texts: list[str]) -> str:
    """Preserve all text when fallback overlap cannot be proven from timing."""
    return "".join(str(text).strip() for text in texts if str(text).strip())


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


def split_fixed_chunks(samples: np.ndarray, sample_rate: int, chunk_seconds: float):
    """Split mono audio into contiguous chunks, retaining the final partial chunk."""
    x = np.asarray(samples)
    if x.ndim != 1:
        raise ValueError("samples must be mono (1-D)")
    if np.issubdtype(x.dtype, np.inexact) and not np.isfinite(x).all():
        raise ValueError("samples must contain only finite values")
    if sample_rate <= 0 or chunk_seconds <= 0:
        raise ValueError("sample_rate and chunk_seconds must be positive")
    if not x.size:
        return []
    size = max(1, int(round(chunk_seconds * sample_rate)))
    return [x[start:min(start + size, len(x))] for start in range(0, len(x), size)]



def diagnose_timestamped_chunk_merge(chunk_results, chunk_start_seconds, chunk_durations_seconds):
    """Return count-only accounting for the existing timestamp merge decision.

    The merge implementation and its output are deliberately untouched. Fallback
    is a separate bucket because text concatenation does not select tokens.
    """
    merged, method = merge_timestamped_chunks(
        chunk_results, chunk_start_seconds, chunk_durations_seconds
    )
    input_count = sum(len(item.get("tokens") or []) for item in chunk_results)
    input_chars = sum(
        len(str(token).replace("▁", " "))
        for item in chunk_results for token in (item.get("tokens") or [])
    )
    kept = rejected = kept_chars = rejected_chars = 0
    boundary_counts = []
    rejection_detail = []
    distance_edges = (0.10, 0.25, 0.50)
    distance_labels = ("lt_100ms", "100_250ms", "250_500ms", "gte_500ms")
    if method == "timestamp":
        n = len(chunk_results)
        starts = [float(v) for v in chunk_start_seconds]
        durations = [float(v) for v in chunk_durations_seconds]
        lower = [float("-inf")] * n
        upper = [float("inf")] * n
        lower_seams = [None] * n
        upper_seams = [None] * n
        for i in range(n - 1):
            previous_end = starts[i] + durations[i]
            next_start = starts[i + 1]
            if next_start < previous_end:
                seam = (next_start + previous_end) / 2.0
                upper[i] = seam
                lower[i + 1] = seam
                upper_seams[i] = seam
                lower_seams[i + 1] = seam
        detail_counts = {}
        rejected_records = []
        kept_records = []
        for i, item in enumerate(chunk_results):
            for token_index, (token, timestamp) in enumerate(zip(item.get("tokens") or [], item.get("timestamps") or [])):
                rendered = str(token).replace("▁", " ")
                absolute_time = starts[i] + float(timestamp)
                record = {"chunk": i, "token_index": token_index, "text": rendered, "time": absolute_time}
                if lower[i] <= absolute_time < upper[i]:
                    kept += 1
                    kept_chars += len(rendered)
                    kept_records.append(record)
                else:
                    rejected += 1
                    rejected_chars += len(rendered)
                    side = seam = neighbor = None
                    if absolute_time >= upper[i] and upper_seams[i] is not None:
                        side, seam, neighbor = "after_seam_from_left_chunk", upper_seams[i], i + 1
                    elif absolute_time < lower[i] and lower_seams[i] is not None:
                        side, seam, neighbor = "before_seam_from_right_chunk", lower_seams[i], i - 1
                    if side is not None:
                        distance = abs(absolute_time - seam)
                        bucket_index = next((j for j, edge in enumerate(distance_edges) if distance < edge), len(distance_edges))
                        key = (side, distance_labels[bucket_index])
                        counts = detail_counts.setdefault(key, [0, 0])
                        counts[0] += 1
                        counts[1] += len(rendered)
                        rejected_records.append({**record, "side": side, "distance_bucket": distance_labels[bucket_index],
                                                 "neighbor": neighbor})
            local = [record for record in rejected_records if record["chunk"] == i]
            if local:
                boundary_counts.append({
                    "chunk_position": i,
                    "rejected_tokens": len(local),
                    "rejected_token_chars": sum(len(record["text"]) for record in local),
                })
        rejection_detail = [
            {"side": side, "distance_bucket": bucket, "rejected_tokens": counts[0],
             "rejected_token_chars": counts[1]}
            for (side, bucket), counts in sorted(detail_counts.items())
        ]

        # Match only against kept tokens in the directly adjacent chunk.
        # Exact token text and <=1s absolute-time distance are required.
        # Greedy minimum-distance assignment guarantees one-to-one matches.
        max_match_delta_seconds = 1.0
        candidate_edges, candidate_counts = [], [0] * len(rejected_records)
        kept_by_chunk = {}
        for record in kept_records:
            kept_by_chunk.setdefault(record["chunk"], []).append(record)
        for rejected_index, record in enumerate(rejected_records):
            for kept_record in kept_by_chunk.get(record["neighbor"], []):
                delta = abs(record["time"] - kept_record["time"])
                if record["text"] == kept_record["text"] and delta <= max_match_delta_seconds:
                    candidate_counts[rejected_index] += 1
                    candidate_edges.append((delta, rejected_index, kept_record["chunk"], kept_record["token_index"]))
        used_rejected, used_kept, correspondence = set(), set(), {}
        for delta, rejected_index, chunk_index, token_index in sorted(candidate_edges):
            kept_key = (chunk_index, token_index)
            if rejected_index in used_rejected or kept_key in used_kept:
                continue
            used_rejected.add(rejected_index)
            used_kept.add(kept_key)
            record = rejected_records[rejected_index]
            status = "matched_ambiguous" if candidate_counts[rejected_index] > 1 else "matched_unique"
            key = (record["side"], record["distance_bucket"], status)
            counts = correspondence.setdefault(key, [0, 0])
            counts[0] += 1
            counts[1] += len(record["text"])
        for rejected_index, record in enumerate(rejected_records):
            if rejected_index not in used_rejected:
                key = (record["side"], record["distance_bucket"], "unmatched")
                counts = correspondence.setdefault(key, [0, 0])
                counts[0] += 1
                counts[1] += len(record["text"])
        correspondence_detail = [
            {"side": side, "distance_bucket": bucket, "status": status,
             "rejected_tokens": counts[0], "rejected_token_chars": counts[1]}
            for (side, bucket, status), counts in sorted(correspondence.items())
        ]
    return merged, method, {
        "input_tokens": input_count,
        "keep_tokens": kept,
        "reject_tokens": rejected,
        "fallback_tokens": input_count if method == "text_fallback" else 0,
        "input_token_chars": input_chars,
        "keep_token_chars": kept_chars,
        "reject_token_chars": rejected_chars,
        "fallback": method == "text_fallback",
        "unaccounted_tokens": (
            input_count - kept - rejected if method == "timestamp" else 0
        ),
        "accounting_applicable": method == "timestamp",
        "boundary_rejections": boundary_counts,
        "boundary_rejection_detail": rejection_detail,\n        "boundary_rejection_correspondence": correspondence_detail if method == "timestamp" else [],
    }
