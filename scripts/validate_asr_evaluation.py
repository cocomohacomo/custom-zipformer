#!/usr/bin/env python3
"""Validate paired ASR outputs and emit aggregate-only execution diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

MODES = {
    "whole": ("transcripts.tsv", "metrics.json"),
    "fixed_15s": ("transcripts_chunked_15s.tsv", "metrics_chunked_15s.json"),
    "silence_aware_15s": ("transcripts_silence_aware_15s.tsv", "metrics_silence_aware_15s.json"),
    "overlap_2s": ("transcripts_overlap_2s.tsv", "metrics_overlap_2s.json"),
}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Required evaluation report is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("Evaluation report must be a JSON object")
    return value


def _read_tsv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter=chr(9)))
    except OSError as exc:
        raise ValueError("Required inference output is missing") from exc
    if not rows or not {"case_id", "hypothesis"}.issubset(rows[0]):
        raise ValueError("Inference output is empty or missing required columns")
    return rows


def _finite_float(value: object, label: str, *, minimum: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} diagnostic is missing or invalid") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} diagnostic is non-finite or out of range")
    return result


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, str) or not value.isdigit():
        raise ValueError(f"{label} diagnostic is missing or invalid")
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} diagnostic is out of range")
    return result


def _audio_diagnostics(rows: list[dict[str, str]], mode: str) -> dict:
    durations = []
    chunk_counts = []
    short_chunks = 0
    silent_chunks = 0
    empty_chunks = 0
    for row in rows:
        duration = _finite_float(row.get("audio_duration_seconds"), "audio duration")
        chunks = _positive_int(row.get("inference_chunks"), "inference chunk count")
        short = row.get("inference_short_chunks_lt_25ms", "0")
        silent = row.get("inference_silent_chunks", "0")
        empty = row.get("inference_empty_chunks")
        if (not isinstance(short, str) or not short.isdigit()
                or not isinstance(silent, str) or not silent.isdigit()
                or not isinstance(empty, str) or not empty.isdigit()):
            raise ValueError("chunk quality diagnostics are missing or invalid")
        short_count, silent_count, empty_count = int(short), int(silent), int(empty)
        if short_count > chunks or silent_count > chunks or empty_count > chunks:
            raise ValueError("chunk quality diagnostics exceed the chunk count")
        durations.append(duration)
        chunk_counts.append(chunks)
        short_chunks += short_count
        silent_chunks += silent_count
        empty_chunks += empty_count

        if mode == "whole" and chunks != 1:
            raise ValueError("whole-utterance mode must decode exactly one chunk per case")
        if mode == "fixed_15s":
            expected = max(1, math.ceil(duration / 15.0 - 1e-12))
            if chunks != expected:
                raise ValueError("fixed chunk count does not match the configured 15-second split")
        if mode == "silence_aware_15s":
            raw = row.get("chunk_boundaries_seconds", "")
            boundaries = [] if not raw else [_finite_float(x, "silence boundary") for x in raw.split(",")]
            if len(boundaries) != chunks - 1:
                raise ValueError("silence-aware boundaries do not match the decoded chunk count")
            if any(x <= 0 or x >= duration for x in boundaries) or any(
                right <= left for left, right in zip(boundaries, boundaries[1:])
            ):
                raise ValueError("silence-aware boundaries are outside the audio or unordered")
        if mode == "overlap_2s":
            raw = row.get("chunk_start_offsets_seconds", "")
            starts = [] if not raw else [_finite_float(x, "overlap start") for x in raw.split(",")]
            if len(starts) != chunks or not starts or starts[0] != 0.0:
                raise ValueError("overlap chunk offsets do not match the decoded chunk count")
            if starts[-1] >= duration or any(right <= left for left, right in zip(starts, starts[1:])):
                raise ValueError("overlap chunk offsets are outside the audio or unordered")
            if row.get("overlap_merge_method") not in {"timestamp", "text_fallback"}:
                raise ValueError("Overlap merge diagnostics are missing")

    ordered = sorted(durations)
    return {
        "audio_duration_seconds": {
            "min": min(durations),
            "median": statistics.median(ordered),
            "max": max(durations),
        },
        "chunking": {
            "single_chunk_cases": sum(value == 1 for value in chunk_counts),
            "multi_chunk_cases": sum(value > 1 for value in chunk_counts),
            "multi_chunk_case_rate": sum(value > 1 for value in chunk_counts) / len(chunk_counts),
            "total_chunks": sum(chunk_counts),
            "max_chunks_per_case": max(chunk_counts),
            "short_chunks_lt_25ms": short_chunks,
            "silent_chunks": silent_chunks,
            "empty_output_chunks": empty_chunks,
            "empty_output_chunk_rate": empty_chunks / sum(chunk_counts),
        },
    }


def validate_evaluation(root: Path) -> dict:
    manifest = _read_json(root / "manifest.json")
    entries = manifest.get("manifest")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Evaluation manifest is empty or invalid")
    expected = [item.get("case_id") for item in entries if isinstance(item, dict)]
    if len(expected) != len(entries) or any(not isinstance(value, str) or not value for value in expected):
        raise ValueError("Evaluation manifest contains invalid case references")
    if len(set(expected)) != len(expected):
        raise ValueError("Evaluation manifest contains duplicate case references")

    results = {}
    reference_durations = None
    reference_texts = None
    for mode, (transcript_name, metrics_name) in MODES.items():
        rows = _read_tsv(root / transcript_name)
        actual = [row.get("case_id") for row in rows]
        if actual != expected:
            raise ValueError(f"{mode}: inference row count/order does not match the manifest")
        references = [row.get("reference", "") for row in rows]
        if any(not isinstance(value, str) for value in references):
            raise ValueError(f"{mode}: reference transcript is missing or invalid")
        if reference_texts is None:
            reference_texts = references
        elif references != reference_texts:
            raise ValueError(f"{mode}: reference transcripts differ from the baseline")
        durations = [_finite_float(row.get("audio_duration_seconds"), "audio duration") for row in rows]
        if reference_durations is None:
            reference_durations = durations
        elif any(abs(a - b) > 1e-5 for a, b in zip(reference_durations, durations)):
            raise ValueError(f"{mode}: paired audio durations do not match the baseline")
        empty_count = sum(not (row.get("hypothesis") or "").strip() for row in rows)

        report = _read_json(root / metrics_name)
        if report.get("case_count") != len(expected):
            raise ValueError(f"{mode}: metric case count does not match the manifest")
        summary = report.get("summary")
        char = summary.get("char") if isinstance(summary, dict) else None
        if not isinstance(char, dict):
            raise ValueError(f"{mode}: character metrics are missing")
        omission_counts = {}
        for section, field in (
            ("complete_omission", "empty_hypothesis_count"),
            ("near_complete_omission", "count"),
            ("severe_deletion", "count"),
        ):
            values = report.get(section)
            count = values.get(field) if isinstance(values, dict) else None
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(expected):
                raise ValueError(f"{mode}: required omission metrics are missing or out of range")
            omission_counts[section] = count
        if omission_counts["complete_omission"] != empty_count:
            raise ValueError(f"{mode}: empty-hypothesis and complete-omission counts disagree")

        cer = char.get("cer")
        deletion_rate = char.get("deletion_rate")
        if (isinstance(cer, bool) or not isinstance(cer, (int, float))
                or not math.isfinite(cer) or cer < 0):
            raise ValueError(f"{mode}: character error rate is invalid")
        if (isinstance(deletion_rate, bool) or not isinstance(deletion_rate, (int, float))
                or not math.isfinite(deletion_rate) or not 0 <= deletion_rate <= 1):
            raise ValueError(f"{mode}: character deletion rate is invalid")
        metric_cases = report.get("cases")
        if isinstance(metric_cases, list) and len(metric_cases) != len(expected):
            raise ValueError(f"{mode}: per-case metric count does not match the manifest")

        result = {
            "case_count": len(expected),
            "empty_hypotheses": empty_count,
            "cer": cer,
            "deletion_rate": deletion_rate,
            "complete_omissions": omission_counts["complete_omission"],
            "near_complete_omissions": omission_counts["near_complete_omission"],
            "severe_deletions": omission_counts["severe_deletion"],
        }
        result.update(_audio_diagnostics(rows, mode))
        if mode == "overlap_2s":
            methods: dict[str, int] = {}
            for row in rows:
                method = row["overlap_merge_method"]
                methods[method] = methods.get(method, 0) + 1
            result["overlap_merge_methods"] = methods
        results[mode] = result
    return {
        "case_count": len(expected),
        "diagnostic_policy": "aggregate only; no audio, transcripts, manifests, or utterance identifiers",
        "modes": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = validate_evaluation(args.root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + chr(10), encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
