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


def _stage_diagnostics(rows: list[dict[str, str]]) -> dict:
    """Aggregate stage boundaries and signed character deltas without text content."""
    required = (
        "raw_token_count", "timestamp_count", "raw_decoder_text_chars",
        "chunk_text_chars", "merged_text_chars", "final_text_chars",
        "chunk_merge_char_delta", "final_normalization_char_delta",
        "token_density_per_second", "decode_seconds",
    )
    totals = {key: 0.0 for key in required}
    shorter_after_merge = 0
    shorter_after_normalization = 0
    timestamp_mismatch = 0
    for row in rows:
        for key in required:
            totals[key] += _finite_float(row.get(key), key, minimum=-1e12 if key.endswith("_delta") else 0.0)
        if float(row["merged_text_chars"]) < float(row["chunk_text_chars"]):
            shorter_after_merge += 1
        if float(row["final_text_chars"]) < float(row["merged_text_chars"]):
            shorter_after_normalization += 1
        if float(row["raw_token_count"]) != float(row["timestamp_count"]):
            timestamp_mismatch += 1
        if int(float(row["chunk_merge_char_delta"])) != (
            int(float(row["merged_text_chars"])) - int(float(row["chunk_text_chars"]))
        ):
            raise ValueError("chunk merge character delta is inconsistent")
        if int(float(row["final_normalization_char_delta"])) != (
            int(float(row["final_text_chars"])) - int(float(row["merged_text_chars"]))
        ):
            raise ValueError("final normalization character delta is inconsistent")
    return {
        "decoder_stages": {
            "total_raw_tokens": int(totals["raw_token_count"]),
            "total_timestamps": int(totals["timestamp_count"]),
            "total_raw_decoder_text_chars": int(totals["raw_decoder_text_chars"]),
            "total_chunk_text_chars": int(totals["chunk_text_chars"]),
            "total_merged_text_chars": int(totals["merged_text_chars"]),
            "total_final_text_chars": int(totals["final_text_chars"]),
            "chunk_merge_char_delta": int(totals["chunk_merge_char_delta"]),
            "final_normalization_char_delta": int(totals["final_normalization_char_delta"]),
            "cases_shorter_after_merge": shorter_after_merge,
            "cases_shorter_after_normalization": shorter_after_normalization,
            "cases_token_timestamp_count_mismatch": timestamp_mismatch,
            "mean_token_density_per_second": totals["token_density_per_second"] / len(rows),
            "total_decode_seconds": totals["decode_seconds"],
        }
    }


def _merge_accounting_diagnostics(rows: list[dict[str, str]]) -> dict:
    """Validate token accounting and emit aggregate-only overlap merge diagnostics."""
    totals = {key: 0 for key in ("input_tokens", "keep_tokens", "reject_tokens", "fallback_tokens", "input_token_chars", "keep_token_chars", "reject_token_chars")}
    methods = {"timestamp": 0, "text_fallback": 0}
    boundaries: dict[str, dict[str, int]] = {}
    boundary_detail: dict[str, dict[str, int]] = {}
    for row in rows:
        method = row.get("overlap_merge_method")
        if method not in methods:
            raise ValueError("overlap merge method is missing or invalid")
        methods[method] += 1
        names = {"input_tokens":"merge_input_tokens","keep_tokens":"merge_keep_tokens","reject_tokens":"merge_reject_tokens","fallback_tokens":"merge_fallback_tokens","unaccounted":"merge_unaccounted_tokens","input_token_chars":"merge_input_token_chars","keep_token_chars":"merge_keep_token_chars","reject_token_chars":"merge_reject_token_chars"}
        values = {}
        for key, column in names.items():
            raw = row.get(column)
            if not isinstance(raw, str) or not raw.isdigit():
                raise ValueError("merge accounting diagnostic is missing or invalid")
            values[key] = int(raw)
        if method == "timestamp":
            if values["fallback_tokens"] or values["unaccounted"] or values["input_tokens"] != values["keep_tokens"] + values["reject_tokens"]:
                raise ValueError("timestamp merge token accounting is inconsistent")
            if values["input_token_chars"] != values["keep_token_chars"] + values["reject_token_chars"]:
                raise ValueError("timestamp merge character accounting is inconsistent")
        elif values["keep_tokens"] or values["reject_tokens"] or values["fallback_tokens"] != values["input_tokens"] or values["unaccounted"]:
            raise ValueError("fallback tokens must remain in the independent fallback bucket")
        for key in totals:
            totals[key] += values[key]
        try:
            records = json.loads(row.get("merge_boundary_rejections", ""))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("merge boundary diagnostics are invalid") from exc
        if not isinstance(records, list):
            raise ValueError("merge boundary diagnostics must be a list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("merge boundary record is invalid")
            position, count, chars = record.get("chunk_position"), record.get("rejected_tokens"), record.get("rejected_token_chars")
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (position, count, chars)):
                raise ValueError("merge boundary counts are invalid")
            bucket = boundaries.setdefault("chunk_" + str(position), {"rejected_tokens": 0, "rejected_token_chars": 0})
            bucket["rejected_tokens"] += count
            bucket["rejected_token_chars"] += chars
        try:
            details = json.loads(row.get("merge_boundary_rejection_detail", ""))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("merge boundary rejection detail is invalid") from exc
        if not isinstance(details, list):
            raise ValueError("merge boundary rejection detail must be a list")
        for detail in details:
            if not isinstance(detail, dict):
                raise ValueError("merge boundary rejection detail record is invalid")
            side, distance = detail.get("side"), detail.get("distance_bucket")
            count, chars = detail.get("rejected_tokens"), detail.get("rejected_token_chars")
            if side not in ("after_seam_from_left_chunk", "before_seam_from_right_chunk"):
                raise ValueError("merge boundary rejection side is invalid")
            if distance not in ("lt_100ms", "100_250ms", "250_500ms", "gte_500ms"):
                raise ValueError("merge boundary rejection distance is invalid")
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (count, chars)):
                raise ValueError("merge boundary rejection detail counts are invalid")
            key = side + "|" + distance
            bucket = boundary_detail.setdefault(key, {"rejected_tokens": 0, "rejected_token_chars": 0})
            bucket["rejected_tokens"] += count
            bucket["rejected_token_chars"] += chars
    if totals["input_tokens"] != totals["keep_tokens"] + totals["reject_tokens"] + totals["fallback_tokens"]:
        raise ValueError("aggregate merge token accounting is inconsistent")
    return {"merge_accounting": {**totals, "timestamp_cases": methods["timestamp"], "fallback_cases": methods["text_fallback"], "unaccounted_tokens": 0, "accounting_consistent": True, "boundary_rejections_by_chunk": boundaries, "boundary_rejections_by_side_and_distance": boundary_detail}}


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
        result.update(_stage_diagnostics(rows))
        if mode == "overlap_2s":
            methods: dict[str, int] = {}
            for row in rows:
                method = row["overlap_merge_method"]
                methods[method] = methods.get(method, 0) + 1
            result["overlap_merge_methods"] = methods
            result.update(_merge_accounting_diagnostics(rows))
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
