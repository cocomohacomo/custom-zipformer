#!/usr/bin/env python3
"""Validate paired ASR evaluation outputs and print aggregate-only diagnostics.

Empty hypotheses are valid evaluation outcomes: they are counted as omissions,
not treated as infrastructure failures. This keeps quality measurement separate
from execution integrity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
            rows = list(csv.DictReader(stream, delimiter="\t"))
    except OSError as exc:
        raise ValueError("Required inference output is missing") from exc
    if not rows or not {"case_id", "hypothesis"}.issubset(rows[0]):
        raise ValueError("Inference output is empty or missing required columns")
    return rows


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
    for mode, (transcript_name, metrics_name) in MODES.items():
        rows = _read_tsv(root / transcript_name)
        actual = [row.get("case_id") for row in rows]
        if actual != expected:
            raise ValueError(f"{mode}: inference row count/order does not match the manifest")
        empty_count = sum(not (row.get("hypothesis") or "").strip() for row in rows)

        report = _read_json(root / metrics_name)
        if report.get("case_count") != len(expected):
            raise ValueError(f"{mode}: metric case count does not match the manifest")
        summary = report.get("summary")
        char = summary.get("char") if isinstance(summary, dict) else None
        if not isinstance(char, dict):
            raise ValueError(f"{mode}: character metrics are missing")
        for section, field in (
            ("complete_omission", "empty_hypothesis_count"),
            ("near_complete_omission", "count"),
            ("severe_deletion", "count"),
        ):
            values = report.get(section)
            if not isinstance(values, dict) or not isinstance(values.get(field), int):
                raise ValueError(f"{mode}: required omission metrics are missing")
        metric_cases = report.get("cases")
        if isinstance(metric_cases, list) and len(metric_cases) != len(expected):
            raise ValueError(f"{mode}: per-case metric count does not match the manifest")

        result = {
            "case_count": len(expected),
            "empty_hypotheses": empty_count,
            "cer": char.get("cer"),
            "deletion_rate": char.get("deletion_rate"),
            "complete_omissions": report["complete_omission"]["empty_hypothesis_count"],
            "near_complete_omissions": report["near_complete_omission"]["count"],
            "severe_deletions": report["severe_deletion"]["count"],
        }

        if mode == "overlap_2s":
            for row in rows:
                try:
                    chunk_count = int(row["inference_chunks"])
                    starts = [float(value) for value in row["chunk_start_offsets_seconds"].split(",") if value]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("Overlap chunk diagnostics are missing or invalid") from exc
                if len(starts) != chunk_count or not starts or starts[0] != 0.0:
                    raise ValueError("Overlap chunk offsets do not match the chunk count")
                if any(not math.isfinite(value) for value in starts):
                    raise ValueError("Overlap chunk offsets contain non-finite values")
                if any(right <= left for left, right in zip(starts, starts[1:])):
                    raise ValueError("Overlap chunk offsets are not strictly increasing")
                if row.get("overlap_merge_method") not in {"timestamp", "text_fallback"}:
                    raise ValueError("Overlap merge diagnostics are missing")
            methods: dict[str, int] = {}
            for row in rows:
                method = row["overlap_merge_method"]
                methods[method] = methods.get(method, 0) + 1
            result["overlap_merge_methods"] = methods
        results[mode] = result
    return {"case_count": len(expected), "modes": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_evaluation(args.root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
