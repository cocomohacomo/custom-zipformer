#!/usr/bin/env python3
"""Create a shareable ASR report containing aggregate-only metrics.

Drops transcript text, audio references, per-utterance IDs, and detailed case rows.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

METRIC_FILES = {
    "whole": "metrics.json",
    "fixed_15s": "metrics_chunked_15s.json",
    "silence_aware_15s": "metrics_silence_aware_15s.json",
    "overlap_2s": "metrics_overlap_2s.json",
}

def sanitize_metrics(report: dict) -> dict:
    summary = report.get("summary", {}).get("char", {})
    return {
        "case_count": report.get("case_count"),
        "character_error_rate": summary.get("cer"),
        "character_deletion_rate": summary.get("deletion_rate"),
        "character_insertion_rate": summary.get("insertion_rate"),
        "character_substitution_rate": summary.get("substitution_rate"),
        "substitutions": summary.get("substitutions"),
        "deletions": summary.get("deletions"),
        "insertions": summary.get("insertions"),
        "complete_omission_count": report.get("complete_omission", {}).get("empty_hypothesis_count"),
        "near_complete_omission_count": report.get("near_complete_omission", {}).get("count"),
        "severe_deletion_count": report.get("severe_deletion", {}).get("count"),
        "category_summary": report.get("category_summary", {}),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--comparison", default="mode_comparison.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = {}
    for mode, filename in METRIC_FILES.items():
        report = json.loads((args.root / filename).read_text(encoding="utf-8"))
        aggregate[mode] = sanitize_metrics(report)
    comparison = json.loads((args.root / args.comparison).read_text(encoding="utf-8"))
    paired = {}
    for mode, item in comparison.get("paired_comparisons", {}).items():
        paired[mode] = {
            key: item.get(key) for key in (
                "baseline", "deletion_delta", "cer_delta",
                "cases_with_fewer_deletions", "cases_with_more_deletions",
                "cases_unchanged"
            )
        }
    output = {
        "privacy_policy": "aggregate metrics only; no audio, transcripts, manifests, or utterance identifiers",
        "modes": aggregate,
        "paired_comparisons": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mode_count": len(aggregate)}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
