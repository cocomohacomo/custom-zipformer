#!/usr/bin/env python3
"""Create a shareable ASR report containing aggregate-only metrics.

Drops transcript text, audio references, per-utterance IDs, and detailed case rows.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CATEGORY_NAMES = {"short", "medium", "long", "uncategorized"}
COUNT_KEYS = {
    "case_count", "substitutions", "deletions", "insertions",
    "reference_units", "hypothesis_units",
}
RATE_KEYS = {"cer", "deletion_rate", "insertion_rate", "substitution_rate"}
CATEGORY_METRIC_KEYS = tuple(sorted(COUNT_KEYS | RATE_KEYS))

METRIC_FILES = {
    "whole": "metrics.json",
    "fixed_15s": "metrics_chunked_15s.json",
    "silence_aware_15s": "metrics_silence_aware_15s.json",
    "overlap_2s": "metrics_overlap_2s.json",
}
ALLOWED_MODES = set(METRIC_FILES) | {"rms_normalized"}


def _number(value: object, *, integer: bool = False, nonnegative: bool = False):
    """Return a finite numeric scalar, never arbitrary text or booleans."""
    if isinstance(value, bool):
        return None
    if integer:
        if not isinstance(value, int):
            return None
        result = value
    else:
        if not isinstance(value, (int, float)):
            return None
        result = value
    if (isinstance(result, float) and not math.isfinite(result)) or (nonnegative and result < 0):
        return None
    if isinstance(result, int) and abs(result) > 10**12:
        return None
    return result


def _metric_value(key: str, value: object):
    if key in COUNT_KEYS:
        return _number(value, integer=True, nonnegative=True)
    if key in RATE_KEYS:
        result = _number(value)
        if result is None or result < 0:
            return None
        if key != "cer" and result > 1:
            return None
        return result
    return None


def sanitize_metrics(report: dict) -> dict:
    if not isinstance(report, dict):
        report = {}
    summary = report.get("summary")
    summary = summary.get("char") if isinstance(summary, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    raw_categories = report.get("category_summary")
    if not isinstance(raw_categories, dict):
        raw_categories = {}

    categories = {}
    for name in sorted(CATEGORY_NAMES.intersection(raw_categories)):
        raw = raw_categories.get(name)
        if isinstance(raw, dict):
            categories[name] = {
                key: _metric_value(key, raw.get(key))
                for key in CATEGORY_METRIC_KEYS
            }

    def nested_count(section: str, key: str):
        value = report.get(section)
        return _number(value.get(key), integer=True, nonnegative=True) if isinstance(value, dict) else None

    return {
        "case_count": _number(report.get("case_count"), integer=True, nonnegative=True),
        "character_error_rate": _metric_value("cer", summary.get("cer")),
        "character_deletion_rate": _metric_value("deletion_rate", summary.get("deletion_rate")),
        "character_insertion_rate": _metric_value("insertion_rate", summary.get("insertion_rate")),
        "character_substitution_rate": _metric_value("substitution_rate", summary.get("substitution_rate")),
        "substitutions": _metric_value("substitutions", summary.get("substitutions")),
        "deletions": _metric_value("deletions", summary.get("deletions")),
        "insertions": _metric_value("insertions", summary.get("insertions")),
        "complete_omission_count": nested_count("complete_omission", "empty_hypothesis_count"),
        "near_complete_omission_count": nested_count("near_complete_omission", "count"),
        "severe_deletion_count": nested_count("severe_deletion", "count"),
        "category_summary": categories,
    }


def sanitize_comparisons(comparison: dict) -> dict:
    if not isinstance(comparison, dict):
        return {}
    raw = comparison.get("paired_comparisons")
    if not isinstance(raw, dict):
        return {}
    paired = {}
    allowed_modes = ALLOWED_MODES - {"whole"}
    for mode in sorted(allowed_modes.intersection(raw)):
        item = raw.get(mode)
        if not isinstance(item, dict):
            continue
        baseline = item.get("baseline")
        paired[mode] = {
            "baseline": baseline if isinstance(baseline, str) and baseline in ALLOWED_MODES else None,
            "deletion_delta": _number(item.get("deletion_delta"), integer=True),
            "cer_delta": _number(item.get("cer_delta")),
            "cases_with_fewer_deletions": _number(item.get("cases_with_fewer_deletions"), integer=True, nonnegative=True),
            "cases_with_more_deletions": _number(item.get("cases_with_more_deletions"), integer=True, nonnegative=True),
            "cases_unchanged": _number(item.get("cases_unchanged"), integer=True, nonnegative=True),
        }
    return paired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--comparison", default="mode_comparison.json")
    parser.add_argument(
        "--report", action="append", default=[], metavar="NAME=PATH",
        help="Explicit metric report mapping; when supplied, all four core modes are required.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extra-report", action="append", default=[], metavar="NAME=PATH",
        help="Optional aggregate report inside --root; repeat as needed.",
    )
    args = parser.parse_args()
    aggregate = {}
    report_files = dict(METRIC_FILES)
    if args.report:
        report_files = {}
        for value in args.report:
            if "=" not in value:
                parser.error(f"Invalid --report {value!r}; expected NAME=PATH")
            name, filename = value.split("=", 1)
            if name not in ALLOWED_MODES or not filename or name in report_files:
                parser.error("Unsupported, empty, or duplicate --report mapping")
            report_files[name] = filename
        missing = set(METRIC_FILES) - set(report_files)
        if missing:
            parser.error("Missing required report modes: " + ", ".join(sorted(missing)))
    for mode, filename in report_files.items():
        report = json.loads((args.root / filename).read_text(encoding="utf-8"))
        aggregate[mode] = sanitize_metrics(report)
    for value in args.extra_report:
        if "=" not in value:
            parser.error(f"Invalid --extra-report {value!r}; expected NAME=PATH")
        name, filename = value.split("=", 1)
        if name not in ({"rms_normalized"} - set(aggregate)) or not filename:
            parser.error("Unsupported or duplicate --extra-report name")
        report = json.loads((args.root / filename).read_text(encoding="utf-8"))
        aggregate[name] = sanitize_metrics(report)
    comparison = json.loads((args.root / args.comparison).read_text(encoding="utf-8"))
    output = {
        "privacy_policy": "aggregate metrics only; no audio, transcripts, manifests, or utterance identifiers",
        "modes": aggregate,
        "paired_comparisons": sanitize_comparisons(comparison),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mode_count": len(aggregate)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
