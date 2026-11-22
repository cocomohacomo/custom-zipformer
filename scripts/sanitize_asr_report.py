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
COMPARISON_MODE_ALIASES = {"silence_aware": "silence_aware_15s"}


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
    for source_mode, item in sorted(raw.items()):
        mode = COMPARISON_MODE_ALIASES.get(source_mode, source_mode)
        if mode not in allowed_modes or not isinstance(item, dict) or mode in paired:
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



def sanitize_execution_diagnostics(report: dict) -> dict:
    """Allowlist aggregate-only runtime diagnostics; never copy arbitrary input keys."""
    if not isinstance(report, dict) or not isinstance(report.get("modes"), dict):
        return {}
    output = {}
    for mode in sorted(ALLOWED_MODES.intersection(report["modes"])):
        item = report["modes"].get(mode)
        if not isinstance(item, dict):
            continue
        duration = item.get("audio_duration_seconds")
        chunking = item.get("chunking")
        if not isinstance(duration, dict) or not isinstance(chunking, dict):
            continue
        clean_duration = {
            key: _number(duration.get(key))
            for key in ("min", "median", "max")
        }
        clean_chunking = {
            key: _number(chunking.get(key), integer=(key != "multi_chunk_case_rate"), nonnegative=True)
            for key in (
                "single_chunk_cases", "multi_chunk_cases", "multi_chunk_case_rate",
                "total_chunks", "max_chunks_per_case", "short_chunks_lt_25ms", "silent_chunks",
                "empty_output_chunks",
            )
        }
        empty_rate = _number(chunking.get("empty_output_chunk_rate"))
        if empty_rate is None or not 0 <= empty_rate <= 1:
            continue
        clean_chunking["empty_output_chunk_rate"] = empty_rate
        if any(value is None for value in clean_duration.values()) or any(
            value is None for value in clean_chunking.values()
        ):
            continue
        clean = {
            "case_count": _number(item.get("case_count"), integer=True, nonnegative=True),
            "audio_duration_seconds": clean_duration,
            "chunking": clean_chunking,
        }
        methods = item.get("overlap_merge_methods")
        if isinstance(methods, dict):
            clean["overlap_merge_methods"] = {
                key: _number(methods.get(key), integer=True, nonnegative=True)
                for key in ("timestamp", "text_fallback")
                if _number(methods.get(key), integer=True, nonnegative=True) is not None
            }
        stages = item.get("decoder_stages")
        if isinstance(stages, dict):
            stage_count_keys = (
                "total_raw_tokens", "total_timestamps", "total_raw_decoder_text_chars",
                "total_chunk_text_chars", "total_merged_text_chars", "total_final_text_chars",
                "cases_shorter_after_merge", "cases_shorter_after_normalization",
                "cases_token_timestamp_count_mismatch",
            )
            clean_stages = {
                key: _number(stages.get(key), integer=True, nonnegative=True)
                for key in stage_count_keys
            }
            clean_stages["chunk_merge_char_delta"] = _number(
                stages.get("chunk_merge_char_delta"), integer=True
            )
            clean_stages["final_normalization_char_delta"] = _number(
                stages.get("final_normalization_char_delta"), integer=True
            )
            clean_stages["mean_token_density_per_second"] = _number(
                stages.get("mean_token_density_per_second"), nonnegative=True
            )
            clean_stages["total_decode_seconds"] = _number(
                stages.get("total_decode_seconds"), nonnegative=True
            )
            if all(value is not None for value in clean_stages.values()):
                clean["decoder_stages"] = clean_stages
        accounting = item.get("merge_accounting")
        if mode == "overlap_2s" and isinstance(accounting, dict):
            keys = ("input_tokens", "keep_tokens", "reject_tokens", "fallback_tokens", "input_token_chars", "keep_token_chars", "reject_token_chars", "timestamp_cases", "fallback_cases", "unaccounted_tokens")
            values = {key: _number(accounting.get(key), integer=True, nonnegative=True) for key in keys}
            raw_boundaries = accounting.get("boundary_rejections_by_chunk")
            clean_boundaries = {}
            if isinstance(raw_boundaries, dict):
                for name, value in sorted(raw_boundaries.items()):
                    if isinstance(name, str) and name.startswith("chunk_") and name[6:].isdigit() and isinstance(value, dict):
                        tokens = _number(value.get("rejected_tokens"), integer=True, nonnegative=True)
                        chars = _number(value.get("rejected_token_chars"), integer=True, nonnegative=True)
                        if tokens is not None and chars is not None:
                            clean_boundaries[name] = {"rejected_tokens": tokens, "rejected_token_chars": chars}
            if all(value is not None for value in values.values()) and accounting.get("accounting_consistent") is True and values["unaccounted_tokens"] == 0:
                clean["merge_accounting"] = {**values, "accounting_consistent": True, "boundary_rejections_by_chunk": clean_boundaries}
                raw_detail = accounting.get("boundary_rejections_by_side_and_distance")
                clean_detail = {}
                if isinstance(raw_detail, dict):
                    allowed_detail_keys = {
                        side + "|" + distance
                        for side in ("after_seam_from_left_chunk", "before_seam_from_right_chunk")
                        for distance in ("lt_100ms", "100_250ms", "250_500ms", "gte_500ms")
                    }
                    for key, value in sorted(raw_detail.items()):
                        if key in allowed_detail_keys and isinstance(value, dict):
                            tokens = _number(value.get("rejected_tokens"), integer=True, nonnegative=True)
                            chars = _number(value.get("rejected_token_chars"), integer=True, nonnegative=True)
                            if tokens is not None and chars is not None:
                                clean_detail[key] = {"rejected_tokens": tokens, "rejected_token_chars": chars}
                clean["merge_accounting"]["boundary_rejections_by_side_and_distance"] = clean_detail
                raw_correspondence = accounting.get("boundary_rejection_correspondence")
                clean_correspondence = {}
                allowed_statuses = ("matched_unique", "matched_ambiguous", "unmatched_no_candidate", "unmatched_conflict")
                if isinstance(raw_correspondence, dict):
                    for key, value in sorted(raw_correspondence.items()):
                        parts = key.split("|") if isinstance(key, str) else []
                        if (len(parts) == 3
                                and parts[0] in ("after_seam_from_left_chunk", "before_seam_from_right_chunk")
                                and parts[1] in ("lt_100ms", "100_250ms", "250_500ms", "gte_500ms")
                                and parts[2] in allowed_statuses and isinstance(value, dict)):
                            tokens = _number(value.get("rejected_tokens"), integer=True, nonnegative=True)
                            chars = _number(value.get("rejected_token_chars"), integer=True, nonnegative=True)
                            if tokens is not None and chars is not None:
                                clean_correspondence[key] = {"rejected_tokens": tokens, "rejected_token_chars": chars}
                clean["merge_accounting"]["boundary_rejection_correspondence"] = clean_correspondence
        output[mode] = clean
    return output

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--comparison", default="mode_comparison.json")
    parser.add_argument("--diagnostics", default=None, help="Optional aggregate-only execution diagnostics JSON")
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
    if args.diagnostics:
        diagnostics = json.loads((args.root / args.diagnostics).read_text(encoding="utf-8"))
        output["execution_diagnostics"] = sanitize_execution_diagnostics(diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mode_count": len(aggregate)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
