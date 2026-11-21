#!/usr/bin/env python3
"""Compare per-case omission metrics across paired ASR inference modes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_report(label: str, path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{label}: metrics report has no cases array")
    indexed = {}
    for item in cases:
        case_id = item.get("case_id")
        if not case_id or case_id in indexed:
            raise ValueError(f"{label}: missing or duplicate case_id: {case_id!r}")
        char = item.get("char")
        if not isinstance(char, dict) or "deletions" not in char:
            raise ValueError(f"{label}: missing character deletion count for {case_id}")
        indexed[case_id] = item
    return {"raw": report, "cases": indexed}


def compare_reports(named_paths: list[tuple[str, Path]]) -> dict:
    if len(named_paths) < 2:
        raise ValueError("at least two named metrics reports are required")
    loaded = [(name, load_report(name, path)) for name, path in named_paths]
    baseline_name, baseline = loaded[0]
    baseline_ids = set(baseline["cases"])
    modes = {}
    for name, data in loaded:
        if set(data["cases"]) != baseline_ids:
            missing = sorted(baseline_ids - set(data["cases"]))
            extra = sorted(set(data["cases"]) - baseline_ids)
            raise ValueError(f"{name}: case set mismatch; missing={missing[:5]}, extra={extra[:5]}")
        rows = list(data["cases"].values())
        summary = data["raw"].get("summary", {}).get("char", {})
        modes[name] = {
            "case_count": len(rows),
            "cer": summary.get("cer"),
            "deletion_rate": summary.get("deletion_rate"),
            "deletions": sum(int(row["char"]["deletions"]) for row in rows),
            "complete_omissions": sum(bool(row.get("complete_omission")) for row in rows),
            "near_complete_omissions": sum(bool(row.get("near_complete_omission")) for row in rows),
            "severe_deletions": sum(bool(row.get("severe_deletion")) for row in rows),
            "timestamp_merge_cases": sum(row.get("overlap_merge_method") == "timestamp" for row in rows),
            "text_fallback_cases": sum(row.get("overlap_merge_method") == "text_fallback" for row in rows),
        }

    baseline_cases = baseline["cases"]
    comparisons = {}
    for name, data in loaded[1:]:
        improved = regressed = unchanged = 0
        improved_ids, regressed_ids = [], []
        for case_id, base_case in baseline_cases.items():
            base_del = int(base_case["char"]["deletions"])
            candidate_del = int(data["cases"][case_id]["char"]["deletions"])
            if candidate_del < base_del:
                improved += 1
                improved_ids.append(case_id)
            elif candidate_del > base_del:
                regressed += 1
                regressed_ids.append(case_id)
            else:
                unchanged += 1
        comparisons[name] = {
            "baseline": baseline_name,
            "deletion_delta": modes[name]["deletions"] - modes[baseline_name]["deletions"],
            "cer_delta": (
                modes[name]["cer"] - modes[baseline_name]["cer"]
                if modes[name]["cer"] is not None and modes[baseline_name]["cer"] is not None
                else None
            ),
            "cases_with_fewer_deletions": improved,
            "cases_with_more_deletions": regressed,
            "cases_unchanged": unchanged,
            "improved_case_ids": improved_ids,
            "regressed_case_ids": regressed_ids,
        }
    return {
        "baseline_mode": baseline_name,
        "modes": modes,
        "paired_comparisons": comparisons,
        "selection_policy": (
            "Do not select a mode by CER alone. Review deletion totals, severe/near-complete "
            "omissions, and per-case regressions together."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", required=True, metavar="NAME=PATH",
        help="Paired metrics report; first report is the baseline. Repeat at least twice.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    named_paths = []
    for value in args.report:
        if "=" not in value:
            parser.error(f"Invalid --report {value!r}; expected NAME=PATH")
        name, filename = value.split("=", 1)
        if not name or not filename:
            parser.error(f"Invalid --report {value!r}; expected NAME=PATH")
        named_paths.append((name, Path(filename)))
    try:
        result = compare_reports(named_paths)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
