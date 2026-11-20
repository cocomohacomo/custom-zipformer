#!/usr/bin/env python3
"""Compute Japanese ASR omission metrics from reference/hypothesis TSV.

CER is computed over Unicode code points without text normalization. Japanese
whitespace-token error is reported only as a diagnostic, never as WER.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence


def edit_counts(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) using deterministic DP."""
    n, m = len(ref), len(hyp)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            prev = dp[i - 1][j - 1]
            sub = (prev[0] + 1, prev[1] + 1, prev[2], prev[3])
            prev = dp[i - 1][j]
            delete = (prev[0] + 1, prev[1], prev[2] + 1, prev[3])
            prev = dp[i][j - 1]
            insert = (prev[0] + 1, prev[1], prev[2], prev[3] + 1)
            dp[i][j] = min(sub, delete, insert, key=lambda x: (x[0], x[1], x[2], x[3]))
    _, s, d, ins = dp[n][m]
    return s, d, ins


def ratio(num: int | float, denom: int | float) -> float | None:
    return num / denom if denom else None


def summarize(items: list[dict]) -> dict:
    s = d = ins = ref_units = hyp_units = 0
    for item in items:
        m = item["char"]
        s += m["substitutions"]
        d += m["deletions"]
        ins += m["insertions"]
        ref_units += m["reference_units"]
        hyp_units += m["hypothesis_units"]
    return {
        "case_count": len(items),
        "substitutions": s, "deletions": d, "insertions": ins,
        "reference_units": ref_units, "hypothesis_units": hyp_units,
        "cer": ratio(s + d + ins, ref_units),
        "deletion_rate": ratio(d, ref_units),
        "insertion_rate": ratio(ins, ref_units),
        "substitution_rate": ratio(s, ref_units),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tsv", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--near-complete-ratio", type=float, default=0.10,
                        help="Flag non-empty hypotheses no longer than this fraction of reference characters (default: 0.10)")
    parser.add_argument("--severe-deletion-rate", type=float, default=0.50,
                        help="Flag cases with character deletion rate at or above this threshold (default: 0.50)")
    args = parser.parse_args()
    if not 0 <= args.near_complete_ratio <= 1:
        parser.error("--near-complete-ratio must be between 0 and 1")
    if not 0 <= args.severe_deletion_rate <= 1:
        parser.error("--severe-deletion-rate must be between 0 and 1")

    with args.input_tsv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"case_id", "category", "reference", "hypothesis"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit("TSV header must contain: case_id, category, reference, hypothesis")
        rows = list(reader)

    per_case = []
    token_total = {"sub": 0, "del": 0, "ins": 0, "ref": 0, "hyp": 0}
    for row in rows:
        ref = row["reference"]
        hyp = row["hypothesis"]
        cs, cd, ci = edit_counts(list(ref), list(hyp))
        ref_chars, hyp_chars = len(ref), len(hyp)
        ref_tokens, hyp_tokens = ref.split(), hyp.split()
        ts, td, ti = edit_counts(ref_tokens, hyp_tokens)
        token_total["sub"] += ts
        token_total["del"] += td
        token_total["ins"] += ti
        token_total["ref"] += len(ref_tokens)
        token_total["hyp"] += len(hyp_tokens)
        deletion_rate = ratio(cd, ref_chars)
        char_ratio = ratio(hyp_chars, ref_chars)
        complete = not hyp.strip()
        near_complete = bool(ref.strip()) and not complete and char_ratio is not None and char_ratio <= args.near_complete_ratio
        item = {
            "case_id": row["case_id"], "category": row["category"],
            "complete_omission": complete,
            "near_complete_omission": near_complete,
            "hypothesis_to_reference_char_ratio": char_ratio,
            "severe_deletion": deletion_rate is not None and deletion_rate >= args.severe_deletion_rate,
            "char": {
                "substitutions": cs, "deletions": cd, "insertions": ci,
                "reference_units": ref_chars, "hypothesis_units": hyp_chars,
                "cer": ratio(cs + cd + ci, ref_chars),
                "deletion_rate": deletion_rate,
            },
        }
        for field in ("audio_duration_seconds", "audio_rms", "audio_peak",
                      "near_silence_fraction", "clipping_fraction",
                      "source_sample_rate", "source_channels", "applied_gain_db",
                      "inference_audio_rms"):
            if row.get(field, "").strip():
                try:
                    item[field] = float(row[field]) if field in {
                        "audio_duration_seconds", "audio_rms", "audio_peak",
                        "near_silence_fraction", "clipping_fraction",
                        "applied_gain_db", "inference_audio_rms"
                    } else int(row[field])
                except ValueError:
                    item[field] = row[field]
        per_case.append(item)

    char_summary = summarize(per_case)
    categories = defaultdict(list)
    for item in per_case:
        categories[item["category"] or "uncategorized"].append(item)
    category_summary = {name: summarize(items) for name, items in sorted(categories.items())}

    empty_hyp = sum(x["complete_omission"] for x in per_case)
    near_hyp = sum(x["near_complete_omission"] for x in per_case)
    severe = sum(x["severe_deletion"] for x in per_case)
    token_errors = token_total["sub"] + token_total["del"] + token_total["ins"]
    output = {
        "case_count": len(rows),
        "metric_policy": {
            "character_unit": "Unicode code point",
            "normalization": "none; raw reference and hypothesis strings",
            "japanese_word_metric": "not reported; whitespace-token diagnostic is not a valid Japanese WER",
            "near_complete_omission_ratio": args.near_complete_ratio,
            "severe_deletion_rate_threshold": args.severe_deletion_rate,
        },
        "complete_omission": {
            "empty_hypothesis_count": empty_hyp,
            "empty_hypothesis_rate": ratio(empty_hyp, len(rows)),
            "empty_reference_count": sum(not x["reference"].strip() for x in rows),
        },
        "near_complete_omission": {
            "count": near_hyp, "rate": ratio(near_hyp, len(rows)),
            "definition": "non-empty hypothesis character count / reference character count <= configured threshold",
        },
        "severe_deletion": {
            "count": severe, "rate": ratio(severe, len(rows)),
            "threshold": args.severe_deletion_rate,
        },
        "summary": {"char": char_summary},
        "category_summary": category_summary,
        "whitespace_token_diagnostic": {
            "substitutions": token_total["sub"], "deletions": token_total["del"],
            "insertions": token_total["ins"], "reference_tokens": token_total["ref"],
            "hypothesis_tokens": token_total["hyp"],
            "error_rate": ratio(token_errors, token_total["ref"]),
            "warning": "Not WER: Japanese text is generally not whitespace-tokenized.",
        },
        "cases": per_case,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
