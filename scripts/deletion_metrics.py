#!/usr/bin/env python3
"""Compute deletion-focused baseline metrics from reference/hypothesis TSV.

Input TSV format (header required):
case_id<TAB>category<TAB>reference<TAB>hypothesis
Character metrics are calculated on raw strings; word metrics use whitespace
splitting. Normalize transcripts upstream only if the normalization policy is
recorded and identical for all runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


def edit_counts(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) using deterministic DP."""
    n, m = len(ref), len(hyp)
    # Each cell: (cost, substitutions, deletions, insertions)
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


def ratio(num: int, denom: int) -> float | None:
    return num / denom if denom else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tsv", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    with args.input_tsv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\\t")
        required = {"case_id", "category", "reference", "hypothesis"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit("TSV header must contain: case_id, category, reference, hypothesis")
        rows = list(reader)

    total = {"char": {"sub": 0, "del": 0, "ins": 0, "ref": 0, "hyp": 0},
             "word": {"sub": 0, "del": 0, "ins": 0, "ref": 0, "hyp": 0}}
    per_case = []
    for row in rows:
        ref_chars, hyp_chars = list(row["reference"]), list(row["hypothesis"])
        ref_words, hyp_words = row["reference"].split(), row["hypothesis"].split()
        cs, cd, ci = edit_counts(ref_chars, hyp_chars)
        ws, wd, wi = edit_counts(ref_words, hyp_words)
        for kind, (s, d, ins, nr, nh) in {
            "char": (cs, cd, ci, len(ref_chars), len(hyp_chars)),
            "word": (ws, wd, wi, len(ref_words), len(hyp_words)),
        }.items():
            total[kind]["sub"] += s
            total[kind]["del"] += d
            total[kind]["ins"] += ins
            total[kind]["ref"] += nr
            total[kind]["hyp"] += nh
        per_case.append({
            "case_id": row["case_id"], "category": row["category"],
            "char": {"substitutions": cs, "deletions": cd, "insertions": ci,
                     "reference_units": len(ref_chars), "cer": ratio(cs + cd + ci, len(ref_chars)),
                     "deletion_rate": ratio(cd, len(ref_chars))},
            "word": {"substitutions": ws, "deletions": wd, "insertions": wi,
                     "reference_units": len(ref_words), "wer": ratio(ws + wd + wi, len(ref_words)),
                     "deletion_rate": ratio(wd, len(ref_words))},
        })

    summary = {}
    for kind, v in total.items():
        errors = v["sub"] + v["del"] + v["ins"]
        summary[kind] = {
            "substitutions": v["sub"], "deletions": v["del"], "insertions": v["ins"],
            "reference_units": v["ref"], "hypothesis_units": v["hyp"],
            "error_rate": ratio(errors, v["ref"]),
            "deletion_rate": ratio(v["del"], v["ref"]),
            "insertion_rate": ratio(v["ins"], v["ref"]),
            "substitution_rate": ratio(v["sub"], v["ref"]),
        }
    output = {"case_count": len(rows), "summary": summary, "cases": per_case}
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.write_text(rendered + "\\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
