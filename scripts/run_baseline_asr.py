#!/usr/bin/env python3
"""Run an existing ASR CLI over a prepared manifest and emit metric-ready TSV.

The command must contain {audio}; stdout is treated as the raw transcript.
Example: --command 'python recognizer.py --input {audio}'
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--command", required=True,
                   help="ASR command template; use {audio} as the clip path")
    p.add_argument("--output-tsv", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()

    if "{audio}" not in args.command:
        p.error("--command must contain the {audio} placeholder")
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent
    template = shlex.split(args.command)
    results = []
    failures = []

    for item in data.get("manifest", []):
        audio = (root / item["audio"]).resolve()
        if not audio.is_file():
            failures.append({"case_id": item["case_id"], "error": "audio file missing"})
            if not args.continue_on_error:
                break
            continue
        command = [part.replace("{audio}", str(audio)) for part in template]
        try:
            proc = subprocess.run(command, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=args.timeout, check=False)
            if proc.returncode != 0:
                failures.append({"case_id": item["case_id"],
                                 "error": f"exit={proc.returncode}",
                                 "stderr": proc.stderr[-2000:]})
                if not args.continue_on_error:
                    break
                continue
            hypothesis = proc.stdout.strip()
            results.append((item, hypothesis))
        except subprocess.TimeoutExpired:
            failures.append({"case_id": item["case_id"], "error": "timeout"})
            if not args.continue_on_error:
                break

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["case_id", "category", "reference", "hypothesis"])
        for item, hypothesis in results:
            writer.writerow([item["case_id"], item["category"],
                             item["reference"], hypothesis])

    report = {"total": len(data.get("manifest", [])), "completed": len(results),
              "failures": failures, "output_tsv": str(args.output_tsv)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
