#!/usr/bin/env python3
"""Run an existing ASR CLI over a prepared manifest and emit metric-ready TSV.

The command must contain {audio}. Stdout is treated as the transcript.
Use --stdout-mode last-line for CLIs that print progress before the transcript.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--command", required=True, help="ASR command template containing {audio}")
    p.add_argument("--output-tsv", type=Path, required=True)
    p.add_argument("--report-json", type=Path, default=None)
    p.add_argument("--stdout-mode", choices=("raw", "last-line"), default="raw")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()

    if "{audio}" not in args.command:
        p.error("--command must contain the {audio} placeholder")
    if args.timeout <= 0:
        p.error("--timeout must be > 0")
    if not args.manifest.is_file():
        p.error(f"manifest not found: {args.manifest}")

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = data.get("manifest")
    if not isinstance(entries, list):
        p.error("manifest JSON must contain a manifest array")
    root = args.manifest.parent
    template = shlex.split(args.command)
    results, failures = [], []

    for item in entries:
        case_id = str(item.get("case_id", "<unknown>"))
        audio = (root / item["audio"]).resolve()
        if not audio.is_file():
            failures.append({"case_id": case_id, "error": "audio file missing"})
            if not args.continue_on_error:
                break
            continue
        command = [part.replace("{audio}", str(audio)) for part in template]
        try:
            proc = subprocess.run(command, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=args.timeout, check=False)
            if proc.returncode != 0:
                failures.append({"case_id": case_id, "error": f"exit={proc.returncode}",
                                 "stderr": proc.stderr[-2000:]})
                if not args.continue_on_error:
                    break
                continue
            stdout = proc.stdout.strip()
            if args.stdout_mode == "last-line":
                lines = [line.strip() for line in stdout.splitlines() if line.strip()]
                hypothesis = lines[-1] if lines else ""
            else:
                hypothesis = stdout
            results.append((item, hypothesis))
        except subprocess.TimeoutExpired:
            failures.append({"case_id": case_id, "error": "timeout"})
            if not args.continue_on_error:
                break

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["case_id", "category", "reference", "hypothesis"])
        for item, hypothesis in results:
            writer.writerow([item["case_id"], item.get("category", ""),
                             item["reference"], hypothesis])

    report = {"total": len(entries), "completed": len(results),
              "failure_count": len(failures), "failures": failures,
              "stdout_mode": args.stdout_mode, "timeout_seconds": args.timeout,
              "command_template": args.command, "output_tsv": str(args.output_tsv)}
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
