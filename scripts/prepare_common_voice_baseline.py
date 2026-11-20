#!/usr/bin/env python3
"""Prepare a deterministic, local Common Voice Japanese baseline subset.

Download Common Voice Scripted Speech 27.0 Japanese from Mozilla Data Collective
through its official UI, accept its terms, and extract it locally. This tool
copies selected clips into a local test directory; it does not upload or
redistribute the corpus.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

SEED = 270021
HEADER = ["case_id", "category", "reference", "hypothesis"]


def classify(sentence: str) -> str:
    n = len(sentence.replace(" ", ""))
    if n <= 8:
        return "short"
    if n <= 24:
        return "medium"
    return "long"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus_dir", type=Path, help="Extracted Common Voice ja corpus root")
    p.add_argument("--split", choices=("test", "dev", "validated"), default="test")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--output", type=Path, default=Path("tests/data/common_voice_ja"))
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    if args.limit < 1:
        p.error("--limit must be >= 1")

    tsv = args.corpus_dir / f"{args.split}.tsv"
    if not tsv.is_file():
        p.error(f"Missing {tsv}; pass the extracted corpus root containing {args.split}.tsv")

    with tsv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    required = {"path", "sentence"}
    if not rows or not required.issubset(rows[0]):
        p.error(f"{tsv} must contain columns: path, sentence")

    # Deterministic stratification by transcript length, then seeded sampling.
    buckets = {"short": [], "medium": [], "long": []}
    for row in rows:
        if row.get("sentence", "").strip():
            buckets[classify(row["sentence"])].append(row)
    rng = random.Random(args.seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected = []
    names = list(buckets)
    while len(selected) < args.limit and any(buckets.values()):
        for name in names:
            if buckets[name] and len(selected) < args.limit:
                selected.append(buckets[name].pop())

    clips_dir = args.corpus_dir / "clips"
    out_clips = args.output / "clips"
    out_clips.mkdir(parents=True, exist_ok=True)
    manifest = []
    missing_audio = []
    for i, row in enumerate(selected):
        source = clips_dir / row["path"]
        if not source.is_file():
            missing_audio.append(str(source))
            continue
        target_name = f"cvja27_{i:05d}{source.suffix.lower()}"
        target = out_clips / target_name
        shutil.copy2(source, target)
        manifest.append({
            "case_id": f"cvja27_{i:05d}",
            "category": classify(row["sentence"]),
            "audio": str(Path("clips") / target_name),
            "reference": row["sentence"].strip(),
            "source_split": args.split,
            "source_path": row["path"],
            "audio_sha256": digest(target),
        })

    if missing_audio:
        examples = ", ".join(missing_audio[:5])
        raise SystemExit(
            f"Selected {len(selected)} rows but {len(missing_audio)} audio files are missing; "
            f"examples: {examples}"
        )
    if len(manifest) != len(selected):
        raise SystemExit(f"Prepared {len(manifest)} clips from {len(selected)} selected rows")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps({
            "dataset": "Mozilla Common Voice Scripted Speech 27.0 Japanese",
            "dataset_id": "cmu5xbhnf00fmnq07z9p3aa70",
            "license": "CC0-1.0",
            "split": args.split,
            "seed": args.seed,
            "requested": args.limit,
            "copied": len(manifest),
            "manifest": manifest,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output / "transcripts.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["case_id", "category", "reference", "hypothesis"])
        for item in manifest:
            w.writerow([item["case_id"], item["category"], item["reference"], ""])
    print(json.dumps({"requested": args.limit, "copied": len(manifest),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0 if manifest else 2


if __name__ == "__main__":
    raise SystemExit(main())
