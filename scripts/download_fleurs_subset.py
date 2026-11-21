#!/usr/bin/env python3
"""Prepare a deterministic Japanese FLEURS evaluation subset locally.

FLEURS is licensed CC-BY-4.0. Audio is downloaded only to the CI runner;
this script never commits or uploads audio.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import csv
import hashlib
import json
import random
import shutil
import tarfile
from pathlib import Path

REPO_ID = "google/fleurs"
SPLIT = "dev"
BASE = f"data/ja_jp"
TRANSCRIPT_FILE = f"{BASE}/{SPLIT}.tsv"
AUDIO_ARCHIVE = f"{BASE}/audio/{SPLIT}.tar.gz"


def classify(text: str) -> str:
    size = len(text)
    if size <= 30:
        return "short"
    if size <= 70:
        return "medium"
    return "long"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {"file_name", "transcription"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"FLEURS TSV missing required columns: {required}")
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        name = Path(row["file_name"]).name
        text = row["transcription"].strip()
        if not name or name != row["file_name"] or not text:
            continue
        if name in unique:
            raise ValueError("Duplicate FLEURS audio filename")
        row["file_name"] = name
        row["transcription"] = text
        unique[name] = row
    return list(unique.values())


def prepare_dataset(limit: int, output: Path, seed: int = 270021) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub to download FLEURS") from exc

    transcript_path = Path(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=TRANSCRIPT_FILE
    ))
    rows = read_rows(transcript_path)
    buckets: dict[str, list[dict[str, str]]] = {
        "short": [], "medium": [], "long": []
    }
    for row in rows:
        buckets[classify(row["transcription"])].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[tuple[str, dict[str, str]]] = []
    while len(selected) < min(limit, len(rows)) and any(buckets.values()):
        for category in buckets:
            if buckets[category] and len(selected) < min(limit, len(rows)):
                selected.append((category, buckets[category].pop()))
    if not selected:
        raise ValueError("No FLEURS utterances selected")

    archive_path = Path(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=AUDIO_ARCHIVE
    ))
    wanted = {row["file_name"] for _, row in selected}
    clips_dir = output / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            basename = Path(member.name).name
            if basename not in wanted:
                continue
            if not member.isfile() or basename in extracted:
                raise ValueError("Unexpected duplicate or non-file archive member")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Cannot read selected archive member")
            target = clips_dir / basename
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted[basename] = target
            if len(extracted) == len(wanted):
                break

    missing = sorted(wanted - extracted.keys())
    if missing:
        raise ValueError(f"FLEURS archive is missing {len(missing)} selected files")

    manifest = []
    for index, (category, row) in enumerate(selected):
        source = extracted[row["file_name"]]
        target_name = f"fleurs_{index:05d}.wav"
        target = clips_dir / target_name
        if source != target:
            source.rename(target)
        manifest.append({
            "case_id": f"fleurs_{index:05d}",
            "category": category,
            "audio": str(Path("clips") / target_name),
            "reference": row["transcription"],
            "source_dataset": REPO_ID,
            "source_split": SPLIT,
            "source_utterance_id": row.get("id", row["file_name"]),
            "audio_sha256": sha256(target),
        })

    metadata = {
        "dataset": "Google FLEURS Japanese",
        "dataset_id": REPO_ID,
        "split": SPLIT,
        "license": "CC-BY-4.0",
        "attribution": "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech",
        "source_transcripts": f"https://huggingface.co/datasets/{REPO_ID}/blob/main/{TRANSCRIPT_FILE}",
        "source_audio": f"https://huggingface.co/datasets/{REPO_ID}/blob/main/{AUDIO_ARCHIVE}",
        "usage_restriction": "CC-BY-4.0 attribution required; audio retained only in ephemeral CI workspace",
        "seed": seed,
        "requested": limit,
        "copied": len(manifest),
        "manifest": manifest,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=270021)
    parser.add_argument("--output", type=Path, default=Path("tests/data/fleurs_ja"))
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 500:
        parser.error("--limit must be between 1 and 500")
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            metadata = prepare_dataset(args.limit, args.output, args.seed)
    except Exception:
        print("FLEURS dataset preparation failed; detailed source information is suppressed.")
        return 1
    print(json.dumps({
        "dataset": metadata["dataset_id"],
        "split": metadata["split"],
        "requested": metadata["requested"],
        "copied": metadata["copied"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
