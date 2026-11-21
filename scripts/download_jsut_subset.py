#!/usr/bin/env python3
"""Prepare a deterministic JSUT evaluation subset from the public HF mirror.

JSUT audio is restricted to research/personal use under its published terms.
This script downloads only the selected WAVs and does not redistribute them.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import hashlib
import json
import random
import shutil
from pathlib import Path

REPO_ID = "FluidInference/JSUT-basic5000"
SUBSET_ROOT = "basic5000"
SUBSET_ROOT_CANDIDATES = ("basic5000", "jsut_ver1.1/basic5000")


def locate_jsut_transcripts(hf_hub_download):
    """Resolve the mirror's actual file layout without exposing lookup errors."""
    for root in SUBSET_ROOT_CANDIDATES:
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=f"{root}/transcript_utf8.txt",
            )
            return Path(path), root
        except Exception:
            continue
    raise ValueError("Required JSUT transcript file could not be located")


def parse_transcripts(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError("Malformed JSUT transcript input")
        utterance_id, sentence = line.split(":", 1)
        utterance_id, sentence = utterance_id.strip(), sentence.strip()
        if not utterance_id or not sentence:
            continue
        if utterance_id in result:
            raise ValueError("Duplicate JSUT transcript entry")
        result[utterance_id] = sentence
    if not result:
        raise ValueError("No valid JSUT transcripts found")
    return result


def classify(sentence: str) -> str:
    length = len(sentence)
    if length <= 15:
        return "short"
    if length <= 40:
        return "medium"
    return "long"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(limit: int, output: Path, seed: int = 270021) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub to download the JSUT evaluation subset") from exc

    transcript_path, dataset_root = locate_jsut_transcripts(hf_hub_download)
    transcripts = parse_transcripts(transcript_path.read_text(encoding="utf-8"))
    buckets = {"short": [], "medium": [], "long": []}
    for utterance_id, sentence in transcripts.items():
        buckets[classify(sentence)].append((utterance_id, sentence))
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for category in buckets:
            if buckets[category] and len(selected) < limit:
                selected.append((category, *buckets[category].pop()))
    if not selected:
        raise ValueError("No JSUT utterances selected")

    clips_dir = output / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (category, utterance_id, sentence) in enumerate(selected):
        filename = f"{dataset_root}/wav/{utterance_id}.wav"
        source = Path(hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=filename
        ))
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError("Downloaded JSUT audio is missing or empty")
        target_name = f"jsut_{index:05d}.wav"
        target = clips_dir / target_name
        shutil.copy2(source, target)
        manifest.append({
            "case_id": f"jsut_{index:05d}",
            "category": category,
            "audio": str(Path("clips") / target_name),
            "reference": sentence,
            "source_dataset": REPO_ID,
            "source_utterance_id": utterance_id,
            "audio_sha256": sha256(target),
        })

    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "JSUT basic5000 evaluation subset (Hugging Face mirror)",
        "dataset_id": REPO_ID,
        "license": "CC-BY-SA-4.0 text; audio research/personal-use terms",
        "usage_restriction": "Research/personal use only; audio redistribution prohibited",
        "seed": seed,
        "requested": limit,
        "copied": len(manifest),
        "manifest": manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=270021)
    parser.add_argument("--output", type=Path, default=Path("tests/data/jsut_ja"))
    args = parser.parse_args()
    if not 1 <= args.limit <= 500:
        parser.error("--limit must be between 1 and 500")
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            metadata = prepare_dataset(args.limit, args.output, args.seed)
    except Exception:
        print("JSUT dataset preparation failed; detailed source information is suppressed.")
        return 1
    print(json.dumps({
        "dataset": metadata["dataset_id"],
        "requested": metadata["requested"],
        "copied": metadata["copied"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
