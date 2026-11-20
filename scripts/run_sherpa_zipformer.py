#!/usr/bin/env python3
"""Decode a manifest with sherpa-onnx Japanese Zipformer and record audio diagnostics."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-tsv", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180)
    args = p.parse_args()
    if args.threads < 1:
        p.error("--threads must be >= 1")

    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        import sherpa_onnx
    except ImportError as e:
        raise SystemExit(f"Missing inference dependency: {e}")

    def model_file(stem: str) -> Path:
        candidates = sorted(args.model_dir.rglob(f"{stem}*.onnx"))
        if not candidates:
            raise SystemExit(f"Cannot find {stem}*.onnx under {args.model_dir}")
        return candidates[0]

    tokens = next(iter(args.model_dir.rglob("tokens.txt")), None)
    if tokens is None:
        raise SystemExit(f"Cannot find tokens.txt under {args.model_dir}")
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_file("encoder")),
        decoder=str(model_file("decoder")),
        joiner=str(model_file("joiner")),
        tokens=str(tokens),
        num_threads=args.threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = data.get("manifest")
    if not isinstance(entries, list) or not entries:
        raise SystemExit("Manifest is empty; refusing to create a false baseline")

    fields = [
        "case_id", "category", "reference", "hypothesis",
        "audio_duration_seconds", "audio_rms", "audio_peak",
        "near_silence_fraction", "clipping_fraction",
        "source_sample_rate", "source_channels",
    ]
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for item in entries:
            audio_path = (args.manifest.parent / item["audio"]).resolve()
            if not audio_path.is_file():
                raise SystemExit(f"Audio missing: {audio_path}")
            samples, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
            channels = int(samples.shape[1])
            mono = samples.mean(axis=1)
            if mono.size == 0:
                raise SystemExit(f"Audio is empty: {audio_path}")
            duration = float(len(mono) / sr)
            rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
            peak = float(np.max(np.abs(mono)))
            near_silence = float(np.mean(np.abs(mono) < 1e-4))
            clipping = float(np.mean(np.abs(mono) >= 0.999))
            if sr != 16000:
                from math import gcd
                g = gcd(int(sr), 16000)
                mono = resample_poly(mono, 16000 // g, int(sr) // g).astype(np.float32)
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, np.asarray(mono, dtype=np.float32))
            recognizer.decode_stream(stream)
            hypothesis = stream.result.text.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
            reference = str(item["reference"]).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
            writer.writerow({
                "case_id": str(item["case_id"]), "category": str(item.get("category", "")),
                "reference": reference, "hypothesis": hypothesis,
                "audio_duration_seconds": f"{duration:.6f}",
                "audio_rms": f"{rms:.8f}", "audio_peak": f"{peak:.8f}",
                "near_silence_fraction": f"{near_silence:.6f}",
                "clipping_fraction": f"{clipping:.6f}",
                "source_sample_rate": int(sr), "source_channels": channels,
            })
    print(json.dumps({"decoded": len(entries), "output_tsv": str(args.output_tsv),
                      "diagnostics": fields[4:]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
