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
    p.add_argument("--normalize-rms-dbfs", type=float, default=None,
                   help="Optionally amplify low-RMS audio toward this RMS dBFS target; no attenuation is applied")
    p.add_argument("--max-gain-db", type=float, default=12.0,
                   help="Maximum gain for optional RMS normalization (default: 12 dB)")
    p.add_argument("--chunk-seconds", type=float, default=None,
                   help="Decode each audio independently in fixed-size chunks; omit for whole-utterance decoding")
    p.add_argument("--silence-aware-chunk-seconds", type=float, default=None,
                   help="Split near low-energy frames around this target duration (mutually exclusive with --chunk-seconds)")
    p.add_argument("--chunk-overlap-seconds", type=float, default=0.0, help="Shared context for fixed chunks; requires --chunk-seconds")
    p.add_argument("--chunk-search-window-seconds", type=float, default=2.0,
                   help="Search +/- this many seconds for a quiet chunk boundary (default: 2)")
    args = p.parse_args()
    if args.threads < 1:
        p.error("--threads must be >= 1")
    if args.max_gain_db < 0:
        p.error("--max-gain-db must be >= 0")
    if args.chunk_seconds is not None and args.chunk_seconds <= 0:
        p.error("--chunk-seconds must be > 0")
    if args.silence_aware_chunk_seconds is not None and args.silence_aware_chunk_seconds <= 0:
        p.error("--silence-aware-chunk-seconds must be > 0")
    if args.chunk_seconds is not None and args.silence_aware_chunk_seconds is not None:
        p.error("--chunk-seconds and --silence-aware-chunk-seconds are mutually exclusive")
    if args.chunk_overlap_seconds < 0 or (args.chunk_overlap_seconds and args.chunk_seconds is None):
        p.error("--chunk-overlap-seconds requires --chunk-seconds and must be >= 0")
    if args.chunk_seconds is not None and args.chunk_overlap_seconds >= args.chunk_seconds:
        p.error("--chunk-overlap-seconds must be smaller than --chunk-seconds")
    if args.chunk_search_window_seconds < 0:
        p.error("--chunk-search-window-seconds must be >= 0")
    if args.normalize_rms_dbfs is not None and not -90 <= args.normalize_rms_dbfs <= 0:
        p.error("--normalize-rms-dbfs must be between -90 and 0 dBFS")

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
        "source_sample_rate", "source_channels", "applied_gain_db", "inference_audio_rms", "inference_chunks", "chunk_boundaries_seconds",
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
            gain = 1.0
            if args.normalize_rms_dbfs is not None and rms > 0:
                target_rms = 10.0 ** (args.normalize_rms_dbfs / 20.0)
                if rms < target_rms:
                    gain = min(10.0 ** (args.max_gain_db / 20.0), target_rms / rms)
                    gain = min(gain, 0.95 / peak) if peak > 0 else gain
                    mono = mono * np.float32(gain)
            inference_rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
            if sr != 16000:
                from math import gcd
                g = gcd(int(sr), 16000)
                mono = resample_poly(mono, 16000 // g, int(sr) // g).astype(np.float32)
            inference_samples = np.asarray(mono, dtype=np.float32)
            boundaries = []
            if args.silence_aware_chunk_seconds is not None:
                from audio_chunking import split_at_low_energy
                chunks, boundaries = split_at_low_energy(
                    inference_samples, 16000, args.silence_aware_chunk_seconds,
                    search_window_seconds=args.chunk_search_window_seconds,
                )
            elif args.chunk_seconds is None:
                chunks = [inference_samples]
            elif args.chunk_overlap_seconds > 0:
                from audio_chunking import split_with_overlap
                chunks, boundaries = split_with_overlap(
                    inference_samples, 16000, args.chunk_seconds,
                    args.chunk_overlap_seconds,
                )
            else:
                chunk_samples = max(1, int(round(args.chunk_seconds * 16000)))
                chunks = [inference_samples[start:start + chunk_samples]
                          for start in range(0, len(inference_samples), chunk_samples)]
            chunk_texts = []
            for chunk in chunks:
                if chunk.size == 0:
                    continue
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, chunk)
                recognizer.decode_stream(stream)
                chunk_texts.append(stream.result.text.strip())
            if args.chunk_overlap_seconds > 0:
                from audio_chunking import merge_overlapping_text
                hypothesis = merge_overlapping_text(chunk_texts)
            else:
                hypothesis = "".join(chunk_texts)
            hypothesis = hypothesis.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
            reference = str(item["reference"]).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
            writer.writerow({
                "case_id": str(item["case_id"]), "category": str(item.get("category", "")),
                "reference": reference, "hypothesis": hypothesis,
                "audio_duration_seconds": f"{duration:.6f}",
                "audio_rms": f"{rms:.8f}", "audio_peak": f"{peak:.8f}",
                "near_silence_fraction": f"{near_silence:.6f}",
                "clipping_fraction": f"{clipping:.6f}",
                "source_sample_rate": int(sr), "source_channels": channels,
                "applied_gain_db": f"{20.0 * np.log10(gain):.4f}",
                "inference_audio_rms": f"{inference_rms:.8f}",
                "inference_chunks": len(chunks),
                "chunk_boundaries_seconds": ",".join(f"{v:.3f}" for v in boundaries),
            })
    print(json.dumps({"decoded": len(entries), "output_tsv": str(args.output_tsv),
                      "normalization": {"target_rms_dbfs": args.normalize_rms_dbfs,
                                        "max_gain_db": args.max_gain_db if args.normalize_rms_dbfs is not None else 0},
                      "chunk_seconds": args.chunk_seconds,
                      "silence_aware_chunk_seconds": args.silence_aware_chunk_seconds,
                      "diagnostics": fields[4:]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
