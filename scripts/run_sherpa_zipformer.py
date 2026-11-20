#!/usr/bin/env python3
"""Decode a manifest with sherpa-onnx's Japanese Zipformer transducer."""
from __future__ import annotations
import argparse
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
    rows = data.get("manifest")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("Manifest is empty; refusing to create a false baseline")
    output = ["case_id\tcategory\treference\thypothesis"]
    for item in rows:
        audio_path = (args.manifest.parent / item["audio"]).resolve()
        if not audio_path.is_file():
            raise SystemExit(f"Audio missing: {audio_path}")
        samples, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if sr != 16000:
            from math import gcd
            g = gcd(int(sr), 16000)
            samples = resample_poly(samples, 16000 // g, int(sr) // g).astype(np.float32)
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, np.asarray(samples, dtype=np.float32))
        recognizer.decode_stream(stream)
        hypothesis = stream.result.text.replace("\t", " ").replace("\n", " ").strip()
        reference = item["reference"].replace("\t", " ").replace("\n", " ").strip()
        output.append("\t".join((str(item["case_id"]), str(item.get("category", "")), reference, hypothesis)))
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.output_tsv.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(json.dumps({"decoded": len(rows), "output_tsv": str(args.output_tsv)}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
