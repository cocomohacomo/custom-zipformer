#!/usr/bin/env python3
"""Decode a manifest with sherpa-onnx Japanese Zipformer and record audio diagnostics."""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
import math
from pathlib import Path



def prepare_inference_audio(samples, source_sample_rate: int, target_sample_rate: int = 16000):
    """Convert finite mono/stereo audio to finite float32 at the model sample rate.

    If averaging stereo channels would nearly cancel an anti-phase signal,
    retain the higher-energy channel instead of feeding near-silence to ASR.
    """
    import numpy as np
    from math import gcd

    audio = np.asarray(samples)
    if audio.ndim not in (1, 2):
        raise ValueError("audio must be mono or (frames, channels)")
    if audio.ndim == 2 and audio.shape[1] < 1:
        raise ValueError("audio must contain at least one channel")
    if audio.shape[0] == 0:
        raise ValueError("audio must contain at least one frame")
    if np.issubdtype(audio.dtype, np.inexact) and not np.isfinite(audio).all():
        raise ValueError("audio must contain only finite samples")
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rates must be positive")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        channels = [audio[:, index] for index in range(audio.shape[1])]
        mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
        channel_rms = [float(np.sqrt(np.mean(np.square(ch, dtype=np.float64)))) for ch in channels]
        mean_rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        max_channel_rms = max(channel_rms, default=0.0)
        if max_channel_rms > 0 and mean_rms < 0.25 * max_channel_rms:
            mono = channels[int(np.argmax(channel_rms))].copy()
    else:
        mono = audio.copy()

    if source_sample_rate != target_sample_rate:
        from scipy.signal import resample_poly
        divisor = gcd(int(source_sample_rate), int(target_sample_rate))
        mono = resample_poly(
            mono, target_sample_rate // divisor, source_sample_rate // divisor
        ).astype(np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError("prepared inference audio is empty or non-finite")
    return np.asarray(mono, dtype=np.float32)



def decode_audio_chunks(recognizer, chunks):
    """Decode every non-empty offline chunk, including the final partial chunk."""
    results = []
    durations = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, chunk)
        # OfflineRecognizer.decode_stream() completes the supplied waveform.
        decode_started = time.perf_counter()
        recognizer.decode_stream(stream)
        decode_seconds = time.perf_counter() - decode_started
        result = stream.result
        raw_text = str(getattr(result, "text", "") or "")
        tokens = list(getattr(result, "tokens", []) or [])
        timestamps = list(getattr(result, "timestamps", []) or [])
        results.append({
            "raw_text": raw_text,
            "text": raw_text.strip(),
            "tokens": tokens,
            "timestamps": timestamps,
            "decode_seconds": decode_seconds,
        })
        durations.append(float(len(chunk) / 16000))
    return results, durations


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-tsv", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--decoding-method", choices=("greedy_search", "modified_beam_search"), default="greedy_search", help="Transducer decoder search method; default preserves the baseline")
    p.add_argument("--blank-penalty", type=float, default=0.0, help="Transducer blank penalty; default preserves the Sherpa-ONNX baseline")
    p.add_argument("--max-active-paths", type=int, default=4, help="Active paths for modified_beam_search (ignored by greedy_search)")
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
    if not math.isfinite(args.blank_penalty):
        p.error("--blank-penalty must be finite")
    if args.max_active_paths < 1:
        p.error("--max-active-paths must be >= 1")
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
        decoding_method=args.decoding_method,
        blank_penalty=args.blank_penalty,
        max_active_paths=args.max_active_paths,
    )
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = data.get("manifest")
    if not isinstance(entries, list) or not entries:
        raise SystemExit("Manifest is empty; refusing to create a false baseline")

    fields = [
        "case_id", "category", "reference", "hypothesis",
        "audio_duration_seconds", "audio_rms", "audio_peak",
        "near_silence_fraction", "clipping_fraction",
        "source_sample_rate", "source_channels", "applied_gain_db", "inference_audio_rms", "inference_chunks", "inference_empty_chunks", "inference_short_chunks_lt_25ms", "inference_silent_chunks", "chunk_boundaries_seconds", "chunk_start_offsets_seconds", "overlap_merge_method",
        "raw_token_count", "timestamp_count", "raw_decoder_text_chars", "chunk_text_chars",
        "merged_text_chars", "final_text_chars", "chunk_merge_char_delta",
        "final_normalization_char_delta", "token_density_per_second", "decode_seconds",
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
            if sr <= 0:
                raise SystemExit(f"Invalid sample rate: {audio_path}")
            duration = float(len(samples) / sr)
            # Downmix with anti-phase protection, then resample exactly once.
            mono = prepare_inference_audio(samples, int(sr), 16000)
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
            inference_samples = np.asarray(mono, dtype=np.float32)
            boundaries = []
            chunk_start_offsets = [0.0]
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
                chunks, overlap_starts = split_with_overlap(
                    inference_samples, 16000, args.chunk_seconds,
                    args.chunk_overlap_seconds,
                )
                chunk_start_offsets = [0.0, *overlap_starts]
            else:
                from audio_chunking import split_fixed_chunks
                chunks = split_fixed_chunks(inference_samples, 16000, args.chunk_seconds)
            chunk_results, chunk_durations = decode_audio_chunks(recognizer, chunks)
            chunk_texts = [item["text"] for item in chunk_results]
            overlap_merge_method = "not_applicable"
            if args.chunk_overlap_seconds > 0:
                from audio_chunking import merge_timestamped_chunks
                hypothesis, overlap_merge_method = merge_timestamped_chunks(
                    chunk_results, chunk_start_offsets, chunk_durations
                )
            else:
                hypothesis = "".join(chunk_texts)
            merged_text = hypothesis
            hypothesis = hypothesis.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
            raw_token_count = sum(len(chunk.get("tokens", [])) for chunk in chunk_results)
            timestamp_count = sum(len(chunk.get("timestamps", [])) for chunk in chunk_results)
            raw_decoder_text_chars = sum(len(chunk.get("raw_text", "")) for chunk in chunk_results)
            chunk_text_chars = sum(len(chunk.get("text", "")) for chunk in chunk_results)
            merged_text_chars = len(merged_text)
            final_text_chars = len(hypothesis)
            chunk_merge_char_delta = merged_text_chars - chunk_text_chars
            final_normalization_char_delta = final_text_chars - merged_text_chars
            decode_seconds = sum(float(chunk.get("decode_seconds", 0.0)) for chunk in chunk_results)
            token_density = raw_token_count / duration if duration > 0 else 0.0
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
                "inference_chunks": len(chunk_durations),
                "inference_empty_chunks": sum(not value for value in chunk_texts),
                "inference_short_chunks_lt_25ms": sum(1 for value in chunk_durations if value < 0.025),
                "inference_silent_chunks": sum(
                    1 for chunk in chunks if chunk.size and float(np.max(np.abs(chunk))) < 1e-8
                ),
                "chunk_boundaries_seconds": ",".join(f"{v:.3f}" for v in boundaries),
                "chunk_start_offsets_seconds": ",".join(f"{v:.3f}" for v in chunk_start_offsets),
                "overlap_merge_method": overlap_merge_method,
                "raw_token_count": raw_token_count,
                "timestamp_count": timestamp_count,
                "raw_decoder_text_chars": raw_decoder_text_chars,
                "chunk_text_chars": chunk_text_chars,
                "merged_text_chars": merged_text_chars,
                "final_text_chars": final_text_chars,
                "chunk_merge_char_delta": chunk_merge_char_delta,
                "final_normalization_char_delta": final_normalization_char_delta,
                "token_density_per_second": f"{token_density:.6f}",
                "decode_seconds": f"{decode_seconds:.6f}",
            })
    print(json.dumps({"decoded": len(entries), "output_tsv": str(args.output_tsv),
                      "normalization": {"target_rms_dbfs": args.normalize_rms_dbfs,
                                        "max_gain_db": args.max_gain_db if args.normalize_rms_dbfs is not None else 0},
                      "decoder": {"decoding_method": args.decoding_method,
                                  "blank_penalty": args.blank_penalty,
                                  "max_active_paths": args.max_active_paths if args.decoding_method == "modified_beam_search" else None},
                      "chunk_seconds": args.chunk_seconds,
                      "silence_aware_chunk_seconds": args.silence_aware_chunk_seconds,
                      "diagnostics": fields[4:]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
