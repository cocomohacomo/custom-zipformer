# Inference trace contract for deletion diagnosis

This document defines a minimal, privacy-conscious trace contract. It is deliberately independent of any specific Zipformer wrapper until the repository's actual inference entry point is confirmed.

## Run-level metadata

Record once per run:
- run ID, UTC timestamp, git commit
- model file identifiers and SHA-256 hashes
- runtime/library versions and execution provider
- sample rate, channel count, audio duration, decoding mode
- VAD/preprocessing options, chunk size/overlap, beam and pruning settings
- whether the run is baseline or an intervention

## Per-stream / per-chunk accounting

Record:
- stream ID (opaque, not a speaker identity)
- chunk index, input sample start/end, input frame count
- resampled frame count and VAD speech start/end
- encoder input frames and output time steps
- chunk overlap and carried-state indicators
- finalization reason and whether the chunk was flushed

These fields detect lost, duplicated, clipped, or unflushed audio before attributing an omission to the neural network.

## Optional token-level diagnostic trace

Only enable for selected diagnostic cases because it can be large:
- encoder time index and layer identifier
- feature vector summary (dimension, finite-value count, L2 norm, mean, variance); avoid storing raw vectors by default
- Joiner top-K token IDs and scores, including Blank score
- predictor state/history identifier (opaque hash, not text)
- selected token, emission time, hypothesis score, beam/pruning decision
- raw token sequence before text conversion and final text after postprocessing

Do not log raw audio, full feature tensors, or recognized personal content by default. Use synthetic or consented test data and explicit retention controls.

## Event format

One JSON object per line (JSONL). Example keys: schema_version, run_id, stream_id, chunk_index, input_sample_start, input_sample_end, input_frames, encoder_input_frames, encoder_output_steps, vad_speech_start, vad_speech_end, finalized.

The example is illustrative, not a claim that the current runtime emits these fields.

## Attribution rules

- Input/VAD attribution requires evidence that the reference speech samples were excluded or clipped.
- Encoder attribution requires a controlled comparison showing the relevant acoustic distinction becomes unavailable or materially degraded at its output. Norm changes alone are insufficient.
- Joiner attribution requires frame-level scores showing the expected token was not competitive despite adequate encoder evidence.
- Search/predictor attribution requires evidence that a viable token/hypothesis was pruned or suppressed.
- Postprocessing attribution requires a token or text span present before postprocessing and absent afterward.
- If the trace is unavailable, report `unresolved`; do not infer a stage from the final transcript alone.
