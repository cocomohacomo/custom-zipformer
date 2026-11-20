# Zipformer Deletion Investigation

## Purpose
This repository is the testbed for investigating and reducing ASR deletions (audio is present but text is omitted). Keep changes and conclusions evidence-based; do not conflate deletions with substitutions or insertions.

## Investigation ledger
- Status: Planning / baseline not yet measured
- First experiment: Pipeline deletion localization (no model or decoding changes)
- Primary metric: Deletion rate (reference tokens/characters omitted)
- Secondary metrics: CER, WER, insertion rate, substitution rate, short-token recall, segment recall, latency, peak RSS
- Baseline variants: current default inference configuration; preserve its exact settings and outputs
- Reproducibility: record commit, model/checkpoint hash, audio hash, runtime/library versions, configuration, and command line

## Verification points

### Input and front-end
- [ ] Confirm sample/audio frames are received and consumed exactly once (frame counts, sample rate, channel count).
- [ ] Compare waveform before and after resampling, normalization, filtering, and other preprocessing.
- [ ] Audit VAD boundaries for speech clipped at the beginning/end and for low-energy words/particles.
- [ ] Audit chunk boundaries, overlap, buffering, and streaming state continuity.

### Encoder
- [ ] Verify input frame count versus encoded time steps and subsampling ratio.
- [ ] Inspect layer-wise feature norms, variance, and similarity to detect abrupt degradation/collapse.
- [ ] Check whether short phonetic events and weak consonants remain represented after subsampling.
- [ ] Compare precision variants (FP32/INT8 where supported) with identical audio and settings.
- [ ] Identify whether the deletion is already evident in encoder output; avoid claiming feature loss from norm alone.

### Joiner
- [ ] For omitted reference regions, inspect Blank and top-K token scores at relevant encoder frames.
- [ ] Measure Blank dominance during speech and Blank-run duration.
- [ ] Separate acoustic evidence from predictor/history influence where implementation permits.
- [ ] Test whether the reference token is present in top-K but lost during selection/search.

### Predictor / Decoder
- [ ] Compare outputs under different predictor histories for identical acoustic frames (where practical).
- [ ] Identify short particles, repeated words, fillers, endings, and low-probability words that are disproportionately omitted.
- [ ] Check if hypotheses containing omitted words survive long enough for later evidence.

### Search, emission, and postprocessing
- [ ] Compare Greedy and Beam decoding under matched inputs.
- [ ] Record beam size, pruning thresholds, token limits, and score settings.
- [ ] Check token emission timing, finalization, and transcript revisions.
- [ ] Compare raw decoder output with postprocessed/final output to identify removals or deduplication.
- [ ] Audit async/buffer handling and token ordering.

## First experiment: baseline pipeline deletion localization
1. Freeze the current code, model, decoding settings, and runtime configuration; capture a commit SHA.
2. Create a small, manually verified Japanese test set covering: short particles/one-mora words, weak speech, fast speech, repeated words, fillers, sentence endings, speech near chunk/VAD boundaries, and ordinary clean speech. Include clean control examples.
3. Annotate reference transcripts and, for each suspected deletion, the time span and token/character.
4. Run the unmodified baseline. Save final transcript, intermediate trace (where available), audio/frame accounting, and configuration.
5. Align reference and hypothesis; classify each error as deletion, insertion, or substitution.
6. For deletion cases, localize the earliest stage where evidence is missing: input/VAD, encoder, joiner candidate scoring, search/predictor, emission, or postprocessing.
7. Report a table of deletion cases, stage attribution, confidence/evidence, and next experiment. If internal traces cannot be accessed, explicitly mark the stage as unresolved instead of guessing.

## Change policy
- One experimental variable per run unless a combined intervention is explicitly justified.
- Preserve baseline artifacts; do not overwrite baseline outputs.
- Any intervention must include a before/after comparison and regression checks.
- Do not apply Blank penalties, acoustic-score boosts, or other heuristic forcing until baseline traces establish a failure pattern. Such interventions may trade deletions for insertions.
- Prefer reversible, inference-time-only changes first; model retraining/weight changes are out of scope for initial experiments.

## Run record template
### YYYY-MM-DD / Run ID
- Commit:
- Model/checkpoint:
- Audio set/version and hashes:
- Runtime/config:
- Intervention:
- Results: CER / WER / deletion / insertion / substitution / short-token recall / segment recall / latency / peak RSS
- Findings:
- Regressions:
- Interpretation:
- Next step:


## Next-stage implementation status (2026-11-21)

- Added regression tests for exact match, Japanese particle deletion, insertion, substitution, empty hypothesis, empty reference, and zero-denominator ratio handling.
- Added `docs/verification/inference-trace-contract.md` defining run metadata, chunk accounting, optional token-level traces, privacy defaults, and evidence rules for stage attribution.
- The repository code-search lookup did not identify the concrete inference entry point in this pass. Do not add speculative hooks to an assumed sherpa-onnx API; first map the checked-out source tree and identify the actual recognizer/session/chunk path.
- Execution status: GitHub commits succeeded. The test suite has been authored but has **not** been executed in the target repository/runtime in this session. No audio baseline or model benchmark has been run.
- Immediate next action: run `python -m unittest tests.test_deletion_metrics` from the repository root; then add integration instrumentation at the real audio-ingest, VAD, encoder, Joiner, decoder, and postprocessing boundaries.


## Common Voice baseline acquisition update (2026-11-21)

- Selected Mozilla Common Voice Scripted Speech 27.0 Japanese, dataset ID `cmu5xbhnf00fmnq07z9p3aa70`, CC0-1.0. Dataset page reports 14.36 GB; the test subset should be downloaded through the official portal and kept local.
- Added a deterministic local subset preparation script. Default selection is 100 clips from the official test split, stratified by transcript character length with a fixed seed. Audio and manifest are excluded from Git.
- Added a generic per-clip ASR command runner. It requires the real recognizer CLI to be supplied; stdout is captured as the hypothesis.
- Repository inspection found no checked-in inference implementation, model files, or recognizer CLI (the root currently contains README, LICENSE, docs, scripts, tests, and .gitignore). Therefore model execution cannot yet be wired or measured from this repository alone.
- No dataset audio was downloaded, no audio files were committed, and no CER/WER/deletion baseline values are claimed. The user/local environment must acquire the portal dataset and provide the inference command/model.
- Next: validate both scripts with a small synthetic Common Voice-shaped fixture; then run the 100-clip test split using the actual custom-zipformer CLI and preserve the run metadata.
