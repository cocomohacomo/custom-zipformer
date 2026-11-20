# Baseline measurement runbook

This repository's baseline pipeline is split into three independently auditable stages:

1. **Prepare**: select a deterministic, transcript-length-stratified subset from the official Common Voice Japanese test split. The manifest records the source clip path and SHA-256 of each copied audio file.
2. **Infer**: run the supplied recognizer command once per clip. The command must contain `{audio}`; no shell is used. Stdout is interpreted as transcript text. Use `--stdout-mode last-line` if the CLI prints progress lines before the final transcript.
3. **Score**: calculate aggregate and per-case character/word edit metrics, including deletion rates.

## Commands

```bash
python scripts/prepare_common_voice_baseline.py /path/to/extracted/ja \
  --split test --limit 100 --seed 270021 --output tests/data/common_voice_ja

python scripts/run_baseline_asr.py tests/data/common_voice_ja/manifest.json \
  --command 'YOUR_ASR_CLI --input {audio}' \
  --stdout-mode last-line --timeout 120 \
  --output-tsv tests/data/common_voice_ja/transcripts.tsv \
  --report-json tests/data/common_voice_ja/run_report.json

python scripts/deletion_metrics.py tests/data/common_voice_ja/transcripts.tsv \
  --output-json tests/data/common_voice_ja/baseline_metrics.json
```

## Required provenance

Keep the following alongside the local result bundle: dataset release and split, sample seed, manifest/audio hashes, recognizer command, model file hashes, repository commit, runtime/library versions, CPU model, thread count, preprocessing and decoding options. Never commit corpus audio or derived clips. Preserve the raw hypothesis TSV and run report before changing normalization.

## Interpretation guardrails

- Raw-codepoint CER is the primary comparable character metric here. The current whitespace-token WER is not a linguistically meaningful Japanese word error rate.
- Report deletion counts/rates separately from substitutions and insertions. A low overall CER can hide systematic omission of particles or short function words.
- Common Voice is scripted read speech. Passing this baseline does not establish robustness to spontaneous speech, overlap, fillers, repetitions, VAD cuts, or streaming chunk boundaries.
- Empty hypotheses are valid inference outputs and count as deletions; failed commands and timeouts are not silently scored as empty transcripts.
- The baseline scripts are tooling, not evidence of measured model performance. Publish numeric results only after the local audio and actual model have been run.
