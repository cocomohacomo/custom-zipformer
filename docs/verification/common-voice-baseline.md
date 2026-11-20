# Common Voice Japanese baseline audio

## Dataset choice

- Dataset: Mozilla Common Voice Scripted Speech 27.0 Japanese (`ja`)
- Dataset ID: `cmu5xbhnf00fmnq07z9p3aa70`
- Release: 2026-09-17
- License: CC0-1.0
- Format: MP3
- Published corpus size: 14.36 GB; do not download the full corpus for the first test.
- The dataset page prohibits speaker identification and re-hosting/redistribution. Keep the downloaded corpus and derived test subset local; do not commit audio, clips, or the original archive to this repository.

Official dataset page: https://mozilladatacollective.com/datasets/cmu5xbhnf00fmnq07z9p3aa70

## Acquisition

1. Open the official dataset page above and download the Japanese 27.0 archive after accepting the portal terms.
2. Extract it locally. The extracted root should contain `test.tsv` (or `dev.tsv`) and `clips/`.
3. Prefer the official `test` split for baseline evaluation. Do not use training clips.
4. Prepare a reproducible 100-clip subset:

```bash
python scripts/prepare_common_voice_baseline.py /path/to/cv-corpus-27.0-2026-09-11/ja \
  --split test --limit 100 --seed 270021 \
  --output tests/data/common_voice_ja
```

The script copies only selected local clips, stores their SHA-256 hashes and references in `manifest.json`, and writes `transcripts.tsv` with empty hypotheses. It does not upload or redistribute data. If the official archive's directory layout differs, pass the directory containing the split TSV and its sibling `clips/` directory.

## Baseline execution

The current repository metric helper does not invoke the recognizer. First run the recognizer on every audio path in `manifest.json`, preserve its raw transcript, and fill the `hypothesis` column in `transcripts.tsv`. Then run:

```bash
python scripts/deletion_metrics.py tests/data/common_voice_ja/transcripts.tsv \
  --output-json tests/data/common_voice_ja/baseline_metrics.json
```

Record the recognizer command, model hash, commit, runtime, audio preprocessing, and decode options. Keep raw outputs immutable.

## Scope and limitations

Common Voice is read speech, not spontaneous conversation. It provides clean, speaker-diverse baseline coverage, but it does not adequately test fillers, interruptions, repetitions, overlapping speech, or VAD/chunk-boundary failures. Add a separate spontaneous-speech corpus only after reviewing its terms and suitability. Do not infer that a short transcript is a one-mora/particle challenge without listening to the clip.


## Generic inference adapter

The repository currently contains investigation/measurement scaffolding but no checked-in recognizer CLI or model artifact. Therefore the runner accepts the actual inference command instead of guessing a project-specific API:

```bash
python scripts/run_baseline_asr.py tests/data/common_voice_ja/manifest.json \
  --command 'YOUR_ASR_CLI --input {audio}' \
  --output-tsv tests/data/common_voice_ja/transcripts.tsv
python scripts/deletion_metrics.py tests/data/common_voice_ja/transcripts.tsv \
  --output-json tests/data/common_voice_ja/baseline_metrics.json
```

The runner invokes each clip separately, substitutes the absolute audio path for `{audio}`, treats stdout as the raw transcript, records failures, and never uses a shell. Confirm the CLI's output convention before running. If it writes transcript files instead of stdout, use a small adapter command that prints only the recognized text.

## Measurement boundary

Actual baseline numbers require all three local prerequisites: (1) Common Voice archive downloaded through the official portal, (2) extracted test split, and (3) the custom-zipformer inference command plus its model files. None of those local assets/credentials are available through the repository connection, so no corpus download or ASR run is claimed by this commit.
