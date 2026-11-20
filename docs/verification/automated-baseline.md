# Automated Japanese Zipformer baseline

## One-time setup

1. Open the Common Voice Scripted Speech 27.0 Japanese dataset in Mozilla Data Collective:
   https://mozilladatacollective.com/datasets/cmu5xbhnf00fmnq07z9p3aa70
2. Accept the dataset terms while signed in to the Mozilla Data Collective account associated with the API token.
3. Store a valid token as the repository Actions secret `MOZILLA_DATA_COLLECTIVE_TOKEN`. Never commit or paste the token into workflow files or logs.

The workflow uses the dataset ID `cmu5xbhnf00fmnq07z9p3aa70` and the official `datacollective` Python client. It does not use a manually supplied archive URL.

## Run

Open **Actions → acquire-and-measure-japanese-baseline → Run workflow** and choose a sample limit (default 100). The workflow downloads the authorized corpus, selects a deterministic subset of the `test` split, downloads the public sherpa-onnx Japanese Zipformer ReazonSpeech comparison model, runs offline inference, computes deletion-focused metrics, and uploads the result bundle as a 30-day artifact.

A push touching the workflow or its baseline scripts also triggers the workflow. This is intentional for continuous verification, but it means code changes can start a data download and inference run.

## Validation gates

The workflow fails if:
- the repository secret is missing or the MDC account has not accepted the dataset terms;
- the extracted dataset has no `test.tsv` or `clips/` directory;
- any selected audio file is missing;
- the prepared manifest is empty, inconsistent, or larger than the requested sample limit;
- any required model file (`encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`, `tokens.txt`) is absent;
- inference does not produce exactly one TSV row per prepared clip;
- the metrics case count differs from the manifest or the complete-omission metric is absent.

An empty hypothesis is not itself treated as an inference failure. It is a valid observation and is counted separately as a complete omission.

## Outputs

- `manifest.json`: sample metadata and per-audio SHA-256.
- `transcripts.tsv`: reference and actual model hypothesis for every decoded clip.
- `baseline_metrics.json`: aggregate and per-clip CER, substitutions, deletions, insertions, and deletion rates.
- `metrics.stdout.json`: stdout copy of the metrics report.

The metrics include `complete_omission.empty_hypothesis_count` and `empty_hypothesis_rate`; each case also carries a `complete_omission` boolean.

## Interpretation limits

The model is a public third-party comparison model, not this repository's custom Zipformer. This scripted-read-speech result does not establish spontaneous-speech or streaming performance. The whitespace-token WER is only a rough diagnostic for Japanese, not a linguistic word-level metric. CER and deletion rates are calculated on raw characters; any future normalization policy must be recorded and applied identically to all runs.

The workflow does not claim a baseline until the full run completes and the artifact is available.


## Extended omission and signal diagnostics

The inference TSV records each clip's duration, source sample rate/channel count, RMS level, peak level, near-silence fraction, and clipping fraction. These are screening signals, not proof of a causal failure.

The metrics report also separates:
- complete omissions: empty hypothesis;
- near-complete omissions: non-empty hypothesis with at most 10% as many characters as the reference;
- severe deletion cases: character deletion rate of at least 50%;
- category-level CER and deletion rate for short, medium, and long references.

Japanese WER is not reported. The previous whitespace-split score is retained only as a clearly labeled diagnostic because Japanese text generally has no spaces between words. CER remains raw Unicode-code-point CER with no normalization; punctuation, width, and script normalization are not silently applied.
