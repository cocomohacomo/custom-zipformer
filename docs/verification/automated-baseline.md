# Automated Japanese Zipformer baseline

## One-time repository setup

Add the repository Actions secret `COMMON_VOICE_ARCHIVE_URL`. Its value must be a direct, authorized download URL for the Common Voice Scripted Speech 27.0 Japanese archive (ZIP or TAR) that contains `test.tsv` and `clips/`. Obtain it through Mozilla Data Collective after accepting the dataset terms. Do not commit the URL, credentials, corpus, or audio files.

## Run

Open **Actions → acquire-and-measure-japanese-baseline → Run workflow** and choose a sample limit (default 100). The workflow downloads the public sherpa-onnx Japanese Zipformer ReazonSpeech comparison model, prepares a deterministic sample, runs inference, computes CER and deletion rates, and uploads the result bundle as a 30-day artifact.

A push touching this workflow or its inference scripts also starts the workflow. If the dataset secret is absent, it intentionally fails before downloading anything.

## Outputs

The metrics report includes a separate `complete_omission` section (`empty_hypothesis_count` and `empty_hypothesis_rate`) so clips with entirely empty model output are tracked independently from character deletion rate. Each case also carries a `complete_omission` boolean. The workflow uploads both the JSON report and its stdout copy.

- `manifest.json`: sample metadata and per-audio SHA-256.
- `transcripts.tsv`: reference and actual model hypothesis for every decoded clip.
- `baseline_metrics.json`: aggregate and per-clip CER, substitutions, deletions, insertions, and deletion rates.

The model is a public third-party comparison model, not this repository's custom Zipformer. This scripted-read-speech result does not establish spontaneous-speech or streaming performance. The existing whitespace-token WER is only a rough diagnostic for Japanese, not a linguistic word-level metric.

## Runtime assumptions

The adapter expects sherpa-onnx's offline transducer model files named `encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`, and `tokens.txt`, recursively under the extracted model directory. It resamples input audio to 16 kHz and averages multichannel audio to mono.
