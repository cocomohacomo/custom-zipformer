# Baseline measurement quickstart

The repository's deletion-focused metric helper consumes TSV references and hypotheses. It does **not** run ASR; first collect transcripts from the existing inference pipeline without changing decoding settings.

## Input format

Create a TSV with these four columns (header required):

```tsv
case_id	category	reference	hypothesis
clean01	clean	今日は会議です	今日は会議です
short01	short-token	それはまだです	それまだです
```

Tabs must be actual tab separators. Transcript fields must not contain tabs or newlines. Character metrics use raw code points (no Unicode normalization or punctuation stripping); word metrics split on whitespace. Keep a normalization policy consistent across baseline and every later run.

## Run

```bash
python scripts/deletion_metrics.py transcripts.tsv --output-json baseline_metrics.json
```

The JSON reports aggregate and per-case character/word substitution, deletion, insertion, error rate, and deletion rate. Rates are undefined (null) if the reference denominator is zero.

## Baseline protocol

- Save raw audio, reference transcript, hypothesis transcript, and run config; keep data private if it contains sensitive speech.
- Record commit SHA, model/checkpoint checksum, runtime version, decoding options, preprocessing/VAD options, audio sample rate/channel count, and audio hashes.
- Do not modify the model, search, VAD, preprocessing, or normalization during baseline capture.
- Add cases from the categories in `docs/verification/deletion-investigation.md`.
- The baseline is a frozen comparison point. Do not overwrite it when running interventions.

## Limitations

This helper's character-level alignment uses raw Python Unicode code points and word-level alignment uses whitespace tokenization. It is a simple baseline metric utility, not a Japanese linguistic tokenizer, and CER values may differ from evaluation tools that normalize Unicode, punctuation, or kana/kanji. For diagnostic alignment, use a separate alignment tool and document its normalization rules.
