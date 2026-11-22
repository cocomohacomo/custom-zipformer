import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sanitize_asr_report import main, sanitize_comparisons, sanitize_execution_diagnostics, sanitize_metrics


class SanitizeAsrReportTests(unittest.TestCase):

    def test_execution_diagnostics_allowlist_only_aggregate_fields(self):
        raw = {
            "modes": {
                "fixed_15s": {
                    "case_count": 2,
                    "audio_duration_seconds": {"min": 3.0, "median": 8.0, "max": 12.0},
                    "chunking": {
                        "single_chunk_cases": 2, "multi_chunk_cases": 0,
                        "multi_chunk_case_rate": 0.0, "total_chunks": 2,
                        "max_chunks_per_case": 1, "short_chunks_lt_25ms": 0,
                        "silent_chunks": 0, "empty_output_chunks": 1,
                        "empty_output_chunk_rate": 0.5,
                    },
                    "case_ids": ["PRIVATE_ID"],
                    "transcripts": ["private text"],
                }
            },
            "private-utterance-id": {"transcripts": ["private"]},
        }
        safe = sanitize_execution_diagnostics(raw)
        rendered = json.dumps(safe)
        self.assertEqual(safe["fixed_15s"]["chunking"]["multi_chunk_cases"], 0)
        self.assertEqual(safe["fixed_15s"]["chunking"]["empty_output_chunks"], 1)
        self.assertEqual(safe["fixed_15s"]["chunking"]["empty_output_chunk_rate"], 0.5)
        self.assertNotIn("PRIVATE_ID", rendered)
        self.assertNotIn("private text", rendered)
        self.assertNotIn("case_ids", rendered)

    def test_only_aggregate_fields_are_retained(self):
        report = {
            "case_count": 3,
            "summary": {"char": {
                "cer": 0.2, "deletion_rate": 0.1, "insertion_rate": 0.05,
                "substitution_rate": 0.05, "substitutions": 1, "deletions": 2,
                "insertions": 1,
            }},
            "complete_omission": {"empty_hypothesis_count": 0},
            "near_complete_omission": {"count": 1},
            "severe_deletion": {"count": 1},
            "category_summary": {"short": {"case_count": 1}},
            "cases": [{"case_id": "private-utterance-id", "reference": "sensitive text",
                       "hypothesis": "recognized text", "char": {"deletions": 2}}],
            "transcripts": ["private text"],
            "audio_path": "/tmp/private.wav",
        }
        report["category_summary"]["private-utterance-id"] = {
            "case_count": 1, "cer": 0.9, "reference": "private text"
        }
        safe = sanitize_metrics(report)
        self.assertEqual(safe["case_count"], 3)
        self.assertEqual(safe["character_error_rate"], 0.2)
        self.assertNotIn("cases", safe)
        self.assertNotIn("transcripts", safe)
        self.assertNotIn("audio_path", safe)
        self.assertNotIn("reference", safe)
        self.assertNotIn("hypothesis", safe)
        self.assertNotIn("private-utterance-id", safe["category_summary"])
        self.assertEqual(safe["category_summary"]["short"]["case_count"], 1)
        self.assertNotIn("reference", safe["category_summary"]["short"])


    def test_metric_values_must_be_finite_numbers_not_untrusted_strings(self):
        report = {
            "case_count": "private-utterance-id",
            "summary": {"char": {
                "cer": "sensitive transcript", "deletion_rate": float("nan"),
                "insertion_rate": 0.25, "substitution_rate": float("inf"),
                "substitutions": True, "deletions": 4, "insertions": "audio.wav",
            }},
            "category_summary": {
                "short": {"case_count": 2, "cer": "private text", "deletions": 1},
                "private-utterance-id": {"case_count": 99, "cer": 1.0},
            },
            "complete_omission": {"empty_hypothesis_count": "private-id"},
        }
        safe = sanitize_metrics(report)
        self.assertIsNone(safe["case_count"])
        self.assertIsNone(safe["character_error_rate"])
        self.assertIsNone(safe["character_deletion_rate"])
        self.assertEqual(safe["character_insertion_rate"], 0.25)
        self.assertIsNone(safe["character_substitution_rate"])
        self.assertIsNone(safe["substitutions"])
        self.assertEqual(safe["deletions"], 4)
        self.assertIsNone(safe["insertions"])
        self.assertIsNone(safe["complete_omission_count"])
        self.assertEqual(safe["category_summary"]["short"]["deletions"], 1)
        self.assertIsNone(safe["category_summary"]["short"]["cer"])
        self.assertNotIn("private-utterance-id", safe["category_summary"])

    def test_paired_comparison_sanitizer_allowlists_modes_and_numeric_fields(self):
        raw = {
            "paired_comparisons": {
                "fixed_15s": {
                    "baseline": "whole", "deletion_delta": -3, "cer_delta": -0.02,
                    "cases_with_fewer_deletions": 2, "cases_with_more_deletions": 1,
                    "cases_unchanged": 4, "improved_case_ids": ["private-id"],
                },
                "private-utterance-id": {
                    "baseline": "private text", "deletion_delta": "sensitive",
                },
                "overlap_2s": {
                    "baseline": "private-id", "deletion_delta": float("nan"),
                    "cer_delta": "private transcript", "cases_unchanged": True,
                },
            }
        }
        safe = sanitize_comparisons(raw)
        self.assertEqual(set(safe), {"fixed_15s", "overlap_2s"})
        self.assertEqual(safe["fixed_15s"]["deletion_delta"], -3)
        self.assertNotIn("improved_case_ids", safe["fixed_15s"])
        self.assertIsNone(safe["overlap_2s"]["baseline"])
        self.assertIsNone(safe["overlap_2s"]["deletion_delta"])
        self.assertIsNone(safe["overlap_2s"]["cer_delta"])
        self.assertIsNone(safe["overlap_2s"]["cases_unchanged"])

    def test_comparison_sanitizer_handles_unhashable_baselines_and_extreme_numbers(self):
        raw = {
            "paired_comparisons": {
                "fixed_15s": {"baseline": ["private"], "deletion_delta": 10 ** 10000},
                "overlap_2s": {"baseline": {"private": "value"}, "cer_delta": -0.1},
            }
        }
        safe = sanitize_comparisons(raw)
        self.assertIsNone(safe["fixed_15s"]["baseline"])
        self.assertIsNone(safe["fixed_15s"]["deletion_delta"])
        self.assertIsNone(safe["overlap_2s"]["baseline"])
        self.assertEqual(safe["overlap_2s"]["cer_delta"], -0.1)

    def test_rates_reject_negative_and_out_of_range_values(self):
        report = {
            "summary": {"char": {
                "cer": -0.1, "deletion_rate": 1.1,
                "insertion_rate": 0.2, "substitution_rate": -0.01,
            }}
        }
        safe = sanitize_metrics(report)
        self.assertIsNone(safe["character_error_rate"])
        self.assertIsNone(safe["character_deletion_rate"])
        self.assertEqual(safe["character_insertion_rate"], 0.2)
        self.assertIsNone(safe["character_substitution_rate"])

    def test_dataset_workflow_logs_do_not_emit_utterance_identifiers(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("jsut", "fleurs"):
            with self.subTest(dataset=dataset):
                workflow = (root / ".github" / "workflows" / f"validate-{dataset}.yml").read_text(encoding="utf-8")
                self.assertNotIn("join(flagged)", workflow)
                self.assertNotIn("Invalid overlap chunk offsets: {", workflow)
                self.assertNotIn("Non-increasing overlap chunk offsets: {", workflow)
                self.assertNotIn("Missing overlap merge method diagnostic: {", workflow)

    def test_dataset_downloaders_suppress_exception_details(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("jsut", "fleurs"):
            with self.subTest(dataset=dataset):
                source = (root / "scripts" / f"download_{dataset}_subset.py").read_text(encoding="utf-8")
                self.assertIn(f"{dataset.upper()} dataset preparation failed", source)
                self.assertIn("except Exception", source)


    def test_dataset_workflows_upload_only_sanitized_aggregate_reports(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("jsut", "fleurs"):
            with self.subTest(dataset=dataset):
                workflow = (root / ".github" / "workflows" / f"validate-{dataset}.yml").read_text(encoding="utf-8")
                self.assertIn("sanitize_asr_report.py", workflow)
                self.assertIn("validate_asr_evaluation.py", workflow)
                self.assertIn("--output-json", workflow)
                self.assertIn("--diagnostics execution_diagnostics.json", workflow)
                self.assertIn("execution_diagnostics.json", workflow)
                self.assertNotIn("Fail on complete, near-complete, or severe deletion", workflow)
                self.assertIn("--report whole=metrics.json", workflow)
                self.assertIn("if: success()", workflow)
                self.assertIn("public_metrics.json", workflow)
                self.assertIn("workflow_dispatch:", workflow)
                self.assertIn("acknowledge_license", workflow)
                self.assertIn('default: "no"', workflow)
                self.assertIn(f"rm -rf tests/data/{dataset}_ja/clips", workflow)
                self.assertIn(f"rm -f tests/data/{dataset}_ja/manifest.json", workflow)
                self.assertIn("transcripts*.tsv", workflow)
                self.assertIn("metrics*.json", workflow)
                self.assertIn("mode_comparison.json", workflow)
                artifact = workflow.split("uses: actions/upload-artifact@v4", 1)[1]
                artifact = artifact.split("retention-days:", 1)[0]
                self.assertIn(f"path: tests/data/{dataset}_ja/public_metrics.json", artifact)
                self.assertNotIn("mode_comparison.json", artifact)
                self.assertNotIn("transcripts", artifact)
                self.assertNotIn("manifest.json", artifact)

    def test_rms_normalized_mode_is_allowed_in_paired_comparisons(self):
        safe = sanitize_comparisons({
            "paired_comparisons": {
                "rms_normalized": {
                    "baseline": "whole", "deletion_delta": -2, "cer_delta": 0.01,
                    "cases_with_fewer_deletions": 3, "cases_with_more_deletions": 1,
                    "cases_unchanged": 6,
                }
            }
        })
        self.assertEqual(safe["rms_normalized"]["baseline"], "whole")
        self.assertEqual(safe["rms_normalized"]["deletion_delta"], -2)

    def test_cli_sanitizes_optional_rms_normalized_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {
                "case_count": 2,
                "summary": {"char": {"cer": 0.2, "deletion_rate": 0.1}},
                "complete_omission": {"empty_hypothesis_count": 0},
                "near_complete_omission": {"count": 1},
                "severe_deletion": {"count": 0},
                "cases": [{"case_id": "PRIVATE_ID", "reference": "private transcript"}],
            }
            names = ("metrics.json", "metrics_chunked_15s.json",
                     "metrics_silence_aware_15s.json", "metrics_overlap_2s.json",
                     "normalized_metrics.json")
            for name in names:
                (root / name).write_text(json.dumps(report), encoding="utf-8")
            (root / "mode_comparison.json").write_text(json.dumps({
                "paired_comparisons": {
                    "rms_normalized": {"baseline": "whole", "deletion_delta": -2}
                }
            }), encoding="utf-8")
            output = root / "public_metrics.json"
            argv = [
                "sanitize_asr_report.py", "--root", str(root),
                "--comparison", "mode_comparison.json",
                "--extra-report", "rms_normalized=normalized_metrics.json",
                "--output", str(output),
            ]
            stdout = io.StringIO()
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                self.assertEqual(main(), 0)
            rendered = output.read_text(encoding="utf-8")
            safe = json.loads(rendered)
            self.assertIn("rms_normalized", safe["modes"])
            self.assertEqual(safe["paired_comparisons"]["rms_normalized"]["deletion_delta"], -2)
            self.assertNotIn("PRIVATE_ID", rendered)
            self.assertNotIn("private transcript", rendered)

    def test_cli_accepts_explicit_report_path_mappings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {"case_count": 2, "summary": {"char": {"cer": 0.3, "deletion_rate": 0.2}},
                      "complete_omission": {"empty_hypothesis_count": 0},
                      "near_complete_omission": {"count": 1}, "severe_deletion": {"count": 1}}
            mapping = {"whole": "metrics.json", "fixed_15s": "chunked_15s_metrics.json",
                       "silence_aware_15s": "silence_aware_15s_metrics.json",
                       "overlap_2s": "overlap_2s_metrics.json", "rms_normalized": "normalized_metrics.json"}
            for filename in mapping.values():
                (root / filename).write_text(json.dumps(report), encoding="utf-8")
            (root / "mode_comparison.json").write_text(json.dumps({"paired_comparisons": {}}), encoding="utf-8")
            output = root / "public_metrics.json"
            argv = ["sanitize_asr_report.py", "--root", str(root), "--comparison", "mode_comparison.json", "--output", str(output)]
            for mode, filename in mapping.items():
                argv.extend(["--report", f"{mode}={filename}"])
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            safe = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(safe["modes"]), set(mapping))

    def test_explicit_report_mapping_requires_all_core_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["sanitize_asr_report.py", "--root", str(root), "--output", str(root / "public_metrics.json"),
                    "--report", "whole=metrics.json"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)

    def test_common_voice_workflow_uploads_only_sanitized_aggregate_report(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "acquire-and-measure-baseline.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("\n  push:\n", workflow)
        self.assertIn("sanitize_asr_report.py", workflow)
        self.assertIn("--report whole=metrics.json", workflow)
        self.assertIn("--report fixed_15s=chunked_15s_metrics.json", workflow)
        self.assertIn("--report silence_aware_15s=silence_aware_15s_metrics.json", workflow)
        self.assertIn("--report overlap_2s=overlap_2s_metrics.json", workflow)
        self.assertIn("--report rms_normalized=normalized_metrics.json", workflow)
        self.assertIn("if: success()", workflow)
        self.assertIn("public_metrics.json", workflow)
        self.assertIn("rm -rf tests/data/common_voice_ja/clips", workflow)
        self.assertIn("rm -rf downloads/mdc corpus", workflow)
        self.assertIn("rm -f tests/data/common_voice_ja/transcripts*.tsv", workflow)
        self.assertIn("rm -f tests/data/common_voice_ja/*_metrics.json", workflow)
        self.assertIn("rm -f tests/data/common_voice_ja/mode_comparison.json", workflow)
        artifact = workflow.split("uses: actions/upload-artifact@v4", 1)[1]
        artifact = artifact.split("retention-days:", 1)[0]
        self.assertIn("path: tests/data/common_voice_ja/public_metrics.json", artifact)
        self.assertNotIn("transcripts", artifact)
        self.assertNotIn("manifest.json", artifact)
        self.assertNotIn("mode_comparison.json", artifact)
        self.assertNotIn("row['case_id']", workflow)


    def test_decoder_stage_diagnostics_are_allowlisted_and_aggregate_only(self):
        safe = sanitize_execution_diagnostics({"modes": {"whole": {
            "case_count": 2,
            "audio_duration_seconds": {"min": 1, "median": 2, "max": 3},
            "chunking": {"single_chunk_cases": 2, "multi_chunk_cases": 0,
                "multi_chunk_case_rate": 0, "total_chunks": 2, "max_chunks_per_case": 1,
                "short_chunks_lt_25ms": 0, "silent_chunks": 0, "empty_output_chunks": 0,
                "empty_output_chunk_rate": 0},
            "decoder_stages": {"total_raw_tokens": 10, "total_timestamps": 10,
                "total_raw_decoder_text_chars": 22, "total_chunk_text_chars": 20,
                "total_merged_text_chars": 19, "total_final_text_chars": 18,
                "chunk_merge_char_delta": -1, "final_normalization_char_delta": -1,
                "cases_shorter_after_merge": 1, "cases_shorter_after_normalization": 1,
                "cases_token_timestamp_count_mismatch": 0,
                "mean_token_density_per_second": 5, "total_decode_seconds": 0.25,
                "private_tokens": ["secret"], "transcript": "secret text"}
        }}})
        stages = safe["whole"]["decoder_stages"]
        self.assertEqual(stages["total_raw_tokens"], 10)
        self.assertEqual(stages["chunk_merge_char_delta"], -1)
        self.assertEqual(stages["final_normalization_char_delta"], -1)
        self.assertEqual(stages["cases_shorter_after_merge"], 1)
        self.assertNotIn("private_tokens", stages)
        self.assertNotIn("transcript", stages)

    def test_silence_aware_comparison_alias_is_exported_canonically(self):
        safe = sanitize_comparisons({
            "paired_comparisons": {
                "silence_aware": {
                    "baseline": "whole", "deletion_delta": -3, "cer_delta": -0.01,
                    "cases_with_fewer_deletions": 2, "cases_with_more_deletions": 0,
                    "cases_unchanged": 8,
                }
            }
        })
        self.assertIn("silence_aware_15s", safe)
        self.assertNotIn("silence_aware", safe)
        self.assertEqual(safe["silence_aware_15s"]["deletion_delta"], -3)


if __name__ == "__main__":
    unittest.main()
