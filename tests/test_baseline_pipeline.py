from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_common_voice_baseline import main as prepare_main
from scripts.run_baseline_asr import main as runner_main


class BaselinePreparationTests(unittest.TestCase):
    def test_stratified_sampling_is_deterministic_and_metric_ready(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "corpus"
            clips = root / "clips"
            clips.mkdir(parents=True)
            rows = []
            for i, sentence in enumerate([
                "短文", "これは中くらいの文章です", "これはベースライン用の長い文章で十分な長さがあります"
            ] * 4):
                name = f"{i}.mp3"
                (clips / name).write_bytes(f"audio-{i}".encode())
                rows.append({"path": name, "sentence": sentence})
            with (root / "test.tsv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "sentence"], delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)

            outputs = [Path(td) / "out1", Path(td) / "out2"]
            for out in outputs:
                sys.argv = ["prepare", str(root), "--limit", "6", "--seed", "12", "--output", str(out)]
                self.assertEqual(prepare_main(), 0)

            m1 = json.loads((outputs[0] / "manifest.json").read_text(encoding="utf-8"))
            m2 = json.loads((outputs[1] / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(m1["manifest"], m2["manifest"])
            self.assertEqual(m1["copied"], 6)
            with (outputs[0] / "transcripts.tsv").open(encoding="utf-8") as f:
                data = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(len(data), 6)
            self.assertTrue(all(row["hypothesis"] == "" for row in data))
            self.assertTrue(all(len(item["audio_sha256"]) == 64 for item in m1["manifest"]))


class BaselineRunnerTests(unittest.TestCase):
    def test_runner_captures_transcript_and_records_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "clip.mp3"
            audio.write_bytes(b"fixture")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"manifest": [{
                "case_id": "case-1", "category": "short",
                "audio": "clip.mp3", "reference": "猫がいる"
            }]}), encoding="utf-8")
            output = root / "result.tsv"
            report = root / "run.json"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "runner", str(manifest), "--command",
                    f"{sys.executable} -c 'print(\"猫がいる\")' {{audio}}",
                    "--output-tsv", str(output), "--report-json", str(report)
                ]
                self.assertEqual(runner_main(), 0)
            finally:
                sys.argv = old_argv
            with output.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(rows[0]["hypothesis"], "猫がいる")
            self.assertEqual(json.loads(report.read_text())["completed"], 1)

    def test_runner_rejects_missing_audio_placeholder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.json"
            manifest.write_text('{"manifest":[]}', encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = ["runner", str(manifest), "--command", "echo hi",
                            "--output-tsv", str(root / "out.tsv")]
                with self.assertRaises(SystemExit) as ctx:
                    runner_main()
                self.assertEqual(ctx.exception.code, 2)
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
