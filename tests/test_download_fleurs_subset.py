from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import download_fleurs_subset as fleurs


class FleursSubsetTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(fleurs.classify("あ" * 30), "short")
        self.assertEqual(fleurs.classify("あ" * 31), "medium")
        self.assertEqual(fleurs.classify("あ" * 71), "long")

    def test_read_rows_rejects_duplicate_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.tsv"
            path.write_text(
                "id\tfile_name\ttranscription\n"
                "1\tclip.wav\tこんにちは\n"
                "2\tclip.wav\tこんばんは\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                fleurs.read_rows(path)

    def test_duplicate_filename_error_does_not_echo_source_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.tsv"
            path.write_text(
                "id\tfile_name\ttranscription\n"
                "1\tPRIVATE_FLEURS_FILE.wav\tこんにちは\n"
                "2\tPRIVATE_FLEURS_FILE.wav\tこんばんは\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as caught:
                fleurs.read_rows(path)
            self.assertNotIn("PRIVATE_FLEURS_FILE.wav", str(caught.exception))


    def test_main_suppresses_downloader_exception_details(self):
        output = io.StringIO()
        def noisy_failure(*args, **kwargs):
            print("PRIVATE_FLEURS_ID transcript.wav")
            print("PRIVATE_FLEURS_ID transcript.wav", file=sys.stderr)
            raise RuntimeError("PRIVATE_FLEURS_ID transcript.wav")
        with patch.object(sys, "argv", ["download_fleurs_subset.py", "--limit", "1"]), \
             patch.object(fleurs, "prepare_dataset", side_effect=noisy_failure), \
             contextlib.redirect_stdout(output):
            result = fleurs.main()
        self.assertEqual(result, 1)
        self.assertNotIn("PRIVATE_FLEURS_ID", output.getvalue())
        self.assertNotIn("transcript.wav", output.getvalue())

    def test_prepare_dataset_extracts_only_selected_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tsv = root / "dev.tsv"
            archive_path = root / "dev.tar.gz"
            with tsv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["id", "file_name", "transcription"])
                writer.writerow(["1", "a.wav", "短い文章です"])
                writer.writerow(["2", "b.wav", "これは中くらいの長さの日本語文章を使った検証用サンプルです"])
                writer.writerow(["3", "c.wav", "これは長い文章のサンプルです。" * 5])
            with tarfile.open(archive_path, "w:gz") as archive:
                for name in ("a.wav", "b.wav", "c.wav", "unused.wav"):
                    source = root / name
                    source.write_bytes((name + "-audio").encode())
                    archive.add(source, arcname=f"audio/{name}")

            def fake_download(*, repo_id, repo_type, filename):
                self.assertEqual(repo_id, fleurs.REPO_ID)
                self.assertEqual(repo_type, "dataset")
                return str(tsv if filename == fleurs.TRANSCRIPT_FILE else archive_path)

            fake_module = SimpleNamespace(hf_hub_download=fake_download)
            output = root / "out"
            with patch.dict(sys.modules, {"huggingface_hub": fake_module}):
                result = fleurs.prepare_dataset(3, output, seed=7)

            self.assertEqual(result["copied"], 3)
            self.assertEqual(result["license"], "CC-BY-4.0")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["manifest"]), 3)
            self.assertTrue(all((output / item["audio"]).is_file() for item in manifest["manifest"]))
            self.assertFalse((output / "clips" / "unused.wav").exists())


if __name__ == "__main__":
    unittest.main()
