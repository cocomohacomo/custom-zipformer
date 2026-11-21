from __future__ import annotations

import unittest
from pathlib import Path


class ZipformerInferenceSourceTests(unittest.TestCase):
    def test_inference_runner_is_valid_python(self):
        source = Path("scripts/run_sherpa_zipformer.py").read_text(encoding="utf-8")
        compile(source, "scripts/run_sherpa_zipformer.py", "exec")

    def test_overlap_inference_uses_waveform_overlap_helper(self):
        source = Path("scripts/run_sherpa_zipformer.py").read_text(encoding="utf-8")
        self.assertIn("split_with_overlap(", source)
        self.assertIn("chunks, overlap_starts = split_with_overlap(", source)
        self.assertIn("merge_timestamped_chunks(", source)
        self.assertIn("overlap_merge_method", source)


if __name__ == "__main__":
    unittest.main()
