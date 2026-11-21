from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.run_sherpa_zipformer import prepare_inference_audio


class ZipformerInferenceSourceTests(unittest.TestCase):
    def test_antiphase_stereo_does_not_cancel_to_silence(self):
        signal = np.sin(np.linspace(0, 20 * np.pi, 8000, dtype=np.float32)) * 0.2
        stereo = np.column_stack((signal, -signal))
        prepared = prepare_inference_audio(stereo, 16000)
        self.assertGreater(float(np.sqrt(np.mean(prepared ** 2))), 0.1)
        np.testing.assert_allclose(prepared, signal, atol=1e-6)

    def test_normal_stereo_is_averaged(self):
        left = np.full(8000, 0.2, dtype=np.float32)
        right = np.full(8000, 0.4, dtype=np.float32)
        prepared = prepare_inference_audio(np.column_stack((left, right)), 16000)
        np.testing.assert_allclose(prepared, 0.3, atol=1e-6)

    def test_resampling_produces_expected_frame_count(self):
        source = np.sin(np.linspace(0, 10 * np.pi, 4410, dtype=np.float32))
        prepared = prepare_inference_audio(source, 44100, 16000)
        self.assertEqual(len(prepared), 1600)
        self.assertEqual(prepared.dtype, np.float32)
        self.assertTrue(np.isfinite(prepared).all())

    def test_audio_conversion_rejects_non_finite_samples(self):
        source = np.array([0.0, np.nan, 0.0], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "finite"):
            prepare_inference_audio(source, 16000)

    def test_audio_conversion_rejects_invalid_sample_rate_and_empty_audio(self):
        with self.assertRaisesRegex(ValueError, "sample rates"):
            prepare_inference_audio(np.ones(10, dtype=np.float32), 0)
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            prepare_inference_audio(np.array([], dtype=np.float32), 16000)

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
