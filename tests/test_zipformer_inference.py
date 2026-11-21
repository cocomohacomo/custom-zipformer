from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

from scripts.run_sherpa_zipformer import prepare_inference_audio, decode_audio_chunks


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

    def test_resampling_calls_polyphase_converter_with_expected_ratio(self):
        source = np.sin(np.linspace(0, 10 * np.pi, 4410, dtype=np.float32))
        scipy_module = types.ModuleType("scipy")
        signal_module = types.ModuleType("scipy.signal")
        calls = []
        def fake_resample_poly(values, up, down):
            calls.append((up, down))
            count = (len(values) * up + down - 1) // down
            return np.zeros(count, dtype=np.float32)
        signal_module.resample_poly = fake_resample_poly
        scipy_module.signal = signal_module
        old_modules = {key: sys.modules.get(key) for key in ("scipy", "scipy.signal")}
        try:
            sys.modules["scipy"] = scipy_module
            sys.modules["scipy.signal"] = signal_module
            prepared = prepare_inference_audio(source, 44100, 16000)
        finally:
            for key, old_module in old_modules.items():
                if old_module is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = old_module
        self.assertEqual(calls, [(160, 441)])
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


    def test_offline_decoder_processes_every_chunk_including_final_partial(self):
        class FakeStream:
            def __init__(self):
                self.audio = None
                self.result = None
            def accept_waveform(self, sample_rate, audio):
                self.audio = np.asarray(audio).copy()
                self.sample_rate = sample_rate

        class FakeRecognizer:
            def __init__(self):
                self.decoded = []
            def create_stream(self):
                return FakeStream()
            def decode_stream(self, stream):
                self.decoded.append((stream.sample_rate, stream.audio.copy()))
                stream.result = types.SimpleNamespace(
                    text=str(len(stream.audio)), tokens=[], timestamps=[]
                )

        recognizer = FakeRecognizer()
        chunks = [np.ones(16000), np.ones(16000), np.ones(37)]
        results, durations = decode_audio_chunks(recognizer, chunks)
        self.assertEqual([len(audio) for _, audio in recognizer.decoded], [16000, 16000, 37])
        self.assertEqual([rate for rate, _ in recognizer.decoded], [16000, 16000, 16000])
        self.assertEqual([item["text"] for item in results], ["16000", "16000", "37"])
        self.assertEqual(durations, [1.0, 1.0, 37 / 16000])


    def test_offline_decoder_keeps_sub_frame_and_silent_chunks(self):
        class FakeStream:
            def accept_waveform(self, sample_rate, audio):
                self.audio = np.asarray(audio).copy()
                self.sample_rate = sample_rate
                self.result = None

        class FakeRecognizer:
            def __init__(self):
                self.received = []
            def create_stream(self):
                return FakeStream()
            def decode_stream(self, stream):
                self.received.append((stream.sample_rate, stream.audio.copy()))
                stream.result = types.SimpleNamespace(text="", tokens=[], timestamps=[])

        recognizer = FakeRecognizer()
        tiny = np.array([0.0], dtype=np.float32)
        silent = np.zeros(400, dtype=np.float32)
        results, durations = decode_audio_chunks(recognizer, [tiny, silent])
        self.assertEqual([len(audio) for _, audio in recognizer.received], [1, 400])
        self.assertTrue(all(rate == 16000 for rate, _ in recognizer.received))
        self.assertEqual(results, [
            {"text": "", "tokens": [], "timestamps": []},
            {"text": "", "tokens": [], "timestamps": []},
        ])
        self.assertEqual(durations, [1 / 16000, 0.025])

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
