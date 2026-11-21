from __future__ import annotations

import unittest
import numpy as np

from scripts.audio_chunking import split_at_low_energy


class SilenceAwareChunkingTests(unittest.TestCase):
    def test_moves_boundary_to_nearby_silence_without_losing_samples(self):
        sr = 1000
        x = np.full(30000, 0.2, dtype=np.float32)
        x[14500:15500] = 0.0
        chunks, boundaries = split_at_low_energy(
            x, sr, target_seconds=15, search_window_seconds=2,
            frame_seconds=0.02, min_chunk_seconds=3,
        )
        self.assertEqual(len(chunks), 2)
        self.assertTrue(14.5 <= boundaries[0] <= 15.5)
        np.testing.assert_array_equal(np.concatenate(chunks), x)

    def test_short_audio_is_one_unchanged_chunk(self):
        x = np.arange(1000, dtype=np.float32)
        chunks, boundaries = split_at_low_energy(x, 1000, 15)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(boundaries, [])
        np.testing.assert_array_equal(chunks[0], x)

    def test_empty_audio_returns_no_chunks(self):
        chunks, boundaries = split_at_low_energy(np.array([], dtype=np.float32), 16000, 15)
        self.assertEqual(chunks, [])
        self.assertEqual(boundaries, [])

    def test_rejects_stereo_input(self):
        with self.assertRaises(ValueError):
            split_at_low_energy(np.zeros((100, 2)), 16000, 15)


if __name__ == "__main__":
    unittest.main()
