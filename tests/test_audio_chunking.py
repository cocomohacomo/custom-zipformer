from __future__ import annotations

import unittest
import numpy as np

from scripts.audio_chunking import merge_overlapping_text, split_at_low_energy, split_with_overlap


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


class OverlapChunkingTests(unittest.TestCase):
    def test_chunks_share_waveform_context(self):
        x = np.arange(10000, dtype=np.float32)
        chunks, starts = split_with_overlap(x, 1000, 4, 1)
        self.assertEqual(starts, [3.0, 6.0])
        self.assertEqual(len(chunks), 3)
        np.testing.assert_array_equal(chunks[0][-1000:], chunks[1][:1000])

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            split_with_overlap(np.zeros(10), 10, 2, 2)

    def test_exact_text_overlap_is_removed(self):
        self.assertEqual(merge_overlapping_text(["今日はとても晴れ", "とても晴れですね"]), "今日はとても晴れですね")

    def test_short_prefix_is_preserved(self):
        self.assertEqual(merge_overlapping_text(["東京", "京都"]), "東京京都")

    def test_two_character_coincidental_suffix_is_preserved(self):
        self.assertEqual(merge_overlapping_text(["今日は天気", "天気予報です"]), "今日は天気天気予報です")

    def test_substantial_exact_overlap_is_removed(self):
        self.assertEqual(merge_overlapping_text(["今日はとても良い天気", "とても良い天気ですね"]), "今日はとても良い天気ですね")

    def test_custom_minimum_overlap_is_honored(self):
        self.assertEqual(merge_overlapping_text(["abcdef", "defghi"], min_overlap_chars=3), "abcdefghi")


if __name__ == "__main__":
    unittest.main()
