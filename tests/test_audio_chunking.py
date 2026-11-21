from __future__ import annotations

import unittest
import numpy as np

from scripts.audio_chunking import merge_overlapping_text, merge_timestamped_chunks, split_at_low_energy, split_with_overlap


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

    def test_overlap_chunks_cover_entire_signal_without_gaps(self):
        for sample_rate in (8000, 16000, 44100):
            for sample_count in (1, 17, 159999, 480001):
                x = np.arange(sample_count, dtype=np.float32)
                chunks, later_starts = split_with_overlap(
                    x, sample_rate, chunk_seconds=3.7, overlap_seconds=0.41
                )
                starts = [0, *[round(value * sample_rate) for value in later_starts]]
                self.assertEqual(len(chunks), len(starts))
                self.assertEqual(starts[0], 0)
                self.assertEqual(starts[-1] + len(chunks[-1]), sample_count)
                for index in range(1, len(chunks)):
                    previous_end = starts[index - 1] + len(chunks[index - 1])
                    self.assertLessEqual(starts[index], previous_end)
                    self.assertGreater(starts[index], starts[index - 1])
                    overlap = previous_end - starts[index]
                    if overlap:
                        np.testing.assert_array_equal(
                            chunks[index - 1][-overlap:], chunks[index][:overlap]
                        )

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


    def test_timestamp_merge_assigns_overlap_to_midpoint(self):
        results = [
            {"text": "abcdef", "tokens": list("abcdef"), "timestamps": [0, 1, 2, 3, 4, 5]},
            {"text": "defghi", "tokens": list("defghi"), "timestamps": [0, 0.2, 1, 2, 3, 4]},
        ]
        merged, method = merge_timestamped_chunks(results, [0.0, 4.0], [6.0, 6.0])
        self.assertEqual(merged, "abcdefghi")
        self.assertEqual(method, "timestamp")

    def test_timestamp_merge_falls_back_when_tokens_do_not_reconstruct_text(self):
        results = [
            {"text": "今日はとても良い天気", "tokens": ["今日は", "天気"], "timestamps": [0, 1]},
            {"text": "とても良い天気ですね", "tokens": ["天気", "ですね"], "timestamps": [0, 1]},
        ]
        merged, method = merge_timestamped_chunks(results, [0.0, 4.0], [6.0, 6.0])
        self.assertEqual(merged, "今日はとても良い天気とても良い天気ですね")
        self.assertEqual(method, "text_fallback")


    def test_timestamp_fallback_preserves_a_genuinely_repeated_phrase(self):
        phrase = "同じ内容をもう一度話す"
        results = [
            {"text": phrase, "tokens": ["不一致"], "timestamps": [0]},
            {"text": phrase, "tokens": ["不一致"], "timestamps": [0]},
        ]
        merged, method = merge_timestamped_chunks(results, [0.0, 2.0], [3.0, 3.0])
        self.assertEqual(merged, phrase + phrase)
        self.assertEqual(method, "text_fallback")

    def test_timestamp_merge_falls_back_on_non_finite_timestamp(self):
        results = [
            {"text": "abc", "tokens": list("abc"), "timestamps": [0, float("nan"), 2]},
        ]
        merged, method = merge_timestamped_chunks(results, [0.0], [3.0])
        self.assertEqual(merged, "abc")
        self.assertEqual(method, "text_fallback")

    def test_timestamp_merge_falls_back_on_non_monotonic_chunk_starts(self):
        results = [
            {"text": "abc", "tokens": list("abc"), "timestamps": [0, 1, 2]},
            {"text": "def", "tokens": list("def"), "timestamps": [0, 1, 2]},
        ]
        merged, method = merge_timestamped_chunks(results, [2.0, 1.0], [3.0, 3.0])
        self.assertEqual(merged, "abcdef")
        self.assertEqual(method, "text_fallback")

    def test_timestamp_merge_falls_back_on_non_positive_duration(self):
        results = [
            {"text": "abc", "tokens": list("abc"), "timestamps": [0, 1, 2]},
        ]
        merged, method = merge_timestamped_chunks(results, [0.0], [0.0])
        self.assertEqual(merged, "abc")
        self.assertEqual(method, "text_fallback")

    def test_timestamp_merge_falls_back_on_non_monotonic_times(self):
        results = [
            {"text": "abc", "tokens": list("abc"), "timestamps": [0, 2, 1]},
        ]
        merged, method = merge_timestamped_chunks(results, [0.0], [3.0])
        self.assertEqual(merged, "abc")
        self.assertEqual(method, "text_fallback")


if __name__ == "__main__":
    unittest.main()
