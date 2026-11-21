from __future__ import annotations

import unittest

from scripts.download_mdc_dataset import download_with_retries


class RateLimitError(Exception):
    pass


class DownloadRetryTests(unittest.TestCase):
    def test_retries_rate_limit_with_exponential_backoff(self):
        calls = []
        delays = []

        def download(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) < 3:
                raise RateLimitError("slow down")
            return "/tmp/dataset"

        result = download_with_retries(
            download, "dataset-id", "/tmp/mdc",
            max_attempts=4, base_delay_seconds=2, sleep=delays.append,
        )
        self.assertEqual(result, "/tmp/dataset")
        self.assertEqual(delays, [2, 4])
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][1]["extract"], True)

    def test_stops_after_max_attempts(self):
        calls = []
        delays = []

        def download(*args, **kwargs):
            calls.append(1)
            raise RateLimitError("slow down")

        with self.assertRaises(RateLimitError):
            download_with_retries(
                download, "dataset-id", "/tmp/mdc",
                max_attempts=3, base_delay_seconds=1, sleep=delays.append,
            )
        self.assertEqual(len(calls), 3)
        self.assertEqual(delays, [1, 2])

    def test_does_not_retry_non_rate_limit_errors(self):
        calls = []
        delays = []

        def download(*args, **kwargs):
            calls.append(1)
            raise PermissionError("denied")

        with self.assertRaises(PermissionError):
            download_with_retries(
                download, "dataset-id", "/tmp/mdc",
                max_attempts=5, sleep=delays.append,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(delays, [])


if __name__ == "__main__":
    unittest.main()
