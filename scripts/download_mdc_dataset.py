"""Retryable Mozilla Data Collective dataset acquisition."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Any


def download_with_retries(
    download_fn: Callable[..., Any],
    dataset_id: str,
    download_directory: str,
    *,
    max_attempts: int = 5,
    base_delay_seconds: int = 60,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Retry MDC rate-limit failures with exponential backoff.

    Only the explicit RateLimitError is retried. Authentication, permission,
    missing-dataset, and other errors fail immediately rather than being
    hidden behind repeated requests.
    """
    if max_attempts < 1 or base_delay_seconds < 0:
        raise ValueError("max_attempts must be >= 1 and base_delay_seconds >= 0")

    for attempt in range(1, max_attempts + 1):
        try:
            return download_fn(
                dataset_id,
                download_directory=download_directory,
                extract=True,
                show_progress=False,
                enable_logging=False,
            )
        except Exception as exc:
            if type(exc).__name__ != "RateLimitError" or attempt == max_attempts:
                raise
            delay = base_delay_seconds * (2 ** (attempt - 1))
            print(
                f"MDC rate limit encountered (attempt {attempt}/{max_attempts}); "
                f"retrying in {delay} seconds."
            )
            sleep(delay)


def main() -> int:
    from datacollective import download_dataset

    dataset_id = os.environ["COMMON_VOICE_JA_DATASET_ID"]
    download_directory = os.environ["MDC_DOWNLOAD_PATH"]
    Path(download_directory).mkdir(parents=True, exist_ok=True)
    print(f"Requesting authorized Common Voice dataset: {dataset_id}")
    path = download_with_retries(download_dataset, dataset_id, download_directory)
    print(f"Dataset downloaded and extracted to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
