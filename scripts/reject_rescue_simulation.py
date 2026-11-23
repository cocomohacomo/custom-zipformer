"""Offline, aggregate-only shadow simulation for rejected-token rescue.

This module is deliberately independent of the production merge path. It accepts
candidate correspondence metadata and reports what could be rescued under a
strict one-to-one rule. It never changes hypotheses or emits token text.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


def simulate_reject_rescue(
    rejected: Iterable[Mapping],
    thresholds_ms: tuple[int, ...] = (100, 250, 500),
) -> dict:
    """Classify rejected tokens for each threshold using strict 1:1 matching.

    Each rejected record has:
      - rejected_id: unique opaque identifier within this call
      - token_chars: non-negative character count
      - candidates: {threshold_ms: [opaque kept-token ids]}

    A rejected token is rescued only when it has exactly one candidate and that
    kept token is referenced by exactly one rejected token. No best-match
    assignment is performed: ambiguity and candidate reuse are conservative
    non-rescue outcomes.
    """
    records = list(rejected)
    ids = [record.get("rejected_id") for record in records]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("rejected_id must be a non-empty string")
    if len(set(ids)) != len(ids):
        raise ValueError("rejected_id values must be unique")
    for record in records:
        chars = record.get("token_chars")
        if not isinstance(chars, int) or isinstance(chars, bool) or chars < 0:
            raise ValueError("token_chars must be a non-negative integer")
        if not isinstance(record.get("candidates", {}), Mapping):
            raise ValueError("candidates must be a threshold mapping")

    results = {}
    for threshold in thresholds_ms:
        if not isinstance(threshold, int) or threshold <= 0:
            raise ValueError("thresholds_ms must contain positive integers")
        candidate_lists = []
        reverse = Counter()
        for record in records:
            raw = record.get("candidates", {}).get(threshold, [])
            if not isinstance(raw, (list, tuple)):
                raise ValueError("candidate lists must be lists or tuples")
            if any(not isinstance(candidate, str) or not candidate for candidate in raw):
                raise ValueError("candidate ids must be non-empty strings")
            if len(set(raw)) != len(raw):
                raise ValueError("candidate ids within a rejected record must be unique")
            candidate_lists.append(list(raw))
            reverse.update(raw)

        counts = Counter()
        rescued_chars = 0
        rescued_ids = []
        for record, candidates in zip(records, candidate_lists):
            if not candidates:
                status = "no_candidate"
            elif len(candidates) > 1:
                status = "ambiguous"
            elif reverse[candidates[0]] > 1:
                status = "candidate_collision"
            else:
                status = "rescued_unique"
                rescued_chars += record["token_chars"]
                rescued_ids.append(record["rejected_id"])
            counts[status] += 1

        total = len(records)
        rescued = counts["rescued_unique"]
        results[threshold] = {
            "rejected_tokens": total,
            "rescued_tokens": rescued,
            "rescued_token_chars": rescued_chars,
            "not_rescued_tokens": total - rescued,
            "classification": {
                key: counts[key]
                for key in (
                    "rescued_unique", "no_candidate", "ambiguous",
                    "candidate_collision",
                )
            },
            "accounting_ok": sum(counts.values()) == total,
            # Opaque identifiers are intentionally not returned.
        }
    return results
