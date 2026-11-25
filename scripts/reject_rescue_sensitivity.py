from collections import Counter


def _edit_counts(reference, hypothesis):
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    dp = [[0] * cols for _ in range(rows)]
    back = [[None] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0], back[i][0] = i, "del"
    for j in range(1, cols):
        dp[0][j], back[0][j] = j, "ins"
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j], back[i][j] = dp[i - 1][j - 1], "match"
            else:
                choices = (
                    (dp[i - 1][j - 1] + 1, "sub"),
                    (dp[i - 1][j] + 1, "del"),
                    (dp[i][j - 1] + 1, "ins"),
                )
                dp[i][j], back[i][j] = min(choices)
    sub = delete = insert = 0
    i, j = len(reference), len(hypothesis)
    while i or j:
        op = back[i][j]
        if op == "match":
            i -= 1
            j -= 1
        elif op == "sub":
            sub += 1
            i -= 1
            j -= 1
        elif op == "del":
            delete += 1
            i -= 1
        elif op == "ins":
            insert += 1
            j -= 1
        else:
            raise AssertionError("invalid traceback")
    return sub, delete, insert


def _metrics(reference, hypothesis):
    sub, delete, insert = _edit_counts(reference, hypothesis)
    denominator = max(1, len(reference))
    return {
        "cer": (sub + delete + insert) / denominator,
        "substitutions": sub,
        "deletions": delete,
        "insertions": insert,
        "errors": sub + delete + insert,
        "reference_chars": len(reference),
        "hypothesis_chars": len(hypothesis),
    }


def counterfactual_rescue_cer(reference, kept_tokens, rejected_tokens, thresholds_ms=(100, 250, 500)):
    """Return aggregate-only CER for strict, collision-free unique rescue candidates.

    Each token has token_id, text, time. Rejected tokens additionally have
    candidates: {threshold_ms: [kept_token_id, ...]}. Input records are
    transient local data and are never copied into the returned result.
    """
    if not isinstance(reference, str):
        raise ValueError("reference must be a string")
    kept, rejected = list(kept_tokens), list(rejected_tokens)
    all_tokens = kept + rejected
    ids = [item.get("token_id") for item in all_tokens]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("token IDs must be unique non-empty strings")
    kept_ids = {item["token_id"] for item in kept}
    for item in all_tokens:
        if not isinstance(item.get("text"), str):
            raise ValueError("token text must be a string")
        if isinstance(item.get("time"), bool) or not isinstance(item.get("time"), (int, float)):
            raise ValueError("token time must be numeric")
    baseline = "".join(item["text"] for item in sorted(kept, key=lambda x: (x["time"], x["token_id"])))
    result = {"baseline": _metrics(reference, baseline), "thresholds": {}}
    for threshold in thresholds_ms:
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
            raise ValueError("threshold must be a positive integer")
        candidates = []
        reverse = Counter()
        for item in rejected:
            raw = item.get("candidates", {}).get(threshold, [])
            if not isinstance(raw, (list, tuple)) or len(raw) != len(set(raw)):
                raise ValueError("candidate list is invalid")
            if any(candidate not in kept_ids for candidate in raw):
                raise ValueError("candidate does not identify a kept token")
            candidates.append(list(raw))
            reverse.update(raw)
        rescued, classes = [], Counter()
        for item, choices in zip(rejected, candidates):
            if not choices:
                classes["no_candidate"] += 1
            elif len(choices) > 1:
                classes["ambiguous"] += 1
            elif reverse[choices[0]] > 1:
                classes["candidate_collision"] += 1
            else:
                classes["rescued_unique"] += 1
                rescued.append(item)
        hypothesis = "".join(
            item["text"] for item in sorted(kept + rescued, key=lambda x: (x["time"], x["token_id"]))
        )
        result["thresholds"][str(threshold)] = {
            **_metrics(reference, hypothesis),
            "rejected_tokens": len(rejected),
            "rescued_tokens": len(rescued),
            "not_rescued_tokens": len(rejected) - len(rescued),
            "classification": {
                key: classes[key] for key in
                ("rescued_unique", "no_candidate", "ambiguous", "candidate_collision")
            },
        }
    return result


def compare_baseline_and_gate(existing_hypothesis, kept_tokens, rejected_tokens):
    """Reconstruct the Keep-only baseline and gate CER analysis on exact match.

    Returns only aggregate diagnostics. The reconstructed text is never returned.
    """
    if not isinstance(existing_hypothesis, str):
        raise ValueError("existing_hypothesis must be a string")
    kept = list(kept_tokens)
    rejected = list(rejected_tokens)
    reconstructed = "".join(
        item["text"] for item in sorted(kept, key=lambda x: (x["time"], x["token_id"]))
    ).strip()
    matched = reconstructed == existing_hypothesis
    mismatch_reason = None
    if not matched:
        raw = "".join(item["text"] for item in sorted(kept, key=lambda x: (x["time"], x["token_id"])))
        if raw.strip() == existing_hypothesis and raw != reconstructed:
            mismatch_reason = "edge_whitespace"
        elif sorted(reconstructed) == sorted(existing_hypothesis):
            mismatch_reason = "character_order"
        elif len(reconstructed) != len(existing_hypothesis):
            mismatch_reason = "character_count"
        else:
            mismatch_reason = "content_or_order"
    return {
        "baseline_match": matched,
        "mismatch_reason": mismatch_reason,
        "baseline_chars": len(existing_hypothesis),
        "reconstructed_chars": len(reconstructed),
        "rejected_tokens": len(rejected),
    }


def gated_counterfactual_rescue_cer(reference, existing_hypothesis, kept_tokens, rejected_tokens, thresholds_ms=(100, 250, 500)):
    """Only calculate counterfactual CER when reconstructed baseline exactly matches.


    Mismatch results intentionally contain diagnostics only and no CER values.
    """
    gate = compare_baseline_and_gate(existing_hypothesis, kept_tokens, rejected_tokens)
    if not gate["baseline_match"]:
        return {"gate": gate, "cer_comparison_status": "stopped_baseline_mismatch"}
    result = counterfactual_rescue_cer(reference, kept_tokens, rejected_tokens, thresholds_ms)
    # The exact existing hypothesis is the accepted baseline for comparison.
    result["baseline"] = _metrics(reference, existing_hypothesis)
    return {"gate": gate, "cer_comparison_status": "baseline_verified", "cer": result}
