"""Train, validation, and test assignment without shared sample ids."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from vaakku.constants import SPLIT_ORDER
from vaakku.data.schema import NormalizedSample
from vaakku.errors import DataError


def hash_split(sample_id: str, seed: int, ratios: dict[str, float]) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    position = int(digest[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name in SPLIT_ORDER:
        cumulative += ratios[name]
        if position < cumulative:
            return name
    return SPLIT_ORDER[-1]


def assign_splits(
    samples: list[NormalizedSample],
    ratios: dict[str, float],
    seed: int,
    speaker_disjoint: bool,
) -> tuple[dict[str, list[NormalizedSample]], str]:
    """Keep an official split when a sample already has one.

    Samples without a split are divided by speaker when speaker ids exist,
    and by a stable hash otherwise.
    """
    buckets: dict[str, list[NormalizedSample]] = {name: [] for name in SPLIT_ORDER}
    floating: list[NormalizedSample] = []
    for sample in samples:
        if sample.split in buckets:
            buckets[sample.split].append(sample)
        elif sample.split:
            raise DataError(f"Unknown split {sample.split!r} on {sample.sample_id}")
        else:
            floating.append(sample)

    if not floating:
        _assert_no_sample_leakage(buckets)
        return buckets, "official"

    speaker_groups: dict[str, list[NormalizedSample]] = defaultdict(list)
    unspeakered: list[NormalizedSample] = []
    for sample in floating:
        if sample.speaker_id:
            speaker_groups[f"{sample.dataset_name}:{sample.speaker_id}"].append(sample)
        else:
            unspeakered.append(sample)

    method = "official_plus_hash"
    if speaker_disjoint and speaker_groups:
        method = "official_plus_speaker_disjoint" if any(buckets.values()) else "speaker_disjoint"
        _assign_speaker_groups(speaker_groups, ratios, seed, buckets)
    else:
        unspeakered.extend(sample for group in speaker_groups.values() for sample in group)
        speaker_groups = {}
        method = "official_plus_hash" if any(buckets.values()) else "hash_sample_id"

    for sample in unspeakered:
        buckets[hash_split(sample.sample_id, seed, ratios)].append(sample)

    _assert_no_sample_leakage(buckets)
    if speaker_disjoint:
        _assert_assigned_speakers_are_disjoint(buckets)
    return buckets, method


def official_speaker_overlap(buckets: dict[str, list[NormalizedSample]]) -> list[str]:
    """Speakers that already occur in more than one official split."""
    seen: dict[str, str] = {}
    overlaps: set[str] = set()
    for split, rows in buckets.items():
        for sample in rows:
            if not sample.speaker_id or sample.split is None:
                continue
            key = f"{sample.dataset_name}:{sample.speaker_id}"
            previous = seen.get(key)
            if previous and previous != split:
                overlaps.add(key)
            seen[key] = split
    return sorted(overlaps)


def _assign_speaker_groups(
    groups: dict[str, list[NormalizedSample]],
    ratios: dict[str, float],
    seed: int,
    buckets: dict[str, list[NormalizedSample]],
) -> None:
    total = sum(len(items) for items in groups.values())
    targets = {name: ratios[name] * total for name in SPLIT_ORDER}
    counts = {name: 0 for name in SPLIT_ORDER}
    speakers = list(groups)
    random.Random(seed).shuffle(speakers)
    for speaker in speakers:
        items = groups[speaker]

        def need(split: str) -> float:
            target = targets[split]
            if target <= 0:
                return -1.0
            return (target - counts[split]) / target

        choice = max(SPLIT_ORDER, key=lambda split: (need(split), targets[split]))
        buckets[choice].extend(items)
        counts[choice] += len(items)


def _assert_no_sample_leakage(buckets: dict[str, list[NormalizedSample]]) -> None:
    seen: set[str] = set()
    for rows in buckets.values():
        for sample in rows:
            if sample.sample_id in seen:
                raise DataError(f"Sample {sample.sample_id} appears in more than one split")
            seen.add(sample.sample_id)


def _assert_assigned_speakers_are_disjoint(buckets: dict[str, list[NormalizedSample]]) -> None:
    seen: dict[str, str] = {}
    for split, rows in buckets.items():
        for sample in rows:
            if sample.split is not None or not sample.speaker_id:
                continue
            key = f"{sample.dataset_name}:{sample.speaker_id}"
            previous = seen.get(key)
            if previous and previous != split:
                raise DataError(f"Speaker {key} leaked from {previous} into {split}")
            seen[key] = split
