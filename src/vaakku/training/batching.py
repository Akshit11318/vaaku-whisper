"""Dynamic padding and duration bucketing."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator

import numpy as np


def pad_mel_features(features: list[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    if not features:
        raise ValueError("Cannot pad an empty batch")
    mel_bins = features[0].shape[0]
    max_time = max(feature.shape[-1] for feature in features)
    batch = np.full((len(features), mel_bins, max_time), pad_value, dtype=np.float32)
    for index, feature in enumerate(features):
        batch[index, :, : feature.shape[-1]] = feature
    return batch


def pad_label_ids(labels: list[list[int]], ignore_index: int = -100) -> np.ndarray:
    if not labels:
        raise ValueError("Cannot pad an empty label batch")
    width = max(len(row) for row in labels)
    batch = np.full((len(labels), width), ignore_index, dtype=np.int64)
    for index, row in enumerate(labels):
        batch[index, : len(row)] = np.asarray(row, dtype=np.int64)
    return batch


class DurationBucketBatchSampler:
    """Group similar durations so padding stays small."""

    def __init__(
        self,
        durations: list[float],
        batch_size: int,
        seed: int,
        shuffle: bool = True,
        bucket_span: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.durations = list(durations)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.bucket_span = bucket_span or batch_size * 8
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        order = list(range(len(self.durations)))
        order.sort(key=lambda index: self.durations[index])
        buckets = [
            order[start : start + self.bucket_span]
            for start in range(0, len(order), self.bucket_span)
        ]
        if self.shuffle:
            for bucket in buckets:
                generator.shuffle(bucket)
            generator.shuffle(buckets)
        flat = [index for bucket in buckets for index in bucket]
        for start in range(0, len(flat), self.batch_size):
            yield flat[start : start + self.batch_size]

    def __len__(self) -> int:
        if not self.durations:
            return 0
        return math.ceil(len(self.durations) / self.batch_size)
