"""In-process request counters. They reset when the process stops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RequestMetrics:
    requests: int = 0
    errors: int = 0
    latency_seconds_sum: float = 0.0

    def observe(self, latency_s: float, failed: bool) -> None:
        self.requests += 1
        self.latency_seconds_sum += latency_s
        if failed:
            self.errors += 1

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "latency_seconds_sum": round(self.latency_seconds_sum, 6),
        }
