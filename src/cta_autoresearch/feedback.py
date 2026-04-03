"""Feedback collection and closed-loop learning.

Collects production outcomes (churn/retain/downgrade) per variant per user,
aggregates performance metrics, and produces a feedback payload that the
Karpathy backend can ingest to improve future research runs.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from threading import RLock
from typing import Any


VALID_OUTCOMES = {"retained", "churned", "downgraded", "paused", "upgraded"}


@dataclass
class FeedbackEvent:
    """A single production outcome for a user who saw a variant."""

    user_id: str
    experiment_id: str
    variant_id: str
    outcome: str
    timestamp: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "meta": self.meta,
        }


@dataclass
class VariantPerformance:
    """Aggregated performance for a single variant."""

    variant_id: str
    variant_name: str
    impressions: int
    outcomes: dict[str, int]

    @property
    def retention_rate(self) -> float:
        if self.impressions == 0:
            return 0.0
        retained = self.outcomes.get("retained", 0) + self.outcomes.get("upgraded", 0)
        return retained / self.impressions

    @property
    def churn_rate(self) -> float:
        if self.impressions == 0:
            return 0.0
        return self.outcomes.get("churned", 0) / self.impressions

    @property
    def save_rate(self) -> float:
        """Any non-churn outcome counts as a save."""
        if self.impressions == 0:
            return 0.0
        churned = self.outcomes.get("churned", 0)
        return (self.impressions - churned) / self.impressions

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "variant_name": self.variant_name,
            "impressions": self.impressions,
            "outcomes": dict(self.outcomes),
            "retention_rate": round(self.retention_rate, 4),
            "churn_rate": round(self.churn_rate, 4),
            "save_rate": round(self.save_rate, 4),
        }


@dataclass
class ExperimentReport:
    """Full performance report for a deployed experiment."""

    experiment_id: str
    variant_reports: list[VariantPerformance]
    total_impressions: int
    overall_retention_rate: float
    champion_lift_vs_control: float
    confidence_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "total_impressions": self.total_impressions,
            "overall_retention_rate": round(self.overall_retention_rate, 4),
            "champion_lift_vs_control": round(self.champion_lift_vs_control, 4),
            "confidence_note": self.confidence_note,
            "variants": [v.to_dict() for v in self.variant_reports],
        }


class FeedbackStore:
    """Persistent feedback storage and aggregation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.events: list[FeedbackEvent] = []
        self.impressions: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self._load_all()

    def _events_path(self, experiment_id: str) -> Path:
        return self.root / f"feedback_{experiment_id}.jsonl"

    def _load_all(self) -> None:
        for path in self.root.glob("feedback_*.jsonl"):
            for line in path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    event = FeedbackEvent(
                        user_id=data["user_id"],
                        experiment_id=data["experiment_id"],
                        variant_id=data["variant_id"],
                        outcome=data["outcome"],
                        timestamp=data.get("timestamp", 0),
                        meta=data.get("meta", {}),
                    )
                    self.events.append(event)
                except (json.JSONDecodeError, KeyError):
                    continue

    def _impressions_path(self, experiment_id: str) -> Path:
        return self.root / f"impressions_{experiment_id}.jsonl"

    def record_impression(self, experiment_id: str, variant_id: str, user_id: str) -> None:
        with self.lock:
            self.impressions[experiment_id][variant_id].add(user_id)
            path = self._impressions_path(experiment_id)
            with path.open("a") as f:
                f.write(json.dumps({
                    "experiment_id": experiment_id,
                    "variant_id": variant_id,
                    "user_id": user_id,
                    "timestamp": time.time(),
                }) + "\n")

    def record_outcome(self, event: FeedbackEvent) -> None:
        if event.outcome not in VALID_OUTCOMES:
            raise ValueError(
                f"Invalid outcome '{event.outcome}'. Must be one of: {VALID_OUTCOMES}"
            )
        with self.lock:
            self.events.append(event)
            path = self._events_path(event.experiment_id)
            with path.open("a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

    def get_variant_performance(
        self,
        experiment_id: str,
        variant_names: dict[str, str] | None = None,
    ) -> list[VariantPerformance]:
        names = variant_names or {}
        with self.lock:
            by_variant: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            impression_counts: dict[str, int] = defaultdict(int)

            for event in self.events:
                if event.experiment_id != experiment_id:
                    continue
                by_variant[event.variant_id][event.outcome] += 1

            for variant_id, users in self.impressions.get(experiment_id, {}).items():
                impression_counts[variant_id] = len(users)

            all_variant_ids = set(by_variant.keys()) | set(impression_counts.keys())
            results = []
            for variant_id in sorted(all_variant_ids):
                outcomes = dict(by_variant.get(variant_id, {}))
                impressions = impression_counts.get(variant_id, sum(outcomes.values()))
                results.append(VariantPerformance(
                    variant_id=variant_id,
                    variant_name=names.get(variant_id, variant_id),
                    impressions=impressions,
                    outcomes=outcomes,
                ))
            return results

    def build_experiment_report(
        self,
        experiment_id: str,
        variant_names: dict[str, str] | None = None,
        control_variant_id: str | None = None,
    ) -> ExperimentReport:
        variant_reports = self.get_variant_performance(experiment_id, variant_names)
        total = sum(v.impressions for v in variant_reports)
        overall_retention = (
            mean([v.retention_rate for v in variant_reports if v.impressions > 0])
            if any(v.impressions > 0 for v in variant_reports)
            else 0.0
        )

        control_rate = 0.0
        champion_rate = 0.0
        for v in variant_reports:
            if v.variant_id == control_variant_id:
                control_rate = v.retention_rate
            elif v.retention_rate > champion_rate:
                champion_rate = v.retention_rate

        lift = champion_rate - control_rate

        if total < 30:
            confidence = "Too few impressions for any signal. Need 30+ per variant."
        elif total < 100:
            confidence = "Early signal only. Directional, not statistically significant."
        elif total < 500:
            confidence = "Moderate confidence. Trends are forming but not conclusive."
        else:
            confidence = "Strong sample size. Results are likely reliable."

        return ExperimentReport(
            experiment_id=experiment_id,
            variant_reports=variant_reports,
            total_impressions=total,
            overall_retention_rate=overall_retention,
            champion_lift_vs_control=lift,
            confidence_note=confidence,
        )

    def build_learning_payload(self, experiment_id: str) -> dict[str, Any]:
        """Build a feedback payload the Karpathy backend can ingest.

        This is the key closed-loop output: real production outcomes
        mapped back to strategy dimensions so the next research run
        can learn from what actually worked.
        """
        report = self.build_experiment_report(experiment_id)
        return {
            "experiment_id": experiment_id,
            "generated_at": time.time(),
            "total_impressions": report.total_impressions,
            "confidence_note": report.confidence_note,
            "variant_outcomes": [v.to_dict() for v in report.variant_reports],
            "champion_lift_vs_control": round(report.champion_lift_vs_control, 4),
            "overall_retention_rate": round(report.overall_retention_rate, 4),
        }
