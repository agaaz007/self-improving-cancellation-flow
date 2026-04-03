"""Experiment deployment manager.

Takes research run output from the Karpathy backend and creates deployable
variant configurations with traffic splits. Serves variant assignments
to production clients via a simple API.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class Variant:
    """A single deployable variant derived from a research strategy."""

    id: str
    name: str
    traffic_pct: int
    message_angle: str
    proof_style: str
    offer: str
    cta: str
    personalization: str
    contextual_grounding: str
    creative_treatment: str
    friction_reducer: str
    sample_message: str
    labels: dict[str, str]
    source_score: float
    source_lift: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "traffic_pct": self.traffic_pct,
            "dimensions": {
                "message_angle": self.message_angle,
                "proof_style": self.proof_style,
                "offer": self.offer,
                "cta": self.cta,
                "personalization": self.personalization,
                "contextual_grounding": self.contextual_grounding,
                "creative_treatment": self.creative_treatment,
                "friction_reducer": self.friction_reducer,
            },
            "labels": self.labels,
            "sample_message": self.sample_message,
            "source_score": self.source_score,
            "source_lift": self.source_lift,
        }

    def render_component(self, user_context: dict[str, str] | None = None) -> dict[str, Any]:
        """Render this variant into UI component data for the client."""
        ctx = user_context or {}
        return {
            "variant_id": self.id,
            "headline": self.labels.get("message_angle_label", ""),
            "body": self.sample_message,
            "proof": self.labels.get("proof_style_label", ""),
            "offer_text": self.labels.get("offer_label", ""),
            "cta_button": self.labels.get("cta_label", ""),
            "personalization_level": self.personalization,
            "grounding": self.labels.get("contextual_grounding_label", ""),
            "treatment": self.labels.get("creative_treatment_label", ""),
            "friction_reducer": self.labels.get("friction_reducer_label", ""),
            "user_name": ctx.get("name", ""),
            "study_context": ctx.get("study_context", ""),
        }


@dataclass
class Experiment:
    """A deployed experiment with multiple variants and a traffic split."""

    id: str
    name: str
    source_run_id: str
    variants: list[Variant]
    status: str = "draft"
    created_at: float = field(default_factory=time.time)
    deployed_at: float | None = None
    stopped_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "source_run_id": self.source_run_id,
            "status": self.status,
            "created_at": self.created_at,
            "deployed_at": self.deployed_at,
            "stopped_at": self.stopped_at,
            "variants": [v.to_dict() for v in self.variants],
            "traffic_split": {v.id: v.traffic_pct for v in self.variants},
        }


def _variant_from_strategy(strategy: dict[str, Any], traffic_pct: int, name: str) -> Variant:
    """Convert a research strategy dict into a deployable Variant."""
    return Variant(
        id=f"var_{uuid4().hex[:8]}",
        name=name,
        traffic_pct=traffic_pct,
        message_angle=strategy.get("message_angle", ""),
        proof_style=strategy.get("proof_style", ""),
        offer=strategy.get("offer", ""),
        cta=strategy.get("cta", ""),
        personalization=strategy.get("personalization", ""),
        contextual_grounding=strategy.get("contextual_grounding", "generic"),
        creative_treatment=strategy.get("creative_treatment", "plain_note"),
        friction_reducer=strategy.get("friction_reducer", "none"),
        sample_message=strategy.get("sample_message", ""),
        labels={
            "message_angle_label": strategy.get("message_angle_label", ""),
            "proof_style_label": strategy.get("proof_style_label", ""),
            "offer_label": strategy.get("offer_label", ""),
            "cta_label": strategy.get("cta_label", ""),
            "personalization_label": strategy.get("personalization_label", ""),
            "contextual_grounding_label": strategy.get("contextual_grounding_label", ""),
            "creative_treatment_label": strategy.get("creative_treatment_label", ""),
            "friction_reducer_label": strategy.get("friction_reducer_label", ""),
        },
        source_score=strategy.get("average_score", 0.0),
        source_lift=strategy.get("baseline_lift", 0.0),
    )


def _control_variant(traffic_pct: int) -> Variant:
    """The control variant — empathetic exit with no offer, the current baseline."""
    return Variant(
        id=f"var_control_{uuid4().hex[:6]}",
        name="Control (current baseline)",
        traffic_pct=traffic_pct,
        message_angle="empathetic_exit",
        proof_style="none",
        offer="none",
        cta="tell_us_why",
        personalization="generic",
        contextual_grounding="generic",
        creative_treatment="plain_note",
        friction_reducer="none",
        sample_message="If something is not working for you right now, we want to make the next step easier instead of boxing you in. CTA: Tell us why you're leaving",
        labels={
            "message_angle_label": "Empathetic Exit",
            "proof_style_label": "No explicit proof",
            "offer_label": "No offer",
            "cta_label": "Tell us why you're leaving",
            "personalization_label": "Generic",
            "contextual_grounding_label": "Generic grounding",
            "creative_treatment_label": "Plain note",
            "friction_reducer_label": "No friction reducer",
        },
        source_score=0.0,
        source_lift=0.0,
    )


def create_experiment_from_run(
    run_result: dict[str, Any],
    source_run_id: str = "",
    traffic_split: list[int] | None = None,
    name: str = "",
) -> Experiment:
    """Create an experiment from research run output.

    Default split: [80, 10, 10] — champion gets 80%, runner-up 10%, control 10%.
    The split list length determines how many variants are created:
      - First N-1 entries map to the top N-1 strategies from the run
      - The last entry is always the control variant
    """
    split = traffic_split or [80, 10, 10]
    if sum(split) != 100:
        raise ValueError(f"Traffic split must sum to 100, got {sum(split)}")

    top_strategies = run_result.get("top_strategies", [])
    if not top_strategies:
        raise ValueError("Run result has no top_strategies to deploy")

    variants: list[Variant] = []
    treatment_count = len(split) - 1

    for i in range(min(treatment_count, len(top_strategies))):
        strategy = top_strategies[i]
        rank_label = ["Champion", "Runner-up", "Challenger"][i] if i < 3 else f"Variant {i + 1}"
        variants.append(_variant_from_strategy(strategy, split[i], rank_label))

    variants.append(_control_variant(split[-1]))

    meta = run_result.get("meta", {})
    exp_name = name or f"CTA experiment from {meta.get('top_strategy', 'research run')[:60]}"

    return Experiment(
        id=f"exp_{uuid4().hex[:10]}",
        name=exp_name,
        source_run_id=source_run_id,
        variants=variants,
    )


def assign_variant(experiment: Experiment, user_id: str) -> Variant:
    """Deterministically assign a user to a variant based on their user_id.

    Uses consistent hashing so the same user always sees the same variant
    for the lifetime of the experiment.
    """
    if experiment.status != "live":
        raise ValueError(f"Experiment {experiment.id} is not live (status={experiment.status})")

    digest = hashlib.sha256(f"{experiment.id}:{user_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100

    cumulative = 0
    for variant in experiment.variants:
        cumulative += variant.traffic_pct
        if bucket < cumulative:
            return variant

    return experiment.variants[-1]


class ExperimentStore:
    """Persistent experiment storage backed by a directory of JSON files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.experiments: dict[str, Experiment] = {}
        self._load_all()

    def _exp_path(self, exp_id: str) -> Path:
        return self.root / f"{exp_id}.json"

    def _load_all(self) -> None:
        for path in self.root.glob("exp_*.json"):
            try:
                data = json.loads(path.read_text())
                variants = [
                    Variant(
                        id=v["id"],
                        name=v["name"],
                        traffic_pct=v["traffic_pct"],
                        message_angle=v["dimensions"]["message_angle"],
                        proof_style=v["dimensions"]["proof_style"],
                        offer=v["dimensions"]["offer"],
                        cta=v["dimensions"]["cta"],
                        personalization=v["dimensions"]["personalization"],
                        contextual_grounding=v["dimensions"]["contextual_grounding"],
                        creative_treatment=v["dimensions"]["creative_treatment"],
                        friction_reducer=v["dimensions"]["friction_reducer"],
                        sample_message=v.get("sample_message", ""),
                        labels=v.get("labels", {}),
                        source_score=v.get("source_score", 0.0),
                        source_lift=v.get("source_lift", 0.0),
                    )
                    for v in data.get("variants", [])
                ]
                exp = Experiment(
                    id=data["id"],
                    name=data["name"],
                    source_run_id=data.get("source_run_id", ""),
                    variants=variants,
                    status=data.get("status", "draft"),
                    created_at=data.get("created_at", 0),
                    deployed_at=data.get("deployed_at"),
                    stopped_at=data.get("stopped_at"),
                )
                self.experiments[exp.id] = exp
            except (json.JSONDecodeError, KeyError):
                continue

    def save(self, experiment: Experiment) -> None:
        with self.lock:
            self.experiments[experiment.id] = experiment
            self._exp_path(experiment.id).write_text(
                json.dumps(experiment.to_dict(), indent=2)
            )

    def get(self, exp_id: str) -> Experiment | None:
        with self.lock:
            return self.experiments.get(exp_id)

    def list_all(self) -> list[Experiment]:
        with self.lock:
            return sorted(
                self.experiments.values(),
                key=lambda e: e.created_at,
                reverse=True,
            )

    def get_live(self) -> Experiment | None:
        with self.lock:
            for exp in self.experiments.values():
                if exp.status == "live":
                    return exp
            return None

    def deploy(self, exp_id: str) -> Experiment:
        with self.lock:
            # Stop any currently live experiment
            for exp in self.experiments.values():
                if exp.status == "live" and exp.id != exp_id:
                    exp.status = "stopped"
                    exp.stopped_at = time.time()
                    self.save(exp)

            exp = self.experiments.get(exp_id)
            if exp is None:
                raise ValueError(f"Experiment {exp_id} not found")
            exp.status = "live"
            exp.deployed_at = time.time()
            self.save(exp)
            return exp

    def stop(self, exp_id: str) -> Experiment:
        with self.lock:
            exp = self.experiments.get(exp_id)
            if exp is None:
                raise ValueError(f"Experiment {exp_id} not found")
            exp.status = "stopped"
            exp.stopped_at = time.time()
            self.save(exp)
            return exp
