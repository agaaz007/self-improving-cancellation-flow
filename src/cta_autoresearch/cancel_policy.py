from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


PRIMARY_REASONS = (
    "price",
    "graduating",
    "break",
    "quality_bug",
    "feature_gap",
    "competition",
    "billing_confusion",
    "other",
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(parsed, minimum)
    return parsed


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_reason(value: Any) -> str:
    reason = str(value or "").strip().lower().replace(" ", "_")
    if reason in PRIMARY_REASONS:
        return reason
    return "other"


def _now() -> float:
    return time.time()


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round(0.95 * (len(sorted_values) - 1)))
    return sorted_values[index]


@dataclass(frozen=True)
class TranscriptExtractionV1:
    primary_reason: str
    secondary_reasons: tuple[str, ...] = ()
    intent_strength: float = 0.5
    save_openness: float = 0.5
    frustration_level: float = 0.5
    trust_risk: float = 0.5
    billing_confusion_flag: bool = False
    competitor_mentions: tuple[str, ...] = ()
    feature_requests: tuple[str, ...] = ()
    bug_signals: tuple[str, ...] = ()
    summary: str = ""
    confidence: float = 0.5
    extractor_version: str = "heuristic-v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "primary_reason": self.primary_reason,
            "secondary_reasons": list(self.secondary_reasons),
            "intent_strength": round(self.intent_strength, 4),
            "save_openness": round(self.save_openness, 4),
            "frustration_level": round(self.frustration_level, 4),
            "trust_risk": round(self.trust_risk, 4),
            "billing_confusion_flag": self.billing_confusion_flag,
            "competitor_mentions": list(self.competitor_mentions),
            "feature_requests": list(self.feature_requests),
            "bug_signals": list(self.bug_signals),
            "summary": self.summary,
            "confidence": round(self.confidence, 4),
            "extractor_version": self.extractor_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TranscriptExtractionV1":
        secondary = tuple(
            reason
            for reason in (_normalize_reason(value) for value in (payload.get("secondary_reasons") or []))
            if reason != "other"
        )
        return cls(
            primary_reason=_normalize_reason(payload.get("primary_reason")),
            secondary_reasons=secondary,
            intent_strength=_clamp(_as_float(payload.get("intent_strength"), 0.5)),
            save_openness=_clamp(_as_float(payload.get("save_openness"), 0.5)),
            frustration_level=_clamp(_as_float(payload.get("frustration_level"), 0.5)),
            trust_risk=_clamp(_as_float(payload.get("trust_risk"), 0.5)),
            billing_confusion_flag=_as_bool(payload.get("billing_confusion_flag")),
            competitor_mentions=tuple(str(item) for item in (payload.get("competitor_mentions") or [])),
            feature_requests=tuple(str(item) for item in (payload.get("feature_requests") or [])),
            bug_signals=tuple(str(item) for item in (payload.get("bug_signals") or [])),
            summary=str(payload.get("summary") or ""),
            confidence=_clamp(_as_float(payload.get("confidence"), 0.5)),
            extractor_version=str(payload.get("extractor_version") or "heuristic-v1"),
        )


@dataclass(frozen=True)
class CancelContextV1:
    session_id: str
    user_id_hash: str
    timestamp: float
    plan_tier: str
    tenure_days: int
    engagement_7d: float
    engagement_30d: float
    prior_cancel_attempts_30d: int
    discount_exposures_30d: int
    transcript_extraction: TranscriptExtractionV1
    eligible_actions: tuple[str, ...] = ()
    context_version: str = "v1"

    def to_dict(self) -> dict[str, object]:
        payload = {
            "session_id": self.session_id,
            "user_id_hash": self.user_id_hash,
            "timestamp": self.timestamp,
            "plan_tier": self.plan_tier,
            "tenure_days": self.tenure_days,
            "engagement_7d": round(self.engagement_7d, 4),
            "engagement_30d": round(self.engagement_30d, 4),
            "prior_cancel_attempts_30d": self.prior_cancel_attempts_30d,
            "discount_exposures_30d": self.discount_exposures_30d,
            "transcript_extraction": self.transcript_extraction.to_dict(),
            "eligible_actions": list(self.eligible_actions),
            "context_version": self.context_version,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CancelContextV1":
        extraction_payload = payload.get("transcript_extraction")
        if not isinstance(extraction_payload, dict):
            raise ValueError("transcript_extraction is required and must be an object")
        return cls(
            session_id=str(payload.get("session_id") or f"session_{uuid4().hex[:12]}"),
            user_id_hash=str(payload.get("user_id_hash") or "anonymous"),
            timestamp=_as_float(payload.get("timestamp"), _now()),
            plan_tier=str(payload.get("plan_tier") or "unknown").strip().lower() or "unknown",
            tenure_days=_as_int(payload.get("tenure_days"), 0, minimum=0),
            engagement_7d=_clamp(_as_float(payload.get("engagement_7d"), 0.0)),
            engagement_30d=_clamp(_as_float(payload.get("engagement_30d"), 0.0)),
            prior_cancel_attempts_30d=_as_int(payload.get("prior_cancel_attempts_30d"), 0, minimum=0),
            discount_exposures_30d=_as_int(payload.get("discount_exposures_30d"), 0, minimum=0),
            transcript_extraction=TranscriptExtractionV1.from_dict(extraction_payload),
            eligible_actions=tuple(str(item) for item in (payload.get("eligible_actions") or [])),
            context_version=str(payload.get("context_version") or "v1"),
        )


@dataclass(frozen=True)
class PolicyDecisionV1:
    decision_id: str
    policy_version: str
    action_id: str
    propensity: float
    exploration_flag: bool
    blocked_action_ids: tuple[str, ...]
    fallback_reason: str | None = None
    holdout_flag: bool = False
    timestamp: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "policy_version": self.policy_version,
            "action_id": self.action_id,
            "propensity": round(self.propensity, 6),
            "exploration_flag": self.exploration_flag,
            "blocked_action_ids": list(self.blocked_action_ids),
            "fallback_reason": self.fallback_reason,
            "holdout_flag": self.holdout_flag,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class CancelOutcomeV1:
    decision_id: str
    session_id: str
    saved_flag: bool
    cancel_completed_flag: bool
    support_escalation_flag: bool
    complaint_flag: bool
    refund_7d_flag: bool | None = None
    outcome_timestamp: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "session_id": self.session_id,
            "saved_flag": self.saved_flag,
            "cancel_completed_flag": self.cancel_completed_flag,
            "support_escalation_flag": self.support_escalation_flag,
            "complaint_flag": self.complaint_flag,
            "refund_7d_flag": self.refund_7d_flag,
            "outcome_timestamp": self.outcome_timestamp,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CancelOutcomeV1":
        decision_id = str(payload.get("decision_id") or "").strip()
        session_id = str(payload.get("session_id") or "").strip()
        if not decision_id:
            raise ValueError("decision_id is required")
        if not session_id:
            raise ValueError("session_id is required")
        refund = payload.get("refund_7d_flag")
        return cls(
            decision_id=decision_id,
            session_id=session_id,
            saved_flag=_as_bool(payload.get("saved_flag")),
            cancel_completed_flag=_as_bool(payload.get("cancel_completed_flag")),
            support_escalation_flag=_as_bool(payload.get("support_escalation_flag")),
            complaint_flag=_as_bool(payload.get("complaint_flag")),
            refund_7d_flag=_as_bool(refund) if refund is not None else None,
            outcome_timestamp=_as_float(payload.get("outcome_timestamp"), _now()),
        )


@dataclass(frozen=True)
class ActionDefinition:
    id: str
    label: str
    offer_kind: str
    aggressive_urgency: bool = False
    is_discount: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "offer_kind": self.offer_kind,
            "aggressive_urgency": self.aggressive_urgency,
            "is_discount": self.is_discount,
            "message": self.message,
        }


DEFAULT_ACTIONS: dict[str, ActionDefinition] = {
    "control_empathic_exit": ActionDefinition(
        id="control_empathic_exit",
        label="Control - Empathetic Exit",
        offer_kind="none",
        message="We can make this easier. Tell us what is not working.",
    ),
    "pause_plan_relief": ActionDefinition(
        id="pause_plan_relief",
        label="Pause Plan",
        offer_kind="pause",
        message="Pause now and resume when your schedule stabilizes.",
    ),
    "downgrade_lite_switch": ActionDefinition(
        id="downgrade_lite_switch",
        label="Downgrade to Lite",
        offer_kind="downgrade",
        message="Switch to a lighter plan and keep your progress.",
    ),
    "targeted_discount_20": ActionDefinition(
        id="targeted_discount_20",
        label="Targeted 20% Discount",
        offer_kind="discount",
        is_discount=True,
        message="Stay on track with a limited 20% discount.",
    ),
    "targeted_discount_40": ActionDefinition(
        id="targeted_discount_40",
        label="Targeted 40% Discount",
        offer_kind="discount",
        is_discount=True,
        aggressive_urgency=True,
        message="High-savings offer available now if you continue.",
    ),
    "concierge_recovery": ActionDefinition(
        id="concierge_recovery",
        label="Concierge Recovery",
        offer_kind="support",
        message="Get a guided reset from our learning support team.",
    ),
    "exam_sprint_focus": ActionDefinition(
        id="exam_sprint_focus",
        label="Exam Sprint Focus",
        offer_kind="extension",
        aggressive_urgency=True,
        message="Use an exam sprint path to finish your current goal.",
    ),
    "feature_value_recap": ActionDefinition(
        id="feature_value_recap",
        label="Feature Value Recap",
        offer_kind="credit",
        message="Quickly recover value from features you have not used yet.",
    ),
    "billing_clarity_reset": ActionDefinition(
        id="billing_clarity_reset",
        label="Billing Clarity + Options",
        offer_kind="billing",
        message="See billing details and simpler options before deciding.",
    ),
}


DEFAULT_REASON_DENYLIST: dict[str, set[str]] = {
    "billing_confusion": {"exam_sprint_focus"},
    "graduating": {"exam_sprint_focus", "targeted_discount_40"},
    "break": {"exam_sprint_focus"},
}


class TranscriptExtractor:
    def __init__(
        self,
        *,
        model_name: str = "gpt-5.4-mini",
        use_openai: bool = True,
        openai_reasoning_effort: str = "low",
    ) -> None:
        self.model_name = model_name
        self.use_openai = use_openai
        self.openai_reasoning_effort = openai_reasoning_effort
        self.request_count = 0
        self.openai_request_count = 0
        self.fallback_count = 0
        self._latency_ms = deque(maxlen=512)

    def _normalize_transcript(self, transcript: object) -> str:
        if isinstance(transcript, str):
            return transcript.strip()
        if isinstance(transcript, list):
            chunks: list[str] = []
            for item in transcript:
                if isinstance(item, dict):
                    role = str(item.get("role") or "").strip()
                    content = str(item.get("content") or "").strip()
                    if role:
                        chunks.append(f"{role}: {content}")
                    elif content:
                        chunks.append(content)
                else:
                    chunks.append(str(item))
            return "\n".join(chunk for chunk in chunks if chunk)
        if isinstance(transcript, dict):
            content = transcript.get("content")
            if isinstance(content, str):
                return content.strip()
            return json.dumps(transcript)
        return str(transcript or "").strip()

    def _extract_via_openai(self, transcript: str, metadata: dict[str, object] | None = None) -> TranscriptExtractionV1:
        if not self.use_openai or OpenAI is None:
            raise RuntimeError("OpenAI extraction disabled")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")

        prompt = "\n".join(
            [
                "You extract cancellation intent into a strict JSON object.",
                "Use keys exactly:",
                "primary_reason, secondary_reasons, intent_strength, save_openness, frustration_level, trust_risk, billing_confusion_flag, competitor_mentions, feature_requests, bug_signals, summary, confidence, extractor_version.",
                f"Allowed primary_reason values: {', '.join(PRIMARY_REASONS)}",
                "Return JSON only.",
                "Conversation transcript:",
                transcript,
                "Metadata:",
                json.dumps(metadata or {}, ensure_ascii=True),
            ]
        )
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=self.model_name,
            input=prompt,
            reasoning={"effort": self.openai_reasoning_effort},
            max_output_tokens=500,
        )
        raw = response.output_text or ""
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("Model response did not include JSON object")
        payload = json.loads(match.group(0))
        payload["extractor_version"] = str(payload.get("extractor_version") or f"openai-{self.model_name}")
        return TranscriptExtractionV1.from_dict(payload)

    def _extract_heuristic(self, transcript: str) -> TranscriptExtractionV1:
        text = " ".join(transcript.lower().split())
        keyword_map = {
            "price": ("expensive", "cost", "price", "afford", "budget", "too much", "payment"),
            "graduating": ("graduat", "finished school", "done with school", "completed degree"),
            "break": ("summer break", "winter break", "on break", "vacation", "holiday"),
            "quality_bug": ("bug", "crash", "slow", "loading", "error", "broken", "fail", "not generating", "pdf"),
            "feature_gap": ("missing", "feature", "flashcard", "anki", "audio", "podcast", "image", "copy paste", "export"),
            "competition": ("chatgpt", "gemini", "notebooklm", "competitor", "switching to"),
            "billing_confusion": ("billing", "charge", "charged", "quota", "limit", "refund", "unexpected"),
        }
        reason_scores = {reason: 0 for reason in PRIMARY_REASONS}
        for reason, keywords in keyword_map.items():
            for keyword in keywords:
                if keyword in text:
                    reason_scores[reason] += 1

        ranked = sorted(reason_scores.items(), key=lambda item: item[1], reverse=True)
        primary_reason = ranked[0][0] if ranked and ranked[0][1] > 0 else "other"
        secondary_reasons = tuple(
            reason
            for reason, score in ranked[1:]
            if score > 0 and reason != "other"
        )[:3]

        save_cues = ("maybe", "if", "unless", "consider", "pause", "discount", "downgrade", "could stay")
        cancel_cues = ("cancel", "leave", "stop", "unsubscribe", "quit", "churn")
        frustration_cues = ("angry", "frustrating", "annoying", "terrible", "hate", "broken", "crash", "slow")
        trust_cues = ("creepy", "misleading", "unexpected charge", "hidden", "not transparent", "trust")

        save_openness_hits = sum(1 for cue in save_cues if cue in text)
        cancel_hits = sum(1 for cue in cancel_cues if cue in text)
        frustration_hits = sum(1 for cue in frustration_cues if cue in text)
        trust_hits = sum(1 for cue in trust_cues if cue in text)

        competitor_mentions = []
        for competitor in ("chatgpt", "gemini", "notebooklm"):
            if competitor in text:
                competitor_mentions.append(competitor)

        feature_requests = []
        feature_terms = {
            "anki_export": ("anki", "export"),
            "audio_mode": ("audio", "podcast"),
            "image_support": ("image",),
            "flashcard_flip": ("flashcard", "flip"),
            "copy_paste_input": ("copy paste", "paste"),
        }
        for label, keywords in feature_terms.items():
            if any(keyword in text for keyword in keywords):
                feature_requests.append(label)

        bug_signals = []
        bug_terms = {
            "question_generation_failure": ("not generating", "generation fail"),
            "crash": ("crash",),
            "slow_loading": ("slow", "loading"),
            "pdf_upload_failure": ("pdf", "upload", "failed"),
        }
        for label, keywords in bug_terms.items():
            if all(keyword in text for keyword in keywords):
                bug_signals.append(label)

        intent_strength = _clamp(0.25 + 0.18 * cancel_hits + 0.10 * reason_scores.get(primary_reason, 0))
        save_openness = _clamp(0.20 + 0.18 * save_openness_hits - 0.08 * cancel_hits)
        frustration_level = _clamp(0.10 + 0.18 * frustration_hits + 0.10 * reason_scores.get("quality_bug", 0))
        trust_risk = _clamp(
            0.12
            + 0.10 * trust_hits
            + 0.12 * reason_scores.get("billing_confusion", 0)
            + 0.08 * frustration_hits
        )
        billing_confusion_flag = primary_reason == "billing_confusion" or reason_scores["billing_confusion"] > 0
        confidence = _clamp(0.42 + 0.07 * sum(score > 0 for _, score in ranked[:3]))

        summary = transcript.strip()
        if len(summary) > 220:
            summary = summary[:217].rstrip() + "..."

        return TranscriptExtractionV1(
            primary_reason=primary_reason,
            secondary_reasons=secondary_reasons,
            intent_strength=intent_strength,
            save_openness=save_openness,
            frustration_level=frustration_level,
            trust_risk=trust_risk,
            billing_confusion_flag=billing_confusion_flag,
            competitor_mentions=tuple(competitor_mentions),
            feature_requests=tuple(feature_requests),
            bug_signals=tuple(bug_signals),
            summary=summary,
            confidence=confidence,
            extractor_version="heuristic-v1",
        )

    def extract(self, transcript: object, *, metadata: dict[str, object] | None = None) -> TranscriptExtractionV1:
        start = time.perf_counter()
        self.request_count += 1
        normalized = self._normalize_transcript(transcript)
        if not normalized:
            raise ValueError("transcript is required")

        try:
            extraction = self._extract_via_openai(normalized, metadata=metadata)
            self.openai_request_count += 1
        except Exception:
            self.fallback_count += 1
            extraction = self._extract_heuristic(normalized)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._latency_ms.append(elapsed_ms)
        return extraction

    def health(self) -> dict[str, object]:
        latencies = list(self._latency_ms)
        return {
            "extractor_model": self.model_name,
            "extractor_use_openai": self.use_openai,
            "requests": self.request_count,
            "openai_requests": self.openai_request_count,
            "fallbacks": self.fallback_count,
            "latency_p95_ms": round(_p95(latencies), 2),
            "latency_avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        }


class CancelPolicyRuntime:
    def __init__(
        self,
        root: str | Path,
        *,
        actions: dict[str, ActionDefinition] | None = None,
        exploration_rate: float = 0.15,
        holdout_rate: float = 0.10,
        sticky_days: int = 7,
        discount_cap_30d: int = 1,
        policy_version: str = "v1",
        seed: int = 7,
        reason_denylist: dict[str, set[str]] | None = None,
        control_action_id: str = "control_empathic_exit",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.actions = dict(actions or DEFAULT_ACTIONS)
        if control_action_id not in self.actions:
            raise ValueError(f"control_action_id '{control_action_id}' must exist in actions")
        self.control_action_id = control_action_id
        self.exploration_rate = _clamp(exploration_rate)
        self.holdout_rate = _clamp(holdout_rate)
        self.sticky_days = max(1, sticky_days)
        self.discount_cap_30d = max(0, discount_cap_30d)
        self.policy_version = policy_version
        self.seed = int(seed)
        self.reason_denylist = {
            _normalize_reason(reason): set(values)
            for reason, values in (reason_denylist or DEFAULT_REASON_DENYLIST).items()
        }

        self.lock = RLock()
        self.state_path = self.root / "policy_state.json"
        self.decisions_path = self.root / "decisions.jsonl"
        self.outcomes_path = self.root / "outcomes.jsonl"
        self._latency_ms = {
            "decide": deque(maxlen=512),
            "outcome": deque(maxlen=512),
            "warm_start": deque(maxlen=512),
        }

        self.state = self._load_state()
        self._decision_index: dict[str, dict[str, object]] = {}
        self._outcome_by_decision: dict[str, dict[str, object]] = {}
        self._load_journals()

    def _default_arms(self) -> dict[str, dict[str, float]]:
        return {
            action_id: {"alpha": 1.0, "beta": 1.0, "impressions": 0.0, "outcomes": 0.0}
            for action_id in self.actions
        }

    def _default_state(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "created_at": _now(),
            "updated_at": _now(),
            "config": {
                "exploration_rate": self.exploration_rate,
                "holdout_rate": self.holdout_rate,
                "sticky_days": self.sticky_days,
                "discount_cap_30d": self.discount_cap_30d,
                "seed": self.seed,
                "control_action_id": self.control_action_id,
            },
            "arms_global": self._default_arms(),
            "arms_context": {},
            "sticky_assignments": {},
            "discount_exposures": {},
            "metrics": {
                "decisions": 0,
                "outcomes": 0,
                "fallbacks": 0,
                "blocked_actions_total": 0,
            },
        }

    def _load_state(self) -> dict[str, object]:
        if not self.state_path.exists():
            return self._default_state()
        try:
            payload = json.loads(self.state_path.read_text())
        except json.JSONDecodeError:
            return self._default_state()

        state = self._default_state()
        state.update(payload)
        arms_global = dict(state.get("arms_global") or {})
        for action_id in self.actions:
            if action_id not in arms_global:
                arms_global[action_id] = {"alpha": 1.0, "beta": 1.0, "impressions": 0.0, "outcomes": 0.0}
        state["arms_global"] = arms_global
        state["arms_context"] = dict(state.get("arms_context") or {})
        state["sticky_assignments"] = dict(state.get("sticky_assignments") or {})
        state["discount_exposures"] = dict(state.get("discount_exposures") or {})
        state["metrics"] = dict(state.get("metrics") or {})
        return state

    def _persist_state(self) -> None:
        self.state["updated_at"] = _now()
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def _append_jsonl(self, path: Path, payload: dict[str, object]) -> None:
        with path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _load_journals(self) -> None:
        if self.decisions_path.exists():
            for line in self.decisions_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                decision_id = str(record.get("decision_id") or "")
                if decision_id:
                    self._decision_index[decision_id] = record
        if self.outcomes_path.exists():
            for line in self.outcomes_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                decision_id = str(record.get("decision_id") or "")
                if decision_id:
                    self._outcome_by_decision[decision_id] = record

    def _context_key(self, context: CancelContextV1, action_id: str) -> str:
        reason = context.transcript_extraction.primary_reason
        plan = context.plan_tier
        return f"{reason}|{plan}|{action_id}"

    def _ensure_arm(self, action_id: str, context_key: str) -> tuple[dict[str, float], dict[str, float]]:
        arms_global: dict[str, dict[str, float]] = self.state["arms_global"]  # type: ignore[assignment]
        arms_context: dict[str, dict[str, float]] = self.state["arms_context"]  # type: ignore[assignment]
        if action_id not in arms_global:
            arms_global[action_id] = {"alpha": 1.0, "beta": 1.0, "impressions": 0.0, "outcomes": 0.0}
        if context_key not in arms_context:
            arms_context[context_key] = {"alpha": 1.0, "beta": 1.0, "impressions": 0.0, "outcomes": 0.0}
        return arms_global[action_id], arms_context[context_key]

    def _posterior_params(self, context: CancelContextV1, action_id: str) -> tuple[float, float]:
        context_key = self._context_key(context, action_id)
        global_arm, context_arm = self._ensure_arm(action_id, context_key)
        alpha = max(float(global_arm["alpha"]) + float(context_arm["alpha"]) - 1.0, 1e-6)
        beta = max(float(global_arm["beta"]) + float(context_arm["beta"]) - 1.0, 1e-6)
        return alpha, beta

    def _hash_bucket(self, value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 100

    def _is_holdout(self, user_id_hash: str) -> bool:
        bucket = self._hash_bucket(f"{self.policy_version}:{user_id_hash}:holdout")
        return bucket < int(round(self.holdout_rate * 100))

    def _user_discount_exposure_count(self, user_id_hash: str, now: float) -> int:
        sticky_seconds = 30 * 24 * 60 * 60
        exposures: dict[str, list[float]] = self.state["discount_exposures"]  # type: ignore[assignment]
        current = [float(ts) for ts in exposures.get(user_id_hash, []) if now - float(ts) <= sticky_seconds]
        exposures[user_id_hash] = current
        return len(current)

    def _blocked_actions(
        self,
        context: CancelContextV1,
        user_id_hash: str,
        eligible_actions: list[str],
    ) -> set[str]:
        blocked: set[str] = set()
        reason = context.transcript_extraction.primary_reason
        denylist = self.reason_denylist.get(reason, set())
        blocked.update(action_id for action_id in eligible_actions if action_id in denylist)

        now = _now()
        prior_exposures = context.discount_exposures_30d + self._user_discount_exposure_count(user_id_hash, now)
        if prior_exposures >= self.discount_cap_30d and self.discount_cap_30d >= 0:
            blocked.update(action_id for action_id in eligible_actions if self.actions[action_id].is_discount)

        if context.transcript_extraction.billing_confusion_flag:
            blocked.update(
                action_id
                for action_id in eligible_actions
                if self.actions[action_id].aggressive_urgency
            )
        return blocked

    def _sticky_action(self, user_id_hash: str, now: float, eligible_actions: list[str]) -> str | None:
        sticky: dict[str, dict[str, object]] = self.state["sticky_assignments"]  # type: ignore[assignment]
        record = sticky.get(user_id_hash)
        if not record:
            return None
        action_id = str(record.get("action_id") or "")
        timestamp = _as_float(record.get("timestamp"), 0.0)
        sticky_window = self.sticky_days * 24 * 60 * 60
        if now - timestamp > sticky_window:
            return None
        if action_id not in eligible_actions:
            return None
        return action_id

    def _seeded_rng(self, context: CancelContextV1, suffix: str) -> random.Random:
        digest = hashlib.sha256(
            f"{self.seed}:{context.session_id}:{context.user_id_hash}:{suffix}".encode("utf-8")
        ).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _select_action(self, context: CancelContextV1, eligible_actions: list[str]) -> tuple[str, float, bool]:
        if len(eligible_actions) == 1:
            return eligible_actions[0], 1.0, False

        rng = self._seeded_rng(context, "bandit")
        exploration_flag = rng.random() < self.exploration_rate
        if exploration_flag:
            chosen = rng.choice(eligible_actions)
            propensity = 1.0 / len(eligible_actions)
            return chosen, propensity, True

        draws: dict[str, float] = {}
        for action_id in eligible_actions:
            alpha, beta = self._posterior_params(context, action_id)
            draws[action_id] = rng.betavariate(alpha, beta)
        chosen = max(draws.items(), key=lambda item: item[1])[0]
        denominator = sum(draws.values()) or 1.0
        propensity = max(draws[chosen] / denominator, 1e-6)
        return chosen, propensity, False

    def _record_discount_exposure(self, user_id_hash: str, action_id: str, now: float) -> None:
        if not self.actions[action_id].is_discount:
            return
        exposures: dict[str, list[float]] = self.state["discount_exposures"]  # type: ignore[assignment]
        exposures.setdefault(user_id_hash, [])
        exposures[user_id_hash].append(now)

    def get_action(self, action_id: str) -> dict[str, object]:
        action = self.actions.get(action_id)
        if action is None:
            raise ValueError(f"Unknown action_id '{action_id}'")
        return action.to_dict()

    def list_actions(self) -> list[dict[str, object]]:
        return [self.actions[action_id].to_dict() for action_id in sorted(self.actions)]

    def decide(self, context: CancelContextV1) -> PolicyDecisionV1:
        start = time.perf_counter()
        with self.lock:
            now = _now()
            user_id_hash = context.user_id_hash or "anonymous"
            provided = list(context.eligible_actions) if context.eligible_actions else list(self.actions)
            eligible = [action_id for action_id in provided if action_id in self.actions]
            if not eligible:
                eligible = [self.control_action_id]

            blocked = self._blocked_actions(context, user_id_hash, eligible)
            filtered = [action_id for action_id in eligible if action_id not in blocked]
            fallback_reason: str | None = None
            if not filtered:
                filtered = [self.control_action_id]
                fallback_reason = "all_actions_blocked"
                self.state["metrics"]["fallbacks"] = _as_int(self.state["metrics"].get("fallbacks"), 0) + 1  # type: ignore[index]

            holdout_flag = self._is_holdout(user_id_hash)
            exploration_flag = False
            propensity = 1.0

            if holdout_flag:
                action_id = self.control_action_id if self.control_action_id in filtered else filtered[0]
                fallback_reason = fallback_reason or "holdout_control"
            else:
                sticky = self._sticky_action(user_id_hash, now, filtered)
                if sticky:
                    action_id = sticky
                    propensity = 1.0
                else:
                    action_id, propensity, exploration_flag = self._select_action(context, filtered)
                    sticky_assignments: dict[str, dict[str, object]] = self.state["sticky_assignments"]  # type: ignore[assignment]
                    sticky_assignments[user_id_hash] = {"action_id": action_id, "timestamp": now}

            self._record_discount_exposure(user_id_hash, action_id, now)
            decision = PolicyDecisionV1(
                decision_id=f"dec_{uuid4().hex[:12]}",
                policy_version=self.policy_version,
                action_id=action_id,
                propensity=_clamp(propensity, low=1e-6, high=1.0),
                exploration_flag=exploration_flag,
                blocked_action_ids=tuple(sorted(blocked)),
                fallback_reason=fallback_reason,
                holdout_flag=holdout_flag,
                timestamp=now,
            )

            decision_record = {
                **decision.to_dict(),
                "session_id": context.session_id,
                "user_id_hash": user_id_hash,
                "primary_reason": context.transcript_extraction.primary_reason,
                "plan_tier": context.plan_tier,
                "eligible_actions": filtered,
                "context_version": context.context_version,
            }
            self._decision_index[decision.decision_id] = decision_record
            self._append_jsonl(self.decisions_path, decision_record)

            metrics: dict[str, object] = self.state["metrics"]  # type: ignore[assignment]
            metrics["decisions"] = _as_int(metrics.get("decisions"), 0) + 1
            metrics["blocked_actions_total"] = _as_int(metrics.get("blocked_actions_total"), 0) + len(blocked)
            self._persist_state()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._latency_ms["decide"].append(elapsed_ms)
        return decision

    def _reward(self, outcome: CancelOutcomeV1) -> float:
        reward = 1.0 if outcome.saved_flag else 0.0
        if outcome.support_escalation_flag:
            reward -= 0.3
        if outcome.complaint_flag:
            reward -= 0.2
        return reward

    def _update_posteriors(
        self,
        *,
        context_reason: str,
        plan_tier: str,
        action_id: str,
        reward: float,
        saved_flag: bool,
    ) -> None:
        context_reason = _normalize_reason(context_reason)
        context_key = f"{context_reason}|{plan_tier}|{action_id}"
        global_arm, context_arm = self._ensure_arm(action_id, context_key)

        success_delta = 1.0 if saved_flag else 0.0
        global_arm["alpha"] = float(global_arm["alpha"]) + success_delta + max(reward, 0.0)
        global_arm["beta"] = float(global_arm["beta"]) + (1.0 - success_delta) + max(-reward, 0.0)
        global_arm["impressions"] = float(global_arm["impressions"]) + 1.0
        global_arm["outcomes"] = float(global_arm["outcomes"]) + (1.0 if saved_flag else 0.0)

        context_arm["alpha"] = float(context_arm["alpha"]) + success_delta + max(reward, 0.0)
        context_arm["beta"] = float(context_arm["beta"]) + (1.0 - success_delta) + max(-reward, 0.0)
        context_arm["impressions"] = float(context_arm["impressions"]) + 1.0
        context_arm["outcomes"] = float(context_arm["outcomes"]) + (1.0 if saved_flag else 0.0)

    def record_outcome(self, outcome: CancelOutcomeV1) -> dict[str, object]:
        start = time.perf_counter()
        with self.lock:
            if outcome.decision_id in self._outcome_by_decision:
                return {
                    "status": "duplicate",
                    "decision_id": outcome.decision_id,
                    "message": "Outcome already recorded for decision",
                }

            decision = self._decision_index.get(outcome.decision_id)
            if decision is None:
                raise ValueError(f"Unknown decision_id '{outcome.decision_id}'")
            if str(decision.get("session_id") or "") != outcome.session_id:
                raise ValueError("session_id does not match decision session")

            reward = self._reward(outcome)
            action_id = str(decision.get("action_id") or self.control_action_id)
            reason = str(decision.get("primary_reason") or "other")
            plan_tier = str(decision.get("plan_tier") or "unknown")
            self._update_posteriors(
                context_reason=reason,
                plan_tier=plan_tier,
                action_id=action_id,
                reward=reward,
                saved_flag=outcome.saved_flag,
            )

            outcome_record = {
                **outcome.to_dict(),
                "reward": reward,
                "action_id": action_id,
                "primary_reason": reason,
                "plan_tier": plan_tier,
                "policy_version": decision.get("policy_version"),
                "holdout_flag": decision.get("holdout_flag", False),
            }
            self._outcome_by_decision[outcome.decision_id] = outcome_record
            self._append_jsonl(self.outcomes_path, outcome_record)

            metrics: dict[str, object] = self.state["metrics"]  # type: ignore[assignment]
            metrics["outcomes"] = _as_int(metrics.get("outcomes"), 0) + 1
            self._persist_state()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._latency_ms["outcome"].append(elapsed_ms)
        return {
            "status": "recorded",
            "decision_id": outcome.decision_id,
            "reward": round(reward, 4),
            "action_id": action_id,
        }

    def warm_start(self, rows: list[dict[str, object]], *, reset_state: bool = False) -> dict[str, object]:
        start = time.perf_counter()
        with self.lock:
            if reset_state:
                self.state["arms_global"] = self._default_arms()
                self.state["arms_context"] = {}

            applied = 0
            skipped = 0
            for row in rows:
                action_id = str(row.get("action_id") or "").strip()
                if action_id not in self.actions:
                    skipped += 1
                    continue
                reason = _normalize_reason(row.get("primary_reason"))
                plan_tier = str(row.get("plan_tier") or "unknown").strip().lower() or "unknown"
                if "reward" in row:
                    reward = _as_float(row.get("reward"), 0.0)
                else:
                    synthetic_outcome = CancelOutcomeV1(
                        decision_id="warm_start",
                        session_id="warm_start",
                        saved_flag=_as_bool(row.get("saved_flag")),
                        cancel_completed_flag=_as_bool(row.get("cancel_completed_flag")),
                        support_escalation_flag=_as_bool(row.get("support_escalation_flag")),
                        complaint_flag=_as_bool(row.get("complaint_flag")),
                    )
                    reward = self._reward(synthetic_outcome)
                saved_flag = _as_bool(row.get("saved_flag")) if "saved_flag" in row else reward > 0.0
                self._update_posteriors(
                    context_reason=reason,
                    plan_tier=plan_tier,
                    action_id=action_id,
                    reward=reward,
                    saved_flag=saved_flag,
                )
                applied += 1

            self._persist_state()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._latency_ms["warm_start"].append(elapsed_ms)
        return {
            "status": "ok",
            "rows_received": len(rows),
            "rows_applied": applied,
            "rows_skipped": skipped,
            "reset_state": reset_state,
        }

    def _joined_records(self) -> list[dict[str, object]]:
        joined: list[dict[str, object]] = []
        for decision_id, outcome in self._outcome_by_decision.items():
            decision = self._decision_index.get(decision_id)
            if not decision:
                continue
            joined.append({**decision, **outcome})
        return joined

    def replay(self, rows: list[dict[str, object]] | None = None) -> dict[str, object]:
        records = list(rows or self._joined_records())
        if not records:
            return {
                "status": "ok",
                "rows": 0,
                "save_rate": 0.0,
                "average_reward": 0.0,
                "by_action": {},
                "by_reason": {},
                "note": "No rows available for replay.",
            }

        by_action: dict[str, dict[str, float]] = defaultdict(lambda: {"rows": 0.0, "saved": 0.0, "reward_sum": 0.0})
        by_reason: dict[str, dict[str, float]] = defaultdict(lambda: {"rows": 0.0, "saved": 0.0, "reward_sum": 0.0})
        total_saved = 0.0
        total_reward = 0.0

        for row in records:
            action_id = str(row.get("action_id") or "")
            if not action_id:
                action_id = self.control_action_id
            reason = _normalize_reason(row.get("primary_reason"))

            if "reward" in row:
                reward = _as_float(row.get("reward"), 0.0)
                saved = 1.0 if reward > 0.0 else 0.0
            else:
                saved = 1.0 if _as_bool(row.get("saved_flag")) else 0.0
                reward = (1.0 if saved else 0.0) - (0.3 if _as_bool(row.get("support_escalation_flag")) else 0.0)
                reward -= 0.2 if _as_bool(row.get("complaint_flag")) else 0.0

            by_action[action_id]["rows"] += 1.0
            by_action[action_id]["saved"] += saved
            by_action[action_id]["reward_sum"] += reward
            by_reason[reason]["rows"] += 1.0
            by_reason[reason]["saved"] += saved
            by_reason[reason]["reward_sum"] += reward
            total_saved += saved
            total_reward += reward

        def _render_bucket(bucket: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
            rendered: dict[str, dict[str, float]] = {}
            for key, value in bucket.items():
                rows_count = value["rows"] or 1.0
                rendered[key] = {
                    "rows": int(value["rows"]),
                    "save_rate": round(value["saved"] / rows_count, 4),
                    "average_reward": round(value["reward_sum"] / rows_count, 4),
                }
            return rendered

        row_count = len(records)
        return {
            "status": "ok",
            "rows": row_count,
            "save_rate": round(total_saved / row_count, 4),
            "average_reward": round(total_reward / row_count, 4),
            "by_action": _render_bucket(by_action),
            "by_reason": _render_bucket(by_reason),
            "note": "Replay is observational and not causal unless input rows are randomized.",
        }

    def regression_check(
        self,
        *,
        min_treatment_samples: int = 60,
        min_holdout_samples: int = 20,
        min_save_lift: float = 0.0,
        max_support_delta: float = 0.03,
        max_complaint_delta: float = 0.02,
    ) -> dict[str, object]:
        records = self._joined_records()
        treatment = [row for row in records if not _as_bool(row.get("holdout_flag"))]
        holdout = [row for row in records if _as_bool(row.get("holdout_flag"))]

        def _rates(rows: list[dict[str, object]]) -> dict[str, float]:
            if not rows:
                return {
                    "samples": 0.0,
                    "save_rate": 0.0,
                    "support_rate": 0.0,
                    "complaint_rate": 0.0,
                    "average_reward": 0.0,
                }
            samples = float(len(rows))
            saved = sum(1.0 for row in rows if _as_bool(row.get("saved_flag")))
            support = sum(1.0 for row in rows if _as_bool(row.get("support_escalation_flag")))
            complaint = sum(1.0 for row in rows if _as_bool(row.get("complaint_flag")))
            reward_sum = sum(_as_float(row.get("reward"), 0.0) for row in rows)
            return {
                "samples": samples,
                "save_rate": saved / samples,
                "support_rate": support / samples,
                "complaint_rate": complaint / samples,
                "average_reward": reward_sum / samples,
            }

        treatment_rates = _rates(treatment)
        holdout_rates = _rates(holdout)
        save_lift = treatment_rates["save_rate"] - holdout_rates["save_rate"]
        support_delta = treatment_rates["support_rate"] - holdout_rates["support_rate"]
        complaint_delta = treatment_rates["complaint_rate"] - holdout_rates["complaint_rate"]
        enough_samples = (
            treatment_rates["samples"] >= float(min_treatment_samples)
            and holdout_rates["samples"] >= float(min_holdout_samples)
        )
        passed = (
            enough_samples
            and save_lift >= min_save_lift
            and support_delta <= max_support_delta
            and complaint_delta <= max_complaint_delta
        )

        return {
            "status": "ok",
            "pass": passed,
            "enough_samples": enough_samples,
            "thresholds": {
                "min_treatment_samples": min_treatment_samples,
                "min_holdout_samples": min_holdout_samples,
                "min_save_lift": min_save_lift,
                "max_support_delta": max_support_delta,
                "max_complaint_delta": max_complaint_delta,
            },
            "metrics": {
                "treatment": {key: round(value, 4) for key, value in treatment_rates.items()},
                "holdout": {key: round(value, 4) for key, value in holdout_rates.items()},
                "save_lift": round(save_lift, 4),
                "support_delta": round(support_delta, 4),
                "complaint_delta": round(complaint_delta, 4),
            },
        }

    def health(self) -> dict[str, object]:
        metrics: dict[str, object] = self.state["metrics"]  # type: ignore[assignment]
        latency_payload = {}
        for key, samples in self._latency_ms.items():
            values = list(samples)
            latency_payload[key] = {
                "p95_ms": round(_p95(values), 2),
                "avg_ms": round(sum(values) / len(values), 2) if values else 0.0,
            }
        return {
            "status": "ok",
            "policy_version": self.state.get("policy_version", self.policy_version),
            "actions": len(self.actions),
            "decisions": _as_int(metrics.get("decisions"), 0),
            "outcomes": _as_int(metrics.get("outcomes"), 0),
            "fallbacks": _as_int(metrics.get("fallbacks"), 0),
            "blocked_actions_total": _as_int(metrics.get("blocked_actions_total"), 0),
            "exploration_rate": self.exploration_rate,
            "holdout_rate": self.holdout_rate,
            "sticky_days": self.sticky_days,
            "discount_cap_30d": self.discount_cap_30d,
            "latency": latency_payload,
        }
