from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


CATEGORIES: dict[str, dict[str, str]] = {
    "winning_lesson": {
        "label": "Winning lessons",
        "description": "Patterns the system should actively reuse.",
    },
    "repeated_failure": {
        "label": "Repeated failures",
        "description": "Mistakes that keep showing up across variants or judges.",
    },
    "contradicted_memory": {
        "label": "Contradicted memories",
        "description": "Lessons that now conflict with newer evidence.",
    },
    "stale_assumption": {
        "label": "Stale assumptions",
        "description": "Old beliefs that need retesting before use.",
    },
    "promoted_strategy": {
        "label": "Promoted strategies",
        "description": "Strategies approved for reuse or rollout.",
    },
    "blocked_strategy": {
        "label": "Blocked strategies",
        "description": "Strategies that should not be generated or shipped.",
    },
}

STATUSES = ("active", "promoted", "blocked", "contradicted", "stale", "archived")


def _now() -> float:
    return time.time()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any, default: float) -> float:
    return max(0.0, min(_as_float(value, default), 1.0))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _clean_category(value: Any) -> str:
    category = str(value or "").strip()
    return category if category in CATEGORIES else "winning_lesson"


def _default_status(category: str) -> str:
    return {
        "promoted_strategy": "promoted",
        "blocked_strategy": "blocked",
        "contradicted_memory": "contradicted",
        "stale_assumption": "stale",
    }.get(category, "active")


def _clean_status(value: Any, category: str) -> str:
    status = str(value or "").strip().lower()
    return status if status in STATUSES else _default_status(category)


def normalize_memory(payload: dict[str, Any], *, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    existing = existing or {}
    category = _clean_category(payload.get("category", existing.get("category")))
    now = _now()
    created_at = _as_float(existing.get("created_at"), now)
    return {
        "id": str(payload.get("id") or existing.get("id") or f"mem_{uuid4().hex[:10]}"),
        "client_id": str(payload.get("client_id") or existing.get("client_id") or "jungle_ai"),
        "category": category,
        "status": _clean_status(payload.get("status", existing.get("status")), category),
        "title": str(payload.get("title") or existing.get("title") or "Untitled learning").strip(),
        "lesson": str(payload.get("lesson") or existing.get("lesson") or "").strip(),
        "evidence": str(payload.get("evidence") or existing.get("evidence") or "").strip(),
        "recommendation": str(payload.get("recommendation") or existing.get("recommendation") or "").strip(),
        "module_id": str(payload.get("module_id") or existing.get("module_id") or "").strip(),
        "strategy_id": str(payload.get("strategy_id") or existing.get("strategy_id") or "").strip(),
        "source": str(payload.get("source") or existing.get("source") or "manual").strip(),
        "confidence": _clamp01(payload.get("confidence", existing.get("confidence", 0.65)), 0.65),
        "impact": _clamp01(payload.get("impact", existing.get("impact", 0.5)), 0.5),
        "tags": _as_list(payload.get("tags", existing.get("tags", []))),
        "created_at": created_at,
        "updated_at": now,
    }


def seed_memories(client_id: str = "jungle_ai") -> list[dict[str, Any]]:
    seeds = [
        {
            "category": "winning_lesson",
            "title": "Specific proof beats generic urgency",
            "lesson": "Judges reward paywalls that explain the exact user value before pushing urgency.",
            "evidence": "Repeated strategy reviews favored progress, unused value, and outcome proof over vague countdown copy.",
            "recommendation": "Prefer utility proof or progress recap as the first treatment for high-intent users.",
            "module_id": "psychological_framing",
            "source": "seed",
            "confidence": 0.78,
            "impact": 0.72,
            "tags": ["proof", "trust", "paywall"],
        },
        {
            "category": "repeated_failure",
            "title": "Discount-first variants overfit synthetic judges",
            "lesson": "Large discounts can raise short-term save probability while hurting revenue and trust scores.",
            "evidence": "Revenue judge repeatedly penalized high-generosity offers unless the user showed strong price sensitivity.",
            "recommendation": "Gate discounts behind price-sensitive segments and test pause or lite plan first.",
            "module_id": "pricing_discounting",
            "source": "seed",
            "confidence": 0.74,
            "impact": 0.68,
            "tags": ["discount", "revenue", "guardrail"],
        },
        {
            "category": "contradicted_memory",
            "title": "Aggressive urgency is not generally safe",
            "lesson": "Older assumptions that urgency improves conversion conflict with trust-judge feedback.",
            "evidence": "Trust judges reject urgency when the user has low deadline pressure or high frustration.",
            "recommendation": "Only use deadline framing when the user has a real exam, renewal, or expiring value window.",
            "module_id": "lifecycle_timing",
            "source": "seed",
            "confidence": 0.66,
            "impact": 0.56,
            "tags": ["urgency", "trust"],
        },
        {
            "category": "stale_assumption",
            "title": "Social proof may be too broad",
            "lesson": "Generic social proof should be retested against segment-specific proof before promotion.",
            "evidence": "The current command-center MVP assumes social proof, but memory suggests utility proof may be sharper.",
            "recommendation": "Compare social proof against personal usage proof in the next experiment.",
            "module_id": "contextual_relevance",
            "source": "seed",
            "confidence": 0.52,
            "impact": 0.48,
            "tags": ["social_proof", "retest"],
        },
        {
            "category": "promoted_strategy",
            "title": "Progress recap for warm learners",
            "lesson": "Warm learners respond well when the paywall shows what they have already built or unlocked.",
            "evidence": "Synthetic panels reward progress reflection when habit strength and activation are visible.",
            "recommendation": "Promote progress recap as a candidate treatment for Jungle high-intent learners.",
            "module_id": "ux_layout",
            "source": "seed",
            "confidence": 0.81,
            "impact": 0.76,
            "tags": ["progress", "warm_user"],
        },
        {
            "category": "blocked_strategy",
            "title": "Fake scarcity copy",
            "lesson": "Do not generate paywalls that imply scarcity, expiry, or limited seats unless the product has that fact.",
            "evidence": "Factuality gates and trust judges reject invented scarcity as manipulative.",
            "recommendation": "Block fake countdowns, invented limited offers, and unsupported exclusivity claims.",
            "module_id": "factuality_gate",
            "source": "seed",
            "confidence": 0.92,
            "impact": 0.84,
            "tags": ["blocked", "factuality", "trust"],
        },
    ]
    return [normalize_memory({**item, "client_id": client_id}) for item in seeds]


def normalize_memory_list(items: list[dict[str, Any]] | None, *, client_id: str) -> list[dict[str, Any]]:
    raw = items if items else seed_memories(client_id)
    normalized = [normalize_memory({**item, "client_id": item.get("client_id") or client_id}) for item in raw]
    normalized.sort(key=lambda item: (item["status"] == "archived", -float(item["impact"]), -float(item["updated_at"])))
    return normalized


def upsert_memory(items: list[dict[str, Any]], payload: dict[str, Any], *, client_id: str) -> list[dict[str, Any]]:
    memory_id = str(payload.get("id") or "")
    updated = []
    found = False
    for item in items:
        if memory_id and item.get("id") == memory_id:
            updated.append(normalize_memory({**payload, "client_id": client_id}, existing=item))
            found = True
        else:
            updated.append(item)
    if not found:
        updated.append(normalize_memory({**payload, "client_id": client_id}))
    return normalize_memory_list(updated, client_id=client_id)


def archive_memory(items: list[dict[str, Any]], memory_id: str, *, client_id: str) -> list[dict[str, Any]]:
    updated = []
    for item in items:
        if item.get("id") == memory_id:
            updated.append(normalize_memory({"status": "archived"}, existing=item))
        else:
            updated.append(item)
    return normalize_memory_list(updated, client_id=client_id)


def summarize_memory(items: list[dict[str, Any]], *, client_id: str) -> dict[str, Any]:
    active = [item for item in items if item.get("status") != "archived"]
    by_category = {key: [] for key in CATEGORIES}
    for item in active:
        by_category.setdefault(str(item.get("category")), []).append(item)
    return {
        "client_id": client_id,
        "updated_at": max((float(item.get("updated_at", 0.0)) for item in active), default=_now()),
        "categories": CATEGORIES,
        "counts": {key: len(by_category.get(key, [])) for key in CATEGORIES},
        "items": active,
        "by_category": by_category,
    }


class FileGBrainMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self, client_id: str) -> list[dict[str, Any]]:
        if not self.path.exists():
            return seed_memories(client_id)
        try:
            payload = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return seed_memories(client_id)
        items = payload.get(client_id, []) if isinstance(payload, dict) else []
        return normalize_memory_list(items, client_id=client_id)

    def save(self, client_id: str, items: list[dict[str, Any]]) -> None:
        payload: dict[str, Any] = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict):
                    payload = loaded
            except json.JSONDecodeError:
                payload = {}
        payload[client_id] = normalize_memory_list(items, client_id=client_id)
        self.path.write_text(json.dumps(payload, indent=2))
