from __future__ import annotations

from cta_autoresearch.gbrain_memory import (
    CATEGORIES,
    FileGBrainMemoryStore,
    archive_memory,
    seed_memories,
    summarize_memory,
    upsert_memory,
)


def test_seed_memories_cover_visible_categories():
    items = seed_memories("jungle_ai")
    summary = summarize_memory(items, client_id="jungle_ai")

    assert set(summary["by_category"]) == set(CATEGORIES)
    for category in CATEGORIES:
        assert summary["counts"][category] == 1


def test_upsert_memory_creates_and_updates_item():
    items = seed_memories("jungle_ai")
    items = upsert_memory(items, {
        "category": "repeated_failure",
        "title": "Weak price anchoring",
        "lesson": "Judges reject discount copy when the value proof is missing.",
        "module_id": "pricing_discounting",
        "confidence": 0.71,
        "impact": 0.62,
    }, client_id="jungle_ai")

    created = next(item for item in items if item["title"] == "Weak price anchoring")
    assert created["status"] == "active"
    assert created["category"] == "repeated_failure"

    items = upsert_memory(items, {
        "id": created["id"],
        "title": "Weak price anchoring",
        "lesson": "Judges reject discount copy unless savings are backed by concrete value proof.",
        "status": "blocked",
    }, client_id="jungle_ai")
    updated = next(item for item in items if item["id"] == created["id"])
    assert updated["status"] == "blocked"
    assert "concrete value proof" in updated["lesson"]


def test_archive_memory_hides_item_from_summary():
    items = seed_memories("jungle_ai")
    target = items[0]
    items = archive_memory(items, target["id"], client_id="jungle_ai")
    summary = summarize_memory(items, client_id="jungle_ai")

    assert target["id"] not in {item["id"] for item in summary["items"]}
    assert any(item["id"] == target["id"] and item["status"] == "archived" for item in items)


def test_file_gbrain_memory_store_persists_client_items(tmp_path):
    store = FileGBrainMemoryStore(tmp_path / "gbrain.json")
    items = upsert_memory([], {
        "category": "promoted_strategy",
        "title": "Progress recap",
        "lesson": "Use concrete progress history as the parent strategy.",
    }, client_id="jungle_ai")

    store.save("jungle_ai", items)
    loaded = store.load("jungle_ai")

    assert len(loaded) == 1
    assert loaded[0]["title"] == "Progress recap"
    assert loaded[0]["status"] == "promoted"
