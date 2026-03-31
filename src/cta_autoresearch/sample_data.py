from __future__ import annotations

import json
from pathlib import Path

from cta_autoresearch.models import UserProfile


def load_seed_profiles(path: str | Path | None = None) -> list[UserProfile]:
    source = Path(path) if path else Path("data/jungle_ai_seed_profiles.json")
    payload = json.loads(source.read_text())
    return [UserProfile.from_dict(item) for item in payload]
