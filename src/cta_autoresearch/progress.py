from __future__ import annotations

from typing import Callable


ProgressCallback = Callable[[str, float, str, dict[str, object] | None], None]


def emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    progress: float,
    message: str,
    details: dict[str, object] | None = None,
) -> None:
    if callback is None:
        return
    callback(stage, max(0.0, min(progress, 1.0)), message, details)
