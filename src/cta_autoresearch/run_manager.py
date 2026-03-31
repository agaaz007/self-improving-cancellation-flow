from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
from typing import Callable
from uuid import uuid4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


def _duration_seconds(started_at: datetime | None, finished_at: datetime | None) -> float | None:
    if started_at is None:
        return None
    end = finished_at or _utc_now()
    return max((end - started_at).total_seconds(), 0.0)


@dataclass
class ResearchRun:
    id: str
    settings: dict[str, object]
    status: str = "queued"
    progress: int = 0
    stage: str = "queued"
    message: str = "Queued"
    created_at: datetime = field(default_factory=_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    events: list[dict[str, object]] = field(default_factory=list)
    result_path: str | None = None
    result_summary: dict[str, object] = field(default_factory=dict)
    error: str | None = None

    def to_summary(self, *, include_events: bool = False) -> dict[str, object]:
        duration = _duration_seconds(self.started_at, self.finished_at)
        eta_seconds = None
        if self.status == "running" and self.progress > 0 and duration is not None:
            eta_seconds = max((duration / max(self.progress, 1)) * (100 - self.progress), 0.0)
        payload = {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "created_at": _isoformat(self.created_at),
            "started_at": _isoformat(self.started_at),
            "finished_at": _isoformat(self.finished_at),
            "duration_seconds": duration,
            "eta_seconds": eta_seconds,
            "settings": self.settings,
            "result_path": self.result_path,
            "result_summary": self.result_summary,
            "error": self.error,
        }
        if include_events:
            payload["events"] = list(self.events)
        return payload


class RunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_metadata(self, run: ResearchRun) -> None:
        meta_path = self._run_dir(run.id) / "meta.json"
        meta_path.write_text(json.dumps(run.to_summary(include_events=True), indent=2))

    def save_result(self, run: ResearchRun, payload: dict[str, object]) -> str:
        result_path = self._run_dir(run.id) / "result.json"
        result_path.write_text(json.dumps(payload, indent=2))
        return str(result_path)

    def load_result(self, run_id: str) -> dict[str, object] | None:
        result_path = self._run_dir(run_id) / "result.json"
        if not result_path.exists():
            return None
        return json.loads(result_path.read_text())

    def load_runs(self) -> list[ResearchRun]:
        runs: list[ResearchRun] = []
        for meta_path in sorted(self.root.glob("*/meta.json"), reverse=True):
            payload = json.loads(meta_path.read_text())
            run = ResearchRun(
                id=str(payload["id"]),
                settings=dict(payload.get("settings") or {}),
                status=str(payload.get("status") or "queued"),
                progress=int(payload.get("progress") or 0),
                stage=str(payload.get("stage") or "queued"),
                message=str(payload.get("message") or "Queued"),
                created_at=datetime.fromisoformat(str(payload["created_at"]).replace("Z", "+00:00")),
                started_at=datetime.fromisoformat(str(payload["started_at"]).replace("Z", "+00:00")) if payload.get("started_at") else None,
                finished_at=datetime.fromisoformat(str(payload["finished_at"]).replace("Z", "+00:00")) if payload.get("finished_at") else None,
                events=list(payload.get("events") or []),
                result_path=payload.get("result_path"),
                result_summary=dict(payload.get("result_summary") or {}),
                error=payload.get("error"),
            )
            runs.append(run)
        return runs


ProgressCallback = Callable[[float, str, str], None]
RunBuilder = Callable[[dict[str, object], ProgressCallback], dict[str, object]]


class RunManager:
    def __init__(self, *, store: RunStore, builder: RunBuilder, max_workers: int = 2) -> None:
        self.store = store
        self.builder = builder
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cta-run")
        self.lock = RLock()
        self.runs: dict[str, ResearchRun] = {run.id: run for run in self.store.load_runs()}
        self.futures: dict[str, Future[None]] = {}

    def list_runs(self) -> list[dict[str, object]]:
        with self.lock:
            runs = sorted(self.runs.values(), key=lambda run: run.created_at, reverse=True)
            return [run.to_summary() for run in runs]

    def get_run(self, run_id: str) -> ResearchRun | None:
        with self.lock:
            return self.runs.get(run_id)

    def get_run_summary(self, run_id: str, *, include_events: bool = False) -> dict[str, object] | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        return run.to_summary(include_events=include_events)

    def get_result(self, run_id: str) -> dict[str, object] | None:
        return self.store.load_result(run_id)

    def create_run(self, settings: dict[str, object]) -> dict[str, object]:
        run = ResearchRun(
            id=f"run_{uuid4().hex[:10]}",
            settings=dict(settings),
            message="Queued on the backend",
        )
        run.events.append(
            {
                "timestamp": _isoformat(run.created_at),
                "stage": "queued",
                "progress": 0,
                "message": "Run accepted by backend",
            }
        )
        with self.lock:
            self.runs[run.id] = run
            self.store.save_metadata(run)
            self.futures[run.id] = self.executor.submit(self._execute_run, run.id)
            return run.to_summary(include_events=True)

    def _apply_progress(self, run_id: str, progress_fraction: float, stage: str, message: str) -> None:
        with self.lock:
            run = self.runs[run_id]
            progress = int(round(max(0.0, min(progress_fraction, 1.0)) * 100))
            run.progress = max(run.progress, min(progress, 100))
            run.stage = stage or run.stage
            run.message = message or run.message
            event_payload = {
                "timestamp": _isoformat(_utc_now()),
                "stage": run.stage,
                "progress": run.progress,
                "message": run.message,
            }
            run.events.append(event_payload)
            self.store.save_metadata(run)

    def _execute_run(self, run_id: str) -> None:
        with self.lock:
            run = self.runs[run_id]
            run.status = "running"
            run.started_at = _utc_now()
            run.message = "Starting research pipeline"
            run.events.append(
                {
                    "timestamp": _isoformat(run.started_at),
                    "stage": "started",
                    "progress": 1,
                    "message": "Research run started",
                }
            )
            self.store.save_metadata(run)

        def progress_callback(progress_fraction: float, stage: str, message: str) -> None:
            self._apply_progress(run_id, progress_fraction, stage, message)

        try:
            payload = self.builder(run.settings, progress_callback)
            with self.lock:
                run = self.runs[run_id]
                run.status = "completed"
                run.progress = 100
                run.stage = "completed"
                run.message = "Research run completed"
                run.finished_at = _utc_now()
                run.result_path = self.store.save_result(run, payload)
                meta = payload.get("meta") or {}
                run.result_summary = {
                    "top_strategy": meta.get("top_strategy"),
                    "top_score": meta.get("top_score"),
                    "model_backend": meta.get("model_backend"),
                    "warnings": meta.get("warnings"),
                }
                run.events.append(
                    {
                        "timestamp": _isoformat(run.finished_at),
                        "stage": "completed",
                        "progress": 100,
                        "message": "Result written and ready to view",
                    }
                )
                self.store.save_metadata(run)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            with self.lock:
                run = self.runs[run_id]
                run.status = "failed"
                run.finished_at = _utc_now()
                run.error = str(exc)
                run.message = "Research run failed"
                run.events.append(
                    {
                        "timestamp": _isoformat(run.finished_at),
                        "stage": "failed",
                        "progress": run.progress,
                        "message": run.error,
                    }
                )
                self.store.save_metadata(run)
