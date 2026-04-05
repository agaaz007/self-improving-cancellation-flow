from __future__ import annotations

import argparse
import errno
import functools
import json
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from cta_autoresearch.lab_dashboard import build_dashboard_dataset, write_dashboard_data
from cta_autoresearch.lab_optimizer import build_report
from cta_autoresearch.personas import generate_personas
from cta_autoresearch.research_settings import DEFAULT_OPENAI_MODEL, build_settings, build_settings_catalog
from cta_autoresearch.sample_data import load_seed_profiles


def _settings_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "population": getattr(args, "population", 120),
        "seed": getattr(args, "seed", 7),
        "top_n": getattr(args, "top_n", 25),
        "strategy_depth": getattr(args, "strategy_depth", "standard"),
        "persona_richness": getattr(args, "persona_richness", "standard"),
        "ideation_agents": getattr(args, "ideation_agents", 4),
        "validation_budget": getattr(args, "validation_budget", None),
        "model_name": getattr(args, "model_name", DEFAULT_OPENAI_MODEL),
        "discount_step": getattr(args, "discount_step", 10),
        "discount_floor": getattr(args, "discount_floor", 0),
        "discount_ceiling": getattr(args, "discount_ceiling", 100),
        "grounding_limit": getattr(args, "grounding_limit", 1),
        "treatment_limit": getattr(args, "treatment_limit", 1),
        "friction_limit": getattr(args, "friction_limit", 1),
        "idea_proposals_per_agent": getattr(args, "idea_proposals_per_agent", 2),
        "persona_shortlist_multiplier": getattr(args, "persona_shortlist_multiplier", 3),
        "segment_focus_limit": getattr(args, "segment_focus_limit", 5),
        "archetype_template_count": getattr(args, "archetype_template_count", 0),
        "persona_blend_every": getattr(args, "persona_blend_every", 0),
        "openai_reasoning_effort": getattr(args, "openai_reasoning_effort", "medium"),
        "api_batch_size": getattr(args, "api_batch_size", 4),
    }


def run_sample(args: argparse.Namespace) -> None:
    settings = build_settings(_settings_from_args(args))
    seed_profiles = load_seed_profiles()
    personas = generate_personas(
        seed_profiles,
        population=settings.population,
        seed=settings.seed,
        richness=settings.persona_richness,
        archetype_template_count=settings.archetype_template_count or None,
        blend_every=settings.persona_blend_every or None,
    )
    report, metrics = build_report(personas, top_n=settings.top_n, settings=settings)

    print("---")
    for key in (
        "baseline_retention_score",
        "expected_retention_score",
        "estimated_lift",
        "trust_safety_score",
        "personas_evaluated",
        "search_space_size",
        "validated_strategy_count",
        "top_strategy",
        "best_non_discount_strategy",
        "model_name",
    ):
        print(f"{key}: {metrics[key]}")

    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report)


def _json_response(handler, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _format_run_command(settings: dict[str, object]) -> str:
    args = [
        "PYTHONPATH=src python3 -m cta_autoresearch.lab_cli run-sample",
        f"--population {settings.get('population', 120)}",
        f"--strategy-depth {settings.get('strategy_depth', 'standard')}",
        f"--persona-richness {settings.get('persona_richness', 'rich')}",
        f"--ideation-agents {settings.get('ideation_agents', 5)}",
        f"--validation-budget {settings.get('validation_budget', 240)}",
        f"--model-name {settings.get('model_name', DEFAULT_OPENAI_MODEL)}",
        f"--top-n {settings.get('top_n', 25)}",
        f"--seed {settings.get('seed', 7)}",
    ]
    return " ".join(args)


def _make_run_manager(directory: Path, defaults: dict[str, object]):
    state = {"lock": threading.Lock(), "jobs": {}, "order": []}

    def summarize(job: dict, *, include_result: bool = False) -> dict:
        elapsed = max(0.0, (job.get("finished_at") or time.time()) - job["created_at"])
        payload = {
            "id": job["id"],
            "status": job["status"],
            "progress": round(job["progress"], 4),
            "stage": job["stage"],
            "message": job["message"],
            "created_at": job["created_at"],
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "elapsed_seconds": round(elapsed, 2),
            "eta_seconds": None if job.get("eta_seconds") is None else round(job["eta_seconds"], 2),
            "settings": job["settings"],
            "command_preview": job["command_preview"],
            "activity_log": job["activity_log"][-30:],
            "result_meta": job.get("result_meta"),
            "error": job.get("error"),
        }
        if include_result:
            payload["result"] = job.get("result")
        return payload

    def append_log(job: dict, progress: float, stage: str, message: str) -> None:
        job["progress"] = max(job["progress"], min(progress, 1.0))
        job["stage"] = stage
        job["message"] = message
        elapsed = max(time.time() - job["started_at"], 0.01)
        if job["progress"] > 0.05 and job["progress"] < 1.0:
            remaining = elapsed * ((1.0 - job["progress"]) / job["progress"])
            job["eta_seconds"] = remaining
        else:
            job["eta_seconds"] = None
        if not job["activity_log"] or job["activity_log"][-1]["message"] != message:
            job["activity_log"].append(
                {
                    "ts": time.time(),
                    "stage": stage,
                    "message": message,
                    "progress": round(job["progress"], 4),
                }
            )

    def run_job(job_id: str) -> None:
        with state["lock"]:
            job = state["jobs"][job_id]
            job["status"] = "running"
            job["started_at"] = time.time()
            append_log(job, 0.02, "queued", "Run accepted by the lab server.")
            settings = dict(job["settings"])

        try:
            payload = build_dashboard_dataset(
                progress_callback=lambda progress, stage, message: _update_job(job_id, progress, stage, message),
                **settings,
            )
            with state["lock"]:
                job = state["jobs"][job_id]
                job["result"] = payload
                job["result_meta"] = payload.get("meta")
                job["status"] = "completed"
                job["finished_at"] = time.time()
                append_log(job, 1.0, "complete", "Research run completed and result payload is ready.")
                (directory / "data.json").write_text(json.dumps(payload, indent=2))
        except Exception as exc:  # pragma: no cover
            with state["lock"]:
                job = state["jobs"][job_id]
                job["status"] = "failed"
                job["finished_at"] = time.time()
                job["error"] = str(exc)
                append_log(job, job["progress"], "error", f"Run failed: {exc}")
                job["activity_log"].append(
                    {
                        "ts": time.time(),
                        "stage": "traceback",
                        "message": traceback.format_exc().strip(),
                        "progress": round(job["progress"], 4),
                    }
                )

    def _update_job(job_id: str, progress: float, stage: str, message: str) -> None:
        with state["lock"]:
            job = state["jobs"].get(job_id)
            if job is None:
                return
            append_log(job, progress, stage, message)

    def create_job(settings: dict[str, object]) -> dict:
        normalized_settings = {**defaults, **settings}
        job_id = uuid.uuid4().hex[:10]
        job = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "stage": "queued",
            "message": "Waiting to start.",
            "created_at": time.time(),
            "settings": normalized_settings,
            "command_preview": _format_run_command(normalized_settings),
            "activity_log": [],
            "result": None,
            "result_meta": None,
            "eta_seconds": None,
            "error": None,
        }
        with state["lock"]:
            state["jobs"][job_id] = job
            state["order"].insert(0, job_id)
            state["order"] = state["order"][:12]
        thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
        thread.start()
        return summarize(job)

    def list_jobs() -> dict:
        with state["lock"]:
            return {"runs": [summarize(state["jobs"][job_id]) for job_id in state["order"] if job_id in state["jobs"]]}

    def get_job(job_id: str) -> dict | None:
        with state["lock"]:
            job = state["jobs"].get(job_id)
            if job is None:
                return None
            return summarize(job, include_result=True)

    return {
        "create_job": create_job,
        "list_jobs": list_jobs,
        "get_job": get_job,
    }


def _make_handler(directory: Path, defaults: dict[str, object]):
    import http.server
    run_manager = _make_run_manager(directory, defaults)

    class DashboardHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/research-runs":
                _json_response(self, run_manager["list_jobs"]())
                return
            if parsed.path.startswith("/api/research-runs/"):
                run_id = parsed.path.rsplit("/", 1)[-1]
                payload = run_manager["get_job"](run_id)
                if payload is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown run id")
                    return
                _json_response(self, payload)
                return
            if parsed.path in {"/data.json", "/api/research"}:
                query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                payload = build_dashboard_dataset(**{**defaults, **query})
                _json_response(self, payload)
                return
            if parsed.path == "/api/controls":
                _json_response(self, build_settings_catalog())
                return
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/research", "/api/research-runs"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
                return

            body_settings = payload.get("settings", payload)
            if not isinstance(body_settings, dict):
                self.send_error(HTTPStatus.BAD_REQUEST, "Expected JSON object for settings")
                return

            if parsed.path == "/api/research-runs":
                result = run_manager["create_job"](body_settings)
                _json_response(self, result, status=HTTPStatus.ACCEPTED)
                return

            result = build_dashboard_dataset(**{**defaults, **body_settings})
            _json_response(self, result)

    return DashboardHandler


def add_shared_research_args(parser: argparse.ArgumentParser, *, dashboard_defaults: bool = False) -> None:
    parser.add_argument("--population", type=int, default=120 if dashboard_defaults else 60)
    parser.add_argument("--top-n", type=int, default=25 if dashboard_defaults else 5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strategy-depth", type=str, default="standard")
    parser.add_argument("--persona-richness", type=str, default="standard")
    parser.add_argument("--ideation-agents", type=int, default=4)
    parser.add_argument("--validation-budget", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--discount-step", type=int, default=10)
    parser.add_argument("--discount-floor", type=int, default=0)
    parser.add_argument("--discount-ceiling", type=int, default=100)
    parser.add_argument("--grounding-limit", type=int, default=1)
    parser.add_argument("--treatment-limit", type=int, default=1)
    parser.add_argument("--friction-limit", type=int, default=1)
    parser.add_argument("--idea-proposals-per-agent", type=int, default=2)
    parser.add_argument("--persona-shortlist-multiplier", type=int, default=3)
    parser.add_argument("--segment-focus-limit", type=int, default=5)
    parser.add_argument("--archetype-template-count", type=int, default=0)
    parser.add_argument("--persona-blend-every", type=int, default=0)
    parser.add_argument("--openai-reasoning-effort", type=str, default="medium")
    parser.add_argument("--api-batch-size", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CTA autoresearch lab")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run-sample", help="Evaluate sample personas and emit a report.")
    add_shared_research_args(run)
    run.add_argument("--output", type=str, default="")

    dashboard = subparsers.add_parser("build-dashboard", help="Generate dashboard data.json.")
    add_shared_research_args(dashboard, dashboard_defaults=True)
    dashboard.add_argument("--output-dir", type=str, default="dashboard")

    serve = subparsers.add_parser("serve-dashboard", help="Serve the dashboard with live research controls.")
    add_shared_research_args(serve, dashboard_defaults=True)
    serve.add_argument("--output-dir", type=str, default="dashboard")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "run-sample"):
        run_sample(args)
        return

    if args.command == "build-dashboard":
        output = write_dashboard_data(args.output_dir, **_settings_from_args(args))
        print(f"Wrote dashboard data to {output}")
        return

    if args.command == "serve-dashboard":
        directory = Path(args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        defaults = _settings_from_args(args)
        import http.server

        handler = _make_handler(directory, defaults)
        try:
            with http.server.ThreadingHTTPServer(("", args.port), handler) as httpd:
                print(f"Serving dashboard at http://127.0.0.1:{args.port}")
                httpd.serve_forever()
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                parser.error(
                    f"Port {args.port} is already in use. Stop the existing dashboard server or choose a different --port."
                )
            raise
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
