"""Production harness server.

Ties the Karpathy research backend, experiment deployment, and feedback
collection into a single HTTP API. This is the closed-loop system:

  Research Run  -->  Experiment (variants + traffic split)
       ^                    |
       |                    v
  Learning payload    Client gets variant via /api/assign
       ^                    |
       |                    v
  Feedback store  <--  Outcome reported via /api/feedback
"""
from __future__ import annotations

import argparse
import errno
import json
import time
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlparse

from cta_autoresearch.cancel_policy import (
    CancelContextV1,
    CancelOutcomeV1,
    CancelPolicyRuntime,
    TranscriptExtractor,
)
from cta_autoresearch.deployment import (
    ExperimentStore,
    assign_variant,
    create_experiment_from_run,
)
from cta_autoresearch.feedback import FeedbackEvent, FeedbackStore


DATA_DIR = Path("harness_data")


def _json_response(handler, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status.value)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _error_response(handler, status: HTTPStatus, message: str) -> None:
    _json_response(handler, {"error": message}, status)


def _read_json_body(handler) -> dict | None:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if not content_length:
        return {}
    raw = handler.rfile.read(content_length)
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def make_handler(
    experiment_store: ExperimentStore,
    feedback_store: FeedbackStore,
    run_result_dir: Path,
    policy_runtime: CancelPolicyRuntime,
    transcript_extractor: TranscriptExtractor,
):
    import http.server

    class HarnessHandler(http.server.BaseHTTPRequestHandler):

        def do_OPTIONS(self) -> None:
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            path = urlparse(self.path).path

            # -- Health check --
            if path == "/api/health":
                live = experiment_store.get_live()
                _json_response(self, {
                    "status": "ok",
                    "has_live_experiment": live is not None,
                    "live_experiment_id": live.id if live else None,
                    "policy": policy_runtime.health(),
                    "extractor": transcript_extractor.health(),
                })
                return

            # -- v1 cancellation policy health --
            if path == "/v1/cancel/policy/health":
                _json_response(
                    self,
                    {
                        "status": "ok",
                        "policy": policy_runtime.health(),
                        "extractor": transcript_extractor.health(),
                        "actions": policy_runtime.list_actions(),
                    },
                )
                return

            # -- List experiments --
            if path == "/api/experiments":
                experiments = experiment_store.list_all()
                _json_response(self, {
                    "experiments": [e.to_dict() for e in experiments],
                })
                return

            # -- Get single experiment --
            if path.startswith("/api/experiments/") and path.count("/") == 3:
                exp_id = path.rsplit("/", 1)[-1]
                exp = experiment_store.get(exp_id)
                if exp is None:
                    _error_response(self, HTTPStatus.NOT_FOUND, "Experiment not found")
                    return
                _json_response(self, exp.to_dict())
                return

            # -- Assign variant to user (production client endpoint) --
            if path == "/api/assign":
                _error_response(self, HTTPStatus.METHOD_NOT_ALLOWED, "Use POST for /api/assign")
                return

            # -- Get experiment report --
            if path.startswith("/api/reports/"):
                exp_id = path.rsplit("/", 1)[-1]
                exp = experiment_store.get(exp_id)
                if exp is None:
                    _error_response(self, HTTPStatus.NOT_FOUND, "Experiment not found")
                    return
                variant_names = {v.id: v.name for v in exp.variants}
                control_id = next(
                    (v.id for v in exp.variants if "control" in v.name.lower()),
                    exp.variants[-1].id if exp.variants else None,
                )
                report = feedback_store.build_experiment_report(
                    exp_id, variant_names, control_id
                )
                _json_response(self, report.to_dict())
                return

            # -- Get learning payload for next research run --
            if path.startswith("/api/learning/"):
                exp_id = path.rsplit("/", 1)[-1]
                payload = feedback_store.build_learning_payload(exp_id)
                _json_response(self, payload)
                return

            _error_response(self, HTTPStatus.NOT_FOUND, f"Unknown path: {path}")

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            body = _read_json_body(self)
            if body is None:
                _error_response(self, HTTPStatus.BAD_REQUEST, "Invalid JSON body")
                return

            # -- Create experiment from a research run result --
            if path == "/api/experiments":
                source_run_id = body.get("source_run_id", "")
                traffic_split = body.get("traffic_split")
                name = body.get("name", "")

                # Load run result from file or inline
                run_result = body.get("run_result")
                if not run_result and source_run_id:
                    result_path = run_result_dir / source_run_id / "result.json"
                    if result_path.exists():
                        run_result = json.loads(result_path.read_text())

                if not run_result:
                    _error_response(
                        self, HTTPStatus.BAD_REQUEST,
                        "Provide run_result inline or a valid source_run_id"
                    )
                    return

                try:
                    experiment = create_experiment_from_run(
                        run_result, source_run_id, traffic_split, name
                    )
                    experiment_store.save(experiment)
                    _json_response(self, experiment.to_dict(), HTTPStatus.CREATED)
                except ValueError as exc:
                    _error_response(self, HTTPStatus.BAD_REQUEST, str(exc))
                return

            # -- Deploy an experiment (make it live) --
            if path.startswith("/api/experiments/") and path.endswith("/deploy"):
                exp_id = path.split("/")[-2]
                try:
                    exp = experiment_store.deploy(exp_id)
                    _json_response(self, exp.to_dict())
                except ValueError as exc:
                    _error_response(self, HTTPStatus.NOT_FOUND, str(exc))
                return

            # -- Stop an experiment --
            if path.startswith("/api/experiments/") and path.endswith("/stop"):
                exp_id = path.split("/")[-2]
                try:
                    exp = experiment_store.stop(exp_id)
                    _json_response(self, exp.to_dict())
                except ValueError as exc:
                    _error_response(self, HTTPStatus.NOT_FOUND, str(exc))
                return

            # -- Assign variant (called by production client) --
            if path == "/api/assign":
                user_id = body.get("user_id", "")
                if not user_id:
                    _error_response(self, HTTPStatus.BAD_REQUEST, "user_id is required")
                    return

                exp = experiment_store.get_live()
                if exp is None:
                    _error_response(
                        self, HTTPStatus.SERVICE_UNAVAILABLE,
                        "No live experiment. Client should show default UX."
                    )
                    return

                try:
                    variant = assign_variant(exp, user_id)
                except ValueError as exc:
                    _error_response(self, HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                    return

                user_context = body.get("user_context", {})
                feedback_store.record_impression(exp.id, variant.id, user_id)

                _json_response(self, {
                    "experiment_id": exp.id,
                    "variant_id": variant.id,
                    "variant_name": variant.name,
                    "component": variant.render_component(user_context),
                })
                return

            # -- Record feedback (called by production client after outcome) --
            if path == "/api/feedback":
                user_id = body.get("user_id", "")
                experiment_id = body.get("experiment_id", "")
                variant_id = body.get("variant_id", "")
                outcome = body.get("outcome", "")

                if not all([user_id, experiment_id, variant_id, outcome]):
                    _error_response(
                        self, HTTPStatus.BAD_REQUEST,
                        "Required: user_id, experiment_id, variant_id, outcome"
                    )
                    return

                try:
                    event = FeedbackEvent(
                        user_id=user_id,
                        experiment_id=experiment_id,
                        variant_id=variant_id,
                        outcome=outcome,
                        meta=body.get("meta", {}),
                    )
                    feedback_store.record_outcome(event)
                    _json_response(self, {"status": "recorded"}, HTTPStatus.CREATED)
                except ValueError as exc:
                    _error_response(self, HTTPStatus.BAD_REQUEST, str(exc))
                return

            # -- Extract transcript into structured context --
            if path == "/v1/cancel/transcript/extract":
                transcript = body.get("transcript")
                if transcript in (None, ""):
                    _error_response(self, HTTPStatus.BAD_REQUEST, "transcript is required")
                    return
                try:
                    extraction = transcript_extractor.extract(
                        transcript,
                        metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
                    )
                except ValueError as exc:
                    _error_response(self, HTTPStatus.BAD_REQUEST, str(exc))
                    return
                _json_response(self, extraction.to_dict(), HTTPStatus.OK)
                return

            # -- Decide cancellation action --
            if path == "/v1/cancel/policy/decide":
                payload = dict(body)
                extraction_payload = payload.get("transcript_extraction")
                if not isinstance(extraction_payload, dict):
                    transcript = payload.get("transcript")
                    if transcript in (None, ""):
                        _error_response(
                            self,
                            HTTPStatus.BAD_REQUEST,
                            "Provide transcript_extraction or transcript.",
                        )
                        return
                    extraction = transcript_extractor.extract(
                        transcript,
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
                    )
                    payload["transcript_extraction"] = extraction.to_dict()

                try:
                    context = CancelContextV1.from_dict(payload)
                    decision = policy_runtime.decide(context)
                except ValueError as exc:
                    _error_response(self, HTTPStatus.BAD_REQUEST, str(exc))
                    return

                response = decision.to_dict()
                response["action"] = policy_runtime.get_action(decision.action_id)
                response["context_version"] = context.context_version
                _json_response(self, response, HTTPStatus.OK)
                return

            # -- Record cancellation outcome --
            if path == "/v1/cancel/policy/outcome":
                try:
                    outcome = CancelOutcomeV1.from_dict(body)
                    result = policy_runtime.record_outcome(outcome)
                except ValueError as exc:
                    _error_response(self, HTTPStatus.BAD_REQUEST, str(exc))
                    return
                _json_response(self, result, HTTPStatus.CREATED)
                return

            # -- Warm-start policy from historical rows --
            if path == "/v1/cancel/policy/warm-start":
                rows = body.get("rows", body.get("historical_rows", []))
                if not isinstance(rows, list):
                    _error_response(self, HTTPStatus.BAD_REQUEST, "rows must be an array")
                    return
                result = policy_runtime.warm_start(
                    rows,
                    reset_state=bool(body.get("reset_state", False)),
                )
                _json_response(self, result, HTTPStatus.OK)
                return

            # -- Offline replay report --
            if path == "/v1/cancel/evals/replay":
                rows = body.get("rows")
                if rows is not None and not isinstance(rows, list):
                    _error_response(self, HTTPStatus.BAD_REQUEST, "rows must be an array when provided")
                    return
                report = policy_runtime.replay(rows=rows)
                _json_response(self, report, HTTPStatus.OK)
                return

            # -- Regression check gate --
            if path == "/v1/cancel/evals/regression-check":
                try:
                    min_treatment_samples = int(body.get("min_treatment_samples", 60))
                    min_holdout_samples = int(body.get("min_holdout_samples", 20))
                    min_save_lift = float(body.get("min_save_lift", 0.0))
                    max_support_delta = float(body.get("max_support_delta", 0.03))
                    max_complaint_delta = float(body.get("max_complaint_delta", 0.02))
                except (TypeError, ValueError):
                    _error_response(self, HTTPStatus.BAD_REQUEST, "Regression thresholds must be numeric.")
                    return
                report = policy_runtime.regression_check(
                    min_treatment_samples=min_treatment_samples,
                    min_holdout_samples=min_holdout_samples,
                    min_save_lift=min_save_lift,
                    max_support_delta=max_support_delta,
                    max_complaint_delta=max_complaint_delta,
                )
                _json_response(self, report, HTTPStatus.OK)
                return

            _error_response(self, HTTPStatus.NOT_FOUND, f"Unknown path: {path}")

        def log_message(self, format: str, *args: object) -> None:
            pass  # Suppress default access logs

    return HarnessHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CTA production harness server")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--data-dir", type=str, default="harness_data",
        help="Root directory for experiment and feedback storage",
    )
    parser.add_argument(
        "--run-result-dir", type=str, default="runs",
        help="Directory containing research run results (from RunStore)",
    )
    parser.add_argument(
        "--policy-dir", type=str, default="",
        help="Directory for policy state and decision/outcome logs. Defaults to <data-dir>/policy.",
    )
    parser.add_argument(
        "--warm-start-file", type=str, default="",
        help="Optional JSON file with historical rows to warm-start posteriors.",
    )
    parser.add_argument(
        "--extractor-model", type=str, default="gpt-5.4-mini",
        help="Model name for transcript extraction when OPENAI_API_KEY is available.",
    )
    parser.add_argument(
        "--extractor-backend", type=str, default="auto", choices=("auto", "heuristic", "openai"),
        help="Transcript extractor backend selection.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    experiment_store = ExperimentStore(data_dir / "experiments")
    feedback_store = FeedbackStore(data_dir / "feedback")
    run_result_dir = Path(args.run_result_dir)
    policy_dir = Path(args.policy_dir) if args.policy_dir else data_dir / "policy"
    use_openai = args.extractor_backend in {"auto", "openai"}
    transcript_extractor = TranscriptExtractor(
        model_name=args.extractor_model,
        use_openai=use_openai,
    )
    policy_runtime = CancelPolicyRuntime(policy_dir)

    if args.warm_start_file:
        warm_start_path = Path(args.warm_start_file)
        if warm_start_path.exists():
            try:
                payload = json.loads(warm_start_path.read_text())
            except json.JSONDecodeError:
                parser.error(f"Warm-start file must contain valid JSON: {warm_start_path}")
                return
            rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
            if isinstance(rows, list):
                summary = policy_runtime.warm_start(rows, reset_state=False)
                print(f"Warm-start loaded: {summary['rows_applied']} rows applied.")
            else:
                parser.error("--warm-start-file must contain an array or an object with a 'rows' array.")
        else:
            parser.error(f"Warm-start file not found: {warm_start_path}")

    handler_class = make_handler(
        experiment_store,
        feedback_store,
        run_result_dir,
        policy_runtime,
        transcript_extractor,
    )

    import http.server
    try:
        with http.server.ThreadingHTTPServer(("", args.port), handler_class) as httpd:
            print(f"Harness server running at http://127.0.0.1:{args.port}")
            print(f"  Experiments: {data_dir / 'experiments'}")
            print(f"  Feedback:    {data_dir / 'feedback'}")
            print(f"  Run results: {run_result_dir}")
            print()
            print("Endpoints:")
            print("  POST /api/experiments          Create experiment from run result")
            print("  POST /api/experiments/:id/deploy  Deploy (make live)")
            print("  POST /api/experiments/:id/stop    Stop experiment")
            print("  POST /api/assign               Get variant for user (production)")
            print("  POST /api/feedback             Record outcome")
            print("  GET  /api/reports/:id          Variant performance report")
            print("  GET  /api/learning/:id         Feedback payload for next run")
            print("  GET  /api/health               Health check")
            print("  POST /v1/cancel/transcript/extract")
            print("  POST /v1/cancel/policy/decide")
            print("  POST /v1/cancel/policy/outcome")
            print("  POST /v1/cancel/policy/warm-start")
            print("  POST /v1/cancel/evals/replay")
            print("  POST /v1/cancel/evals/regression-check")
            print("  GET  /v1/cancel/policy/health")
            httpd.serve_forever()
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.error(f"Port {args.port} is already in use.")
        raise


if __name__ == "__main__":
    main()
