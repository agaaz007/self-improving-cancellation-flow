from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from cta_autoresearch.config import ResearchSettings, build_persona_cohort
from cta_autoresearch.dashboard_builder import build_dashboard_dataset, write_dashboard_data
from cta_autoresearch.optimizer import build_report
from cta_autoresearch.sample_data import load_seed_profiles


def run_sample(settings: ResearchSettings, top_n: int, output: str | None) -> None:
    seed_profiles = load_seed_profiles()
    personas = build_persona_cohort(seed_profiles=seed_profiles, settings=settings)
    report, metrics = build_report(personas, top_n=top_n, settings=settings)

    print("---")
    for key in (
        "baseline_retention_score",
        "expected_retention_score",
        "estimated_lift",
        "trust_safety_score",
        "personas_evaluated",
        "search_space_size",
        "structural_candidates_evaluated",
        "generated_candidates_proposed",
        "top_strategy",
        "best_non_discount_strategy",
    ):
        if key in metrics:
            print(f"{key}: {metrics[key]}")

    for key, value in (
        ("research_depth", settings.depth),
        ("persona_richness", settings.persona_richness),
        ("ideation_rounds", settings.ideation_rounds),
        ("strategy_budget", settings.strategy_budget or settings.effective_workbench_limit),
        ("configured_model", settings.model),
        ("active_model", settings.active_model),
        ("use_llm", settings.use_llm),
        ("llm_runtime_state", settings.llm_runtime_state),
    ):
        print(f"{key}: {value}")

    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report)


def add_research_settings_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--population", type=int, default=120, help="Base persona population before richness/depth expansion.")
    parser.add_argument("--depth", type=int, default=1, help="How broadly to expand the research cohort.")
    parser.add_argument("--strategy-budget", type=int, default=0, help="Generated candidate rows to validate beyond the structural search space.")
    parser.add_argument("--persona-richness", type=int, default=1, help="1=standard, 2=rich, 3+=extreme persona detail.")
    parser.add_argument("--ideation-rounds", type=int, default=1, help="How many independent persona generation rounds to merge.")
    parser.add_argument("--model", type=str, default="heuristic-simulator", help="Configured backend model name for metadata and future LLM mode.")
    parser.add_argument("--use-llm", action="store_true", help="Record that the run should use LLM-assisted ideation when wired.")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="Environment variable name that would hold the LLM API key.")
    parser.add_argument("--seed", type=int, default=7)


def _settings_from_query(parsed_query: dict[str, list[str]], base: ResearchSettings) -> ResearchSettings:
    def first(name: str, default):
        values = parsed_query.get(name)
        return values[0] if values else default

    return ResearchSettings(
        base_population=int(first("population", base.base_population)),
        depth=int(first("depth", base.depth)),
        strategy_budget=int(first("strategy_budget", base.strategy_budget)),
        persona_richness=int(first("persona_richness", base.persona_richness)),
        ideation_rounds=int(first("ideation_rounds", base.ideation_rounds)),
        model=str(first("model", base.model)),
        use_llm=str(first("use_llm", "1" if base.use_llm else "0")).lower() in {"1", "true", "yes", "on"},
        api_key_env=str(first("api_key_env", base.api_key_env)),
        seed=int(first("seed", base.seed)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CTA autoresearch prototype")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run-sample", help="Evaluate sample personas and emit a report.")
    run.add_argument("--top-n", type=int, default=5)
    run.add_argument("--output", type=str, default="")
    add_research_settings_arguments(run)

    dashboard = subparsers.add_parser("build-dashboard", help="Generate dashboard data.json.")
    dashboard.add_argument("--output-dir", type=str, default="dashboard")
    add_research_settings_arguments(dashboard)

    serve = subparsers.add_parser("serve-dashboard", help="Serve the dashboard directory plus a live research API.")
    serve.add_argument("--output-dir", type=str, default="dashboard")
    serve.add_argument("--port", type=int, default=8000)
    add_research_settings_arguments(serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = ResearchSettings.from_namespace(args)

    if args.command in (None, "run-sample"):
        top_n = getattr(args, "top_n", 5)
        output = getattr(args, "output", "")
        run_sample(settings=settings, top_n=top_n, output=output or None)
        return

    if args.command == "build-dashboard":
        output = write_dashboard_data(output_dir=args.output_dir, settings=settings)
        print(f"Wrote dashboard data to {output}")
        return

    if args.command == "serve-dashboard":
        output = write_dashboard_data(output_dir=args.output_dir, settings=settings)
        print(f"Wrote dashboard data to {output}")
        directory = Path(args.output_dir)
        default_settings = settings

        import http.server
        import socketserver

        class DashboardHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *handler_args, **handler_kwargs):
                super().__init__(*handler_args, directory=str(directory), **handler_kwargs)

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/api/research":
                    live_settings = _settings_from_query(parse_qs(parsed.query), default_settings)
                    payload = build_dashboard_dataset(settings=live_settings)
                    body = json.dumps(payload).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                super().do_GET()

        handler = functools.partial(DashboardHandler)
        with socketserver.TCPServer(("", args.port), handler) as httpd:
            print(f"Serving dashboard at http://127.0.0.1:{args.port}")
            httpd.serve_forever()
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
