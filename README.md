# CTA Autoresearch Lab

This project turns your Jungle AI churn-reduction idea into a small, runnable research lab.

It uses the core `autoresearch` pattern from [karpathy/autoresearch](https://github.com/karpathy/autoresearch): keep the evaluation harness mostly fixed, expose one high-leverage strategy file, and let an agent iterate on messaging strategy against a clear metric.

## What is here

- `docs/2026-03-31-office-hours-jungle-ai-churn-cta.md` — the `/office-hours` design doc.
- `PLAN.md` — refined product and engineering plan after `/plan-ceo-review`, `/plan-eng-review`, and `/plan-design-review`.
- `DESIGN.md` — internal tool design system for a retention research console.
- `program.md` — an `autoresearch`-style research loop for this CTA optimization problem.
- `data/jungle_ai_seed_profiles.json` — sanitized seed profiles derived from the sample Amplitude-style user summaries.
- `src/cta_autoresearch/` — the simulator, synthetic persona generator, optimizer, and CLI.
- `tests/` — lightweight regression tests for feature derivation and strategy scoring.
- `repo/` — cloned upstream `karpathy/autoresearch` reference repository.

## Quick start

```bash
cd "/Users/ramesh/Documents/AutoResearch (CTA)"
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m cta_autoresearch.cli run-sample --population 60 --top-n 5 --output outputs/sample_report.md
```

The CLI prints a compact metric block and writes a fuller markdown report to `outputs/sample_report.md`.

To build the analyst UI data and open the dashboard locally:

```bash
PYTHONPATH=src python3 -m cta_autoresearch.cli build-dashboard --population 120 --output-dir dashboard
PYTHONPATH=src python3 -m cta_autoresearch.cli serve-dashboard --population 120 --output-dir dashboard --port 8000
```

The dashboard evaluates the full search space offline, then loads a ranked working set into the browser so the UI stays responsive.

## New swarm lab

There is now a deeper lab path for the exact workflow you asked for:

- multi-agent strategy ideation
- richer persona dossiers
- configurable validation budget
- controllable search depth, persona richness, and ideation agent count
- optional OpenAI-backed ideation model selection

Run it with:

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m cta_autoresearch.lab_cli run-sample --population 120 --strategy-depth deep --persona-richness rich --ideation-agents 6 --validation-budget 240 --top-n 5
PYTHONPATH=src python3 -m cta_autoresearch.lab_cli serve-dashboard --population 120 --strategy-depth deep --persona-richness rich --ideation-agents 6 --validation-budget 240 --port 8000
```

To enable OpenAI-backed ideation safely:

```bash
export OPENAI_API_KEY="your-new-key-here"
PYTHONPATH=src python3 -m cta_autoresearch.lab_cli serve-dashboard --population 120 --strategy-depth deep --persona-richness rich --ideation-agents 6 --validation-budget 240 --model-name gpt-5.4-mini --port 8000
```

To force the hosted lab into API-driven research mode for both ideation and evaluation:

```bash
export OPENAI_API_KEY="your-new-key-here"
PYTHONPATH=src python3 -m cta_autoresearch.lab_cli serve-dashboard --population 120 --strategy-depth quick --persona-richness rich --ideation-agents 6 --validation-budget 120 --execution-mode api_only --model-name gpt-5.4-mini --port 8000
```

Notes:

- `execution_mode=hybrid` uses OpenAI for ideation and the local simulator for validation.
- `execution_mode=api_only` routes both ideation and evaluation through OpenAI, with the dashboard/server staying hosted-safe via env vars.
- Do not hardcode API keys into source files or commit them into the repository.

Current backend truth:

- Validation still runs through the local heuristic simulator.
- The model selector changes ideation mode, not the core scoring math.
- `heuristic-simulator` is the default active backend model today.
- If you supply `OPENAI_API_KEY` and select `gpt-5.4-mini` or `gpt-5.4`, the lab can use OpenAI for idea generation while keeping validation local.

## How this maps to your idea

The lab now optimizes five decision surfaces together:

1. Message angle
2. Proof style
3. Offer or incentive, including discount magnitude
4. Exit CTA
5. Personalization intensity

It does this against a synthetic cohort generated from seed user profiles and edtech retention archetypes. The scoring model is deliberately interpretable so the team can inspect why a strategy wins before wiring it into real experiments.

## Important assumptions

- This is a pre-production research harness, not a real-time decision engine.
- It uses heuristic simulation, not causal ground truth.
- The next real step after this prototype is to validate top strategies with live A/B tests and offline historical backtests.
