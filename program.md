# CTA Autoresearch Program

This is an adaptation of the `karpathy/autoresearch` operating model for cancellation-flow optimization.

## Goal

Maximize the quality of exit-flow decisions for Jungle AI style subscription products by improving:

- message angle,
- offer selection,
- CTA choice,

against a fixed synthetic evaluation harness.

## In-scope files

Read these before starting:

- `README.md`
- `PLAN.md`
- `docs/2026-03-31-office-hours-jungle-ai-churn-cta.md`
- `data/jungle_ai_seed_profiles.json`
- `src/cta_autoresearch/features.py`
- `src/cta_autoresearch/personas.py`
- `src/cta_autoresearch/simulator.py`
- `src/cta_autoresearch/strategy_policy.py`
- `src/cta_autoresearch/optimizer.py`

## The mutable surface

Only modify:

- `src/cta_autoresearch/strategy_policy.py`

That file is the equivalent of `train.py` in the original `autoresearch` repo. The rest of the harness should stay fixed so results remain comparable.

## Metric

The CLI prints a summary block like:

```text
---
baseline_retention_score: 0.4231
expected_retention_score: 0.6118
estimated_lift: 0.1887
trust_safety_score: 0.8924
personas_evaluated: 120
top_strategy: progress_reflection + pause_plan + finish_this_exam_with_us
```

Primary objective:

- maximize `expected_retention_score`

Guardrail:

- keep `trust_safety_score >= 0.80`

## Run loop

Use:

```bash
PYTHONPATH=src python3 -m cta_autoresearch.cli run-sample --population 120 --top-n 10 --output outputs/run-report.md > run.log 2>&1
```

Then inspect:

```bash
grep "^baseline_retention_score:\|^expected_retention_score:\|^estimated_lift:\|^trust_safety_score:\|^top_strategy:" run.log
```

If the run crashes:

```bash
tail -n 50 run.log
```

## Results log

Maintain a local `results.tsv` with:

```text
commit	expected_retention_score	trust_safety_score	status	description
```

- `status` is `keep`, `discard`, or `crash`
- keep only strategies that beat the previous best without violating the guardrail

## Research heuristics

- Improve decision fit, not copy novelty
- Penalize unnecessary discounts
- Prefer pause/downgrade style offers when they save margin and preserve trust
- Favor interpretable policy changes over brittle prompt hacks
- If two options score similarly, prefer the simpler rule set
