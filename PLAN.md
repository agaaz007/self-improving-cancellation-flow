# Plan: Jungle AI Exit CTA Optimization Lab

Generated on 2026-03-31  
Status: Ready for prototype implementation

## Vision

Build a retention research lab that converts product analytics into better exit-intervention decisions: which message angle, which incentive, and which CTA should appear for a specific canceling user at a specific moment.

The 10-star version is not “generate copy with AI.” It is a system that learns:

- which user signals matter most during cancellation,
- which persuasion style fits each user segment,
- when discounting helps versus destroys margin, and
- when the right move is pause, downgrade, support, or direct feedback instead of “stay now.”

## Accepted Scope

- Ingest amplitude-like behavioral profiles and derive interpretable features
- Generate synthetic user personas from seed profiles plus edtech churn archetypes
- Score message angle + offer + CTA combinations on a simulated cohort
- Output ranked strategies overall and per persona segment
- Adapt the `autoresearch` pattern so future agent iterations can improve one strategy file against a fixed harness

## Not In Scope

- Direct production integration with Amplitude, Mixpanel, PostHog, or Jungle AI APIs
- True causal inference from historical treatment data
- Real-time experimentation platform, assignment service, or deployment pipeline
- LLM-generated copy in the loop for every run

## CEO Review

### Scope decision

Selective expansion was the right move.

Why:

- The base idea is already valuable.
- The hidden 10x move is not “more copy variants.” It is policy learning: knowing when to avoid discounts and when to offer a softer landing like pause.
- The dangerous trap is building a giant personalization engine before validating the decision surfaces.

### Product corrections

- Optimize for **decision quality**, not just message generation volume.
- Treat **offer selection** as equally important as copy selection.
- Add **trust-safety** as a first-class metric so the system does not become creepy or overfit on invasive personalization.
- Start with the narrowest wedge:
  edtech monthly subscription cancellation flow for students with clear study-context signals.

## Architecture Review

### System shape

The prototype should use a fixed evaluation harness with one strategy policy file that is intentionally easy to mutate.

```text
seed profiles
    |
    v
feature derivation
    |
    v
synthetic persona generator
    |
    v
strategy enumerator ----> strategy policy (mutable)
    |                           |
    v                           |
simulation scorer <-------------
    |
    v
ranked strategies + report
```

### Why this architecture

- It mirrors `autoresearch`: fixed harness, small mutation surface, fast experiment loop.
- It keeps the system inspectable for growth and lifecycle teams.
- It gives you a bridge from heuristics now to real experimentation later.

## Error And Rescue Map

- Missing or sparse behavioral data:
  fall back to less personalized, higher-trust copy angles.
- Dormant users with little signal:
  prefer feature-reminder or pause flows over hyper-specific callbacks to old behavior.
- Margin-damaging discount spirals:
  add explicit penalties for unnecessary discounts to already engaged users.
- Creepy personalization:
  add trust-safety penalties when copy becomes too specific for low-trust segments.

## Security And Trust Model

- Do not store or render raw PII in reports.
- Keep emails out of seed data and synthetic data.
- Never surface private study content in cancellation copy.
- Make “because you used X” explanations coarse enough to feel helpful, not invasive.

## Data Flow And Edge Cases

- Active power users may respond best to progress-reflection and pause, not discounts.
- Dormant free users may need feature education or proof before pricing pressure.
- Monthly quota remaining at full value is a strong “awareness gap” clue, not necessarily low intent.
- Retry-after-mistake behavior is a powerful learning-intent signal and should increase support-oriented messaging weight.

## Test Plan

```text
feature derivation
  |- engaged user > dormant user on habit strength
  |- dormant user > engaged user on awareness gap
  |- pricing-sensitive user > power user on discount affinity

strategy scoring
  |- high-urgency learner should prefer progress/exam-support over generic exit survey
  |- dormant free learner should prefer feature unlock or bonus credits over “stay on current plan”
  |- trust-sensitive user should be penalized for over-specific copy
```

## Performance Review

- Candidate space is intentionally small and enumerable in-memory.
- The main scaling lever is synthetic cohort size.
- Even at a few hundred personas, the current approach is cheap and fast.

## Observability

- Print compact benchmark lines after every CLI run
- Emit markdown reports for human review
- Keep future `results.tsv` records comparable across runs

## Design And UX Review

- Present outputs as side-by-side strategy comparisons, not a single black-box winner
- Always show lift, confidence caveat, and trust-safety score together
- Make the analyst console explain “why this strategy fits this user” in plain language

## Next Build Steps

1. Backtest the heuristic scorer against historical churn interventions if you can recover outcome labels.
2. Add real event schema ingestion from Amplitude exports.
3. Add live experiment templates for the top 3 strategies per segment.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | CLEAR | Narrowed to decision quality, added trust-safety and offer policy |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | CLEAR | Converted the concept into a small fixed-harness prototype instead of a vague AI workflow |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR | Fixed harness + mutable strategy file + explicit test diagram |
| Design Review | `/plan-design-review` | UI/UX gaps | 1 | CLEAR | Added Signal Room direction, anti-slop rules, and analyst-first explanation surfaces |

- **CODEX:** Built a runnable simulation prototype with sample data, tests, and an autoresearch-style control file.
- **CROSS-MODEL:** Product, engineering, and design all converged on the same thesis: optimize policy choice, not just copy generation.
- **UNRESOLVED:** 0
- **VERDICT:** CEO + ENG + DESIGN CLEARED — ready to implement prototype.
