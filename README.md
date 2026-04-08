# CTA Autoresearch Lab

This project turns your churn-reduction idea into a small, runnable research lab — and now serves trained retention policies via serverless API endpoints.

It uses the core `autoresearch` pattern from [karpathy/autoresearch](https://github.com/karpathy/autoresearch): keep the evaluation harness mostly fixed, expose one high-leverage strategy file, and let an agent iterate on messaging strategy against a clear metric.

---

## Live API — Retention Offer Endpoints

Two Vercel serverless endpoints serve retention offers for different clients. Both accept a conversation transcript + user context and return a personalized retention UI card.

**Base URL:** `https://semarang-liart.vercel.app`

### Endpoints

| Endpoint | Client | Default Mode | Description |
|---|---|---|---|
| `POST /api/zeo_present_offers` | Zeo Auto (route optimization) | Reason routing | Routes based on cancel reason analysis of 85 real cancel events |
| `POST /api/jungle_present_offers` | Jungle AI (edtech) | Bandit (Thompson sampling) | Uses fully trained policy (300 decisions, 54 strategy generations) |

### Request

**Method:** `POST`
**Content-Type:** `application/json`

```json
{
  "user_id": "user_123",
  "transcript": "I want to cancel because it's too expensive for me right now",
  "metadata": {
    "plan_tier": "starter",
    "tenure_days": 120,
    "engagement_7d": 0.4,
    "engagement_30d": 0.6,
    "session_id": "optional_session_id",
    "prior_cancel_attempts_30d": 0,
    "discount_exposures_30d": 0
  },
  "personalize": true,
  "use_bandit": false
}
```

#### Request Fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `user_id` | string | No | `"anonymous"` | Unique user identifier. Used for sticky assignment in bandit mode. |
| `transcript` | string | No | `""` | Conversation transcript from the 11 Labs agent. Can be the full transcript or last few turns. Empty string triggers silent/webhook churn flow. |
| `metadata` | object | No | `{}` | User context for the bandit. See metadata fields below. |
| `personalize` | bool | No | `true` | Enable LLM-generated copy (uses OpenAI gpt-4o-mini). Falls back to fixed text if disabled or if LLM is unavailable. |
| `use_bandit` | bool | No | Zeo: `false`, Jungle: `true` | `true` = Thompson sampling bandit picks the action. `false` = reason-based routing. |

#### Metadata Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `plan_tier` | string | `"unknown"` | User's current plan. Zeo: `weekly`, `monthly`, `quarterly`, `annual`. Jungle: `free`, `starter`, `super_learner`. |
| `tenure_days` | int | `0` | Days since the user first subscribed. |
| `engagement_7d` | float | `0.0` | Engagement score over the last 7 days (0.0 - 1.0). |
| `engagement_30d` | float | `0.0` | Engagement score over the last 30 days (0.0 - 1.0). |
| `session_id` | string | auto-generated | Optional session identifier for tracking. |
| `prior_cancel_attempts_30d` | int | `0` | Number of prior cancel attempts in the last 30 days. |
| `discount_exposures_30d` | int | `0` | Number of times the user has been shown a discount in the last 30 days. |

### Response

```json
{
  "decision_id": "dec_844d2226718d",
  "action_id": "targeted_discount_20",
  "ui": {
    "header_title": "We Understand Your Concerns",
    "body_text": "As a dedicated student, we know every penny counts. Would a 20% discount help you continue your journey?",
    "cta_button_text": "Get My Discount",
    "cta_button_action": "targeted_discount_20",
    "offer_kind": "discount",
    "secondary_action_text": "No thanks, continue canceling",
    "secondary_action": "dismiss"
  },
  "meta": {
    "policy_version": "v1",
    "exploration_flag": false,
    "primary_reason": "price",
    "frustration_level": 0.72,
    "save_openness": 0.18,
    "client_id": "jungle_ai",
    "personalized": true,
    "routing_mode": "bandit",
    "strategy_dims": {
      "message_angle": "progress_reflection",
      "proof_style": "quantified_outcome",
      "personalization": "behavioral",
      "contextual_grounding": "unused_value",
      "creative_treatment": "comeback_plan",
      "friction_reducer": "smart_resume_date"
    }
  }
}
```

#### Response Fields

| Field | Type | Description |
|---|---|---|
| `decision_id` | string | Unique ID for this decision. Use for outcome tracking. |
| `action_id` | string | The retention action selected. See action tables below. |
| `ui` | object | The UI card data to render. See UI fields below. |
| `meta` | object | Metadata about how the decision was made. |

#### UI Object

| Field | Type | Description |
|---|---|---|
| `header_title` | string | Card headline (3-8 words). |
| `body_text` | string | Card body copy (1-2 sentences). |
| `cta_button_text` | string | Primary call-to-action button label. |
| `cta_button_action` | string | Action ID the CTA maps to (same as top-level `action_id`). |
| `offer_kind` | string | Offer category: `discount`, `pause`, `downgrade`, `extension`, `support`, `credit`, `billing`, `none`. |
| `secondary_action_text` | string | Secondary action label (always "No thanks, continue canceling"). |
| `secondary_action` | string | Always `"dismiss"`. |

#### Meta Object

| Field | Type | Description |
|---|---|---|
| `policy_version` | string | `"v1"` (bandit) or `"reason_routing_v1"` (reason-based). |
| `exploration_flag` | bool | `true` if the bandit is exploring (not exploiting). |
| `primary_reason` | string | Detected cancel reason from the transcript. |
| `frustration_level` | float | 0.0-1.0 frustration score from transcript analysis. |
| `save_openness` | float | 0.0-1.0 score of how open the user is to being retained. |
| `client_id` | string | `"zeo_auto"` or `"jungle_ai"`. |
| `personalized` | bool | `true` if the UI text was LLM-generated, `false` if fixed fallback. |
| `routing_mode` | string | `"bandit"` or `"reason"`. |
| `strategy_dims` | object or null | Optimizer strategy dimensions used for LLM personalization (when available). |

### Available Actions

#### Zeo Auto Actions

| Action ID | Offer Kind | Description |
|---|---|---|
| `control_graceful_exit` | none | Let the user go gracefully |
| `pause_plan` | pause | Pause subscription, resume later |
| `downgrade_basic` | downgrade | Switch to a lighter plan |
| `discount_20` | discount | 20% discount on next cycle |
| `discount_40` | discount | 40% discount on next cycle |
| `discount_50` | discount | 50% discount for one month (max offer) |
| `free_week` | extension | One free week, no commitment |
| `route_fix_commitment` | extension | Empathize + commit to fixing issues + free week |
| `route_optimization_demo` | support | Show how to get more value |
| `fleet_rightsize` | credit | Adjust plan to match current needs |

#### Jungle AI Actions

| Action ID | Offer Kind | Description |
|---|---|---|
| `control_empathic_exit` | none | Empathetic exit, ask what's wrong |
| `pause_plan_relief` | pause | Pause and resume when ready |
| `downgrade_lite_switch` | downgrade | Switch to a lighter plan |
| `targeted_discount_20` | discount | 20% discount |
| `targeted_discount_40` | discount | 40% discount |
| `concierge_recovery` | support | Guided reset from learning support |
| `exam_sprint_focus` | extension | Exam sprint path to finish current goal |
| `feature_value_recap` | credit | Show unused features and value |
| `billing_clarity_reset` | billing | Billing clarity and plan adjustment |

### Cancel Reasons

#### Zeo Auto Reasons

Detected from transcript via LLM extraction. Routing based on analysis of 85 real cancel events.

| Reason | Frequency | Default Action | Logic |
|---|---|---|---|
| `price` | 31% | `discount_50` | Escalates: 20% if open to staying, 50% if frustrated |
| `webhook` | 22% | `free_week` | Silent churn, re-engage with free trial |
| `user_initiated` | 22% | `pause_plan` | Generic cancel, low friction pause |
| `route_quality` | 8% | `route_fix_commitment` | Empathize, commit to fix, free week |
| `job_change` | 7% | `control_graceful_exit` | Life event, unsaveable |
| `low_usage` | — | `downgrade_basic` | Use less, pay less |
| `no_need` | 2% | `control_graceful_exit` | Graceful exit |
| `billing_confusion` | rare | `fleet_rightsize` | Clarify + adjust plan |
| `other` | — | `pause_plan` | Default soft pause |

#### Jungle AI Reasons

Routing derived from trained bandit context arm posteriors (300 decisions).

| Reason | Default Action | Win Rate | Logic |
|---|---|---|---|
| `price` | `pause_plan_relief` | 75.0% | Pause beats discounts for price |
| `graduating` | `downgrade_lite_switch` | 74.5% | Keep on lite, they may return |
| `break` | `pause_plan_relief` | 67.9% | Natural fit for seasonal breaks |
| `quality_bug` | `feature_value_recap` | 81.2% | Show value they missed + empathize |
| `feature_gap` | `targeted_discount_40` | 78.9% | Big discount to retain while features ship |
| `competition` | `targeted_discount_40` | 72.2% | Price war, match value |
| `billing_confusion` | `feature_value_recap` | 78.6% | Reframe value before fixing billing |
| `other` | `concierge_recovery` | 80.8% | Generic reason, human touch works best |

### Example Requests

**Zeo Auto — price complaint:**
```bash
curl -X POST https://semarang-liart.vercel.app/api/zeo_present_offers \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_456",
    "transcript": "Your pricing is way too high for what I get. I need to cancel.",
    "metadata": {"plan_tier": "monthly", "tenure_days": 90}
  }'
```

**Jungle AI — student graduating (bandit mode):**
```bash
curl -X POST https://semarang-liart.vercel.app/api/jungle_present_offers \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "student_789",
    "transcript": "I am graduating next month and wont need this anymore",
    "metadata": {"plan_tier": "super_learner", "tenure_days": 365}
  }'
```

**Zeo Auto — silent churn (empty transcript):**
```bash
curl -X POST https://semarang-liart.vercel.app/api/zeo_present_offers \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_silent",
    "transcript": "",
    "metadata": {"plan_tier": "weekly"}
  }'
```

**Jungle AI — reason routing mode (bandit off):**
```bash
curl -X POST https://semarang-liart.vercel.app/api/jungle_present_offers \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_competition",
    "transcript": "I am switching to ChatGPT, it does everything I need",
    "metadata": {"plan_tier": "free"},
    "use_bandit": "false"
  }'
```

### Error Responses

| Status | Body | Cause |
|---|---|---|
| `400` | `{"error": "Invalid JSON"}` | Malformed request body |
| `500` | Vercel error page | Server error (check Vercel function logs) |

### Environment Variables (Vercel)

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes (for personalization) | OpenAI API key for transcript extraction and LLM copy generation. Without it, transcript extraction falls back to heuristic and personalization is disabled. |
| `CLIENT_ID` | No | Override default client. Not typically needed since each endpoint has its own default. |
| `PERSONALIZE_MODEL` | No | Override the LLM model for copy generation. Default: `gpt-4o-mini`. |

---

## Architecture

```
11 Labs Voice Agent
        │
        ▼ POST /api/{client}_present_offers
┌─────────────────────────────┐
│   Vercel Serverless Function │
│                              │
│  1. Transcript Extraction    │  ← OpenAI LLM or heuristic fallback
│     (reason, frustration,    │
│      save_openness)          │
│                              │
│  2. Action Selection         │  ← Bandit (Thompson sampling) or
│     (which retention offer)  │    Reason-based routing
│                              │
│  3. UI Personalization       │  ← LLM writes copy constrained by
│     (header, body, CTA)      │    optimizer's strategy dimensions
│                              │
│  4. Fixed Fallback           │  ← Hardcoded UI if LLM unavailable
└─────────────────────────────┘
        │
        ▼ JSON response
  Retention Screen Component
```

### Three-Layer Decision Stack

1. **Action Selection** — The bandit (or reason router) picks WHICH retention action to show (discount, pause, downgrade, etc.)
2. **Strategy Dimensions** — The optimizer's strategy pool provides creative guardrails (message angle, proof style, personalization level, etc.)
3. **LLM Copy Generation** — GPT-4o-mini writes personalized UI text, constrained by the action + strategy dims

---

## Research Lab (offline)

### What is here

- `docs/2026-03-31-office-hours-jungle-ai-churn-cta.md` — the `/office-hours` design doc.
- `PLAN.md` — refined product and engineering plan after `/plan-ceo-review`, `/plan-eng-review`, and `/plan-design-review`.
- `DESIGN.md` — internal tool design system for a retention research console.
- `program.md` — an `autoresearch`-style research loop for this CTA optimization problem.
- `data/jungle_ai_seed_profiles.json` — sanitized seed profiles derived from the sample Amplitude-style user summaries.
- `src/cta_autoresearch/` — the simulator, synthetic persona generator, optimizer, and CLI.
- `tests/` — lightweight regression tests for feature derivation and strategy scoring.

### Quick start

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

### How this maps to the product

The lab optimizes five decision surfaces together:

1. Message angle
2. Proof style
3. Offer or incentive, including discount magnitude
4. Exit CTA
5. Personalization intensity

It does this against a synthetic cohort generated from seed user profiles and retention archetypes. The scoring model is deliberately interpretable so the team can inspect why a strategy wins before wiring it into real experiments.
