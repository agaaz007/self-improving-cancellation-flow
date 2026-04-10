# CTA Autoresearch Lab

This project turns your churn-reduction idea into a small, runnable research lab — and now serves trained retention policies via serverless API endpoints with live outcome tracking and a monitoring dashboard.

It uses the core `autoresearch` pattern from [karpathy/autoresearch](https://github.com/karpathy/autoresearch): keep the evaluation harness mostly fixed, expose one high-leverage strategy file, and let an agent iterate on messaging strategy against a clear metric.

---

## Live API — Production Deployment

**Base URL:** `https://semarang-liart.vercel.app`

**Dashboard:** `https://semarang-liart.vercel.app/monitor`

### Endpoints Overview

| Endpoint | Method | Description |
|---|---|---|
| `/api/zeo_present_offers` | POST | Zeo Auto retention offers (route optimization SaaS) |
| `/api/jungle_present_offers` | POST | Jungle AI retention offers (edtech) |
| `/api/record_outcome` | POST | Record retention outcome + update bandit posteriors |
| `/api/dashboard_data` | GET | Dashboard metrics API (all views) |
| `/monitor` | GET | Monitoring dashboard UI |

---

## POST /api/zeo_present_offers

Returns a personalized retention offer for Zeo Auto (route optimization SaaS). Default mode: reason-based routing from analysis of 85 real cancel events.

### Request

```json
{
  "user_id": "user_123",
  "transcript": "I want to cancel because it's too expensive for me right now",
  "metadata": {
    "plan_tier": "monthly",
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
| `user_id` | string | No | `"anonymous"` | Unique user identifier |
| `transcript` | string | No | `""` | Conversation transcript. Empty = silent/webhook churn flow |
| `metadata` | object | No | `{}` | User context. See metadata fields below |
| `personalize` | bool | No | `true` | Enable LLM-generated copy (gpt-4o-mini). Falls back to fixed text if disabled or LLM unavailable |
| `use_bandit` | bool | No | `false` | `true` = Thompson sampling. `false` = reason-based routing |

#### Metadata Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `plan_tier` | string | `"unknown"` | `weekly`, `monthly`, `quarterly`, `annual` |
| `tenure_days` | int | `0` | Days since first subscribed |
| `engagement_7d` | float | `0.0` | 7-day engagement score (0.0-1.0) |
| `engagement_30d` | float | `0.0` | 30-day engagement score (0.0-1.0) |
| `session_id` | string | auto | Optional session identifier |
| `prior_cancel_attempts_30d` | int | `0` | Prior cancel attempts in last 30 days |
| `discount_exposures_30d` | int | `0` | Discount exposures in last 30 days |

### Response

```json
{
  "decision_id": "dec_844d2226718d",
  "action_id": "discount_50",
  "ui": {
    "header_title": "We'd like to keep you",
    "body_text": "Here's our best offer — 50% off your next month.",
    "cta_button_text": "Claim 50% off",
    "cta_button_action": "discount_50",
    "offer_kind": "discount",
    "secondary_action_text": "No thanks, continue canceling",
    "secondary_action": "dismiss"
  },
  "meta": {
    "policy_version": "reason_routing_v1",
    "exploration_flag": false,
    "primary_reason": "price",
    "frustration_level": 0.72,
    "save_openness": 0.18,
    "client_id": "zeo_auto",
    "personalized": true,
    "routing_mode": "reason",
    "strategy_dims": null
  }
}
```

### Zeo Auto Actions

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

### Zeo Reason Routing

| Reason | Frequency | Default Action | Logic |
|---|---|---|---|
| `price` | 31% | `discount_50` | Escalates: 20% if open, 50% if frustrated |
| `webhook` | 22% | `free_week` | Silent churn, re-engage with free trial |
| `user_initiated` | 22% | `pause_plan` | Generic cancel, low friction pause |
| `route_quality` | 8% | `route_fix_commitment` | Empathize, commit to fix, free week |
| `job_change` | 7% | `control_graceful_exit` | Life event, unsaveable |
| `low_usage` | — | `downgrade_basic` | Use less, pay less |
| `no_need` | 2% | `control_graceful_exit` | Graceful exit |
| `billing_confusion` | rare | `fleet_rightsize` | Clarify + adjust plan |
| `other` | — | `pause_plan` | Default soft pause |

---

## POST /api/jungle_present_offers

Returns a personalized retention offer for Jungle AI (edtech). Default mode: Thompson sampling bandit (trained on 300 decisions, 54 strategy generations).

### Request

Same schema as Zeo. Key differences:

| Field | Default | Notes |
|---|---|---|
| `use_bandit` | `true` | Bandit is default (trained policy is mature) |
| `metadata.plan_tier` | `"unknown"` | Values: `free`, `starter`, `super_learner` |

### Jungle AI Actions

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

### Jungle Reason Routing (fallback)

Derived from trained bandit context arm posteriors (300 decisions).

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

---

## POST /api/record_outcome

Records a retention outcome and updates the bandit's posteriors in Redis. Call this after you know whether the customer was saved or churned.

### Request

```json
{
  "client_id": "zeo_auto",
  "decision_id": "dec_844d2226718d",
  "saved_flag": true,
  "cancel_completed_flag": false,
  "support_escalation_flag": false,
  "complaint_flag": false
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `client_id` | string | Yes | — | `"zeo_auto"` or `"jungle_ai"` |
| `decision_id` | string | Yes | — | The `decision_id` from the present_offers response |
| `saved_flag` | bool | No | `false` | Customer was retained |
| `cancel_completed_flag` | bool | No | `false` | Customer completed cancellation |
| `support_escalation_flag` | bool | No | `false` | Customer escalated to support (-0.3 reward penalty) |
| `complaint_flag` | bool | No | `false` | Customer complained (-0.2 reward penalty) |

### Response

```json
{
  "status": "recorded",
  "reward": 1.0,
  "action_id": "discount_50"
}
```

### Reward Computation

```
reward = saved_flag(1.0) - escalation(0.3) - complaint(0.2)
```

| Scenario | Reward |
|---|---|
| Saved, no issues | 1.0 |
| Saved but escalated | 0.7 |
| Saved but complained | 0.8 |
| Churned, no issues | 0.0 |
| Churned + escalated + complained | -0.5 |

### Error Responses

| Status | Body | Cause |
|---|---|---|
| `400` | `{"error": "decision_id required"}` | Missing decision_id |
| `404` | `{"error": "Unknown decision_id: ..."}` | decision_id not found in Redis |
| `500` | `{"error": "No policy state in Redis"}` | Redis has no policy for this client |
| `503` | `{"error": "Redis not configured"}` | Upstash Redis env vars missing |

---

## GET /api/dashboard_data

Serves dashboard metrics from Redis. Used by the monitoring UI.

### Query Parameters

| Param | Default | Values |
|---|---|---|
| `client` | `zeo_auto` | `zeo_auto`, `jungle_ai` |
| `view` | `health` | See views below |

### Views

| View | Description |
|---|---|
| `health` | Policy status, decision/outcome counts, exploration rate |
| `arms` | Global arm posteriors with mean, CI, impressions, win rate |
| `context-arms` | Context-specific arm posteriors (reason\|plan_tier\|action) |
| `replay` | Outcome aggregation by action and reason (save rates, avg reward) |
| `strategies` | Strategy pool with dims and scores per action |
| `generation` | Generation metadata and config |
| `recent-decisions` | Last 30 decisions logged |
| `recent-outcomes` | Last 30 outcomes recorded |

### Example

```bash
curl "https://semarang-liart.vercel.app/api/dashboard_data?client=jungle_ai&view=arms"
```

---

## Response Fields Reference

### UI Object (in present_offers response)

| Field | Type | Description |
|---|---|---|
| `header_title` | string | Card headline (3-8 words) |
| `body_text` | string | Card body copy (1-2 sentences) |
| `cta_button_text` | string | Primary CTA button label |
| `cta_button_action` | string | Action ID the CTA maps to |
| `offer_kind` | string | `discount`, `pause`, `downgrade`, `extension`, `support`, `credit`, `billing`, `none` |
| `secondary_action_text` | string | Always "No thanks, continue canceling" |
| `secondary_action` | string | Always `"dismiss"` |

### Meta Object (in present_offers response)

| Field | Type | Description |
|---|---|---|
| `policy_version` | string | `"v1"` (bandit) or `"reason_routing_v1"` |
| `exploration_flag` | bool | `true` if bandit is exploring |
| `primary_reason` | string | Detected cancel reason from transcript |
| `frustration_level` | float | 0.0-1.0 frustration score |
| `save_openness` | float | 0.0-1.0 openness to being retained |
| `client_id` | string | `"zeo_auto"` or `"jungle_ai"` |
| `personalized` | bool | `true` if LLM-generated copy |
| `routing_mode` | string | `"bandit"` or `"reason"` |
| `strategy_dims` | object/null | Strategy dimensions used for LLM personalization |

---

## 11Labs Voice Agent Integration

### Webhook Tool Configuration (Zeo Auto)

Use this JSON to add the Zeo Auto endpoint as a webhook tool in your 11Labs Conversational AI agent:

```json
{
  "type": "webhook",
  "name": "present_offers",
  "description": "Call the Zeo Auto retention API to get personalized cancellation offers. Call this once you understand the customer's cancellation reason. Sends the conversation transcript and returns an action, strategy, and copy to present to the customer.",
  "api_schema": {
    "url": "https://semarang-liart.vercel.app/api/zeo_present_offers",
    "method": "POST",
    "path_params_schema": [],
    "query_params_schema": [],
    "request_body_schema": {
      "id": "request_body",
      "description": "Request body for the Zeo Auto retention API",
      "required": true,
      "type": "object",
      "properties": [
        {
          "id": "transcript",
          "type": "string",
          "value_type": "llm_prompt",
          "description": "The full conversation transcript so far. Include everything the customer has said about why they want to cancel.",
          "dynamic_variable": "",
          "constant_value": "",
          "enum": null,
          "is_system_provided": false,
          "required": true
        },
        {
          "id": "plan_tier",
          "type": "string",
          "value_type": "llm_prompt",
          "description": "The customer's current subscription plan. One of: weekly, monthly, quarterly, annual. Default to monthly if unknown.",
          "dynamic_variable": "",
          "constant_value": "",
          "enum": ["weekly", "monthly", "quarterly", "annual"],
          "is_system_provided": false,
          "required": false
        },
        {
          "id": "tenure_months",
          "type": "number",
          "value_type": "llm_prompt",
          "description": "How many months the customer has been subscribed. Default to 6 if unknown.",
          "dynamic_variable": "",
          "constant_value": "",
          "enum": null,
          "is_system_provided": false,
          "required": false
        }
      ]
    },
    "request_headers": [],
    "content_type": "application/json",
    "auth_connection": null
  },
  "response_timeout_secs": 20,
  "dynamic_variables": {
    "dynamic_variable_placeholders": {}
  },
  "assignments": [],
  "disable_interruptions": false,
  "force_pre_tool_speech": "auto",
  "tool_call_sound": null,
  "tool_call_sound_behavior": "auto",
  "execution_mode": "immediate",
  "tool_error_handling_mode": "auto",
  "response_mocks": []
}
```

### Webhook Tool Configuration (Jungle AI)

Same structure, swap URL and plan tiers:

```json
{
  "type": "webhook",
  "name": "present_offers",
  "description": "Call the Jungle AI retention API to get personalized cancellation offers. Call this once you understand the student's cancellation reason. Sends the conversation transcript and returns an action, strategy, and copy to present.",
  "api_schema": {
    "url": "https://semarang-liart.vercel.app/api/jungle_present_offers",
    "method": "POST",
    "path_params_schema": [],
    "query_params_schema": [],
    "request_body_schema": {
      "id": "request_body",
      "description": "Request body for the Jungle AI retention API",
      "required": true,
      "type": "object",
      "properties": [
        {
          "id": "transcript",
          "type": "string",
          "value_type": "llm_prompt",
          "description": "The full conversation transcript so far. Include everything the student has said about why they want to cancel.",
          "dynamic_variable": "",
          "constant_value": "",
          "enum": null,
          "is_system_provided": false,
          "required": true
        },
        {
          "id": "plan_tier",
          "type": "string",
          "value_type": "llm_prompt",
          "description": "The student's current subscription plan. One of: free, starter, super_learner. Default to starter if unknown.",
          "dynamic_variable": "",
          "constant_value": "",
          "enum": ["free", "starter", "super_learner"],
          "is_system_provided": false,
          "required": false
        },
        {
          "id": "tenure_months",
          "type": "number",
          "value_type": "llm_prompt",
          "description": "How many months the student has been subscribed. Default to 6 if unknown.",
          "dynamic_variable": "",
          "constant_value": "",
          "enum": null,
          "is_system_provided": false,
          "required": false
        }
      ]
    },
    "request_headers": [],
    "content_type": "application/json",
    "auth_connection": null
  },
  "response_timeout_secs": 20,
  "dynamic_variables": {
    "dynamic_variable_placeholders": {}
  },
  "assignments": [],
  "disable_interruptions": false,
  "force_pre_tool_speech": "auto",
  "tool_call_sound": null,
  "tool_call_sound_behavior": "auto",
  "execution_mode": "immediate",
  "tool_error_handling_mode": "auto",
  "response_mocks": []
}
```

### 11Labs Agent System Prompt (recommended)

Add this to the agent's system prompt to guide tool usage:

```
You have access to a retention tool called present_offers. When the customer tells you
why they want to cancel, call present_offers with the full conversation transcript.
The tool returns a personalized retention offer.

After calling the tool:
1. Read the ui.header_title and ui.body_text to understand what offer was selected
2. Present the offer conversationally — don't read the JSON, paraphrase it naturally
3. Use the ui.cta_button_text as your call to action
4. If the customer declines, acknowledge gracefully and let them continue canceling

Never ask the customer for their plan tier or tenure — estimate from context or omit.
```

---

## Example curl Commands

**Zeo Auto — price complaint:**
```bash
curl -X POST https://semarang-liart.vercel.app/api/zeo_present_offers \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Your pricing is way too high for what I get. I need to cancel.",
    "metadata": {"plan_tier": "monthly", "tenure_days": 90}
  }'
```

**Jungle AI — student graduating (bandit mode):**
```bash
curl -X POST https://semarang-liart.vercel.app/api/jungle_present_offers \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "I am graduating next month and wont need this anymore",
    "metadata": {"plan_tier": "super_learner", "tenure_days": 365}
  }'
```

**Record outcome:**
```bash
curl -X POST https://semarang-liart.vercel.app/api/record_outcome \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "zeo_auto",
    "decision_id": "dec_844d2226718d",
    "saved_flag": true,
    "support_escalation_flag": false,
    "complaint_flag": false
  }'
```

**Dashboard data:**
```bash
curl "https://semarang-liart.vercel.app/api/dashboard_data?client=jungle_ai&view=health"
```

---

## Architecture

```
11 Labs Voice Agent
        |
        v POST /api/{client}_present_offers
+-------------------------------+
|   Vercel Serverless Function  |
|                               |
|  1. Transcript Extraction     |  <- OpenAI LLM or heuristic fallback
|     (reason, frustration,     |
|      save_openness)           |
|                               |
|  2. Action Selection          |  <- Bandit (Thompson sampling) or
|     (which retention offer)   |     Reason-based routing
|                               |
|  3. UI Personalization        |  <- LLM writes copy constrained by
|     (header, body, CTA)       |     optimizer's strategy dimensions
|                               |
|  4. Fixed Fallback            |  <- Hardcoded UI if LLM unavailable
+-------------------------------+
        |
        v JSON response
  Voice Agent presents offer
        |
        v Customer responds
  POST /api/record_outcome       <- Closes feedback loop
        |                           Updates bandit posteriors in Redis
        v
  Upstash Redis (shared state)  <- All serverless instances share posteriors
        |
        v
  /monitor dashboard             <- Real-time monitoring UI
```

### Three-Layer Decision Stack

1. **Action Selection** — The bandit (or reason router) picks WHICH retention action to show (discount, pause, downgrade, etc.)
2. **Strategy Dimensions** — The optimizer's strategy pool provides creative guardrails (message angle, proof style, personalization level, etc.)
3. **LLM Copy Generation** — GPT-4o-mini writes personalized UI text, constrained by the action + strategy dims

### Live Posterior Updates

When outcomes are recorded via `/api/record_outcome`:
- Reward is computed: `saved(1.0) - escalation(0.3) - complaint(0.2)`
- Global arm posteriors updated: `alpha += success + max(reward, 0)`, `beta += (1-success) + max(-reward, 0)`
- Context arm posteriors updated (keyed by `reason|plan_tier|action`)
- Updated state saved to Redis — next request from any serverless instance sees the new posteriors

---

## Environment Variables (Vercel)

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes (for personalization) | OpenAI API key for transcript extraction and LLM copy generation |
| `UPSTASH_REDIS_REST_URL` | Yes (for outcome loop) | Upstash Redis REST URL. Without it, endpoints fall back to bundled JSON with no outcome tracking |
| `UPSTASH_REDIS_REST_TOKEN` | Yes (for outcome loop) | Upstash Redis REST token |
| `CLIENT_ID` | No | Override default client for Zeo endpoint. Not typically needed |
| `PERSONALIZE_MODEL` | No | Override LLM model for copy generation. Default: `gpt-4o-mini` |

---

## Error Responses

| Status | Body | Cause |
|---|---|---|
| `400` | `{"error": "Invalid JSON"}` | Malformed request body |
| `400` | `{"error": "decision_id required"}` | Missing decision_id on record_outcome |
| `404` | `{"error": "Unknown decision_id: ..."}` | decision_id not found (record_outcome) |
| `503` | `{"error": "Redis not configured"}` | Redis env vars missing (record_outcome, dashboard_data) |
| `500` | Vercel error page | Server error (check Vercel function logs) |

---

## Known Limitations

1. **No server-side dedup on outcomes** — recording the same decision_id twice will update posteriors again. Client must ensure idempotency.
2. **Decision count in health view** — reflects training count (300/200), not live production count. Live count is in the `dlog` list.
3. **Cold start latency** — first request after idle may take 3-5s (Vercel cold start + Redis policy load).

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

### How this maps to the product

The lab optimizes five decision surfaces together:

1. Message angle
2. Proof style
3. Offer or incentive, including discount magnitude
4. Exit CTA
5. Personalization intensity

It does this against a synthetic cohort generated from seed user profiles and retention archetypes. The scoring model is deliberately interpretable so the team can inspect why a strategy wins before wiring it into real experiments.
