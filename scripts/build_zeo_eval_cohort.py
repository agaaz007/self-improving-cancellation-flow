"""Build Zeo Auto eval cohort from the enriched CSV.

Reads the Zeo AMP analysis CSV, filters to cancel events, and outputs
a JSON eval cohort in the format the policy optimizer expects.

Usage:
    python scripts/build_zeo_eval_cohort.py [input_csv] [output_json]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/zeo_amp_analysis_output.csv")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/zeo_eval_cohort.json")

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    # Import Zeo client to reuse reason_from_raw
    from cta_autoresearch.clients.zeo_auto import reason_from_raw

    with open(input_path) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Filter to rows with real cancel events (skip N/A, developer cancels)
    skip_reasons = {"", "nan", "n/a", "subscription was canceled by the developer"}
    cancel_rows = []
    for row in all_rows:
        cancel_reason = str(row.get("cancel_reason", "")).strip()
        if cancel_reason.lower() in skip_reasons:
            continue
        cancel_rows.append(row)

    print(f"Total rows: {len(all_rows)}, Cancel events: {len(cancel_rows)}")

    # Build eval cohort rows
    cohort_rows = []
    for row in cancel_rows:
        raw_reason = str(row.get("cancel_reason", ""))
        raw_note = str(row.get("cancel_note", ""))
        primary_reason = reason_from_raw(raw_reason, raw_note)

        # Map payment_interval to plan_tier
        interval = str(row.get("payment_interval", "month")).lower().strip()
        plan_map = {"week": "weekly", "month": "monthly", "quarter": "quarterly", "year": "annual"}
        plan_tier = plan_map.get(interval, "monthly")

        # Derive basic features from available data
        num_sessions = _int(row.get("num_sessions", 0))
        fleet_size = _int(row.get("fleet_size", 1))
        routes_created = _int(row.get("total_routes_created", 0))

        # Heuristic frustration/openness based on reason
        frustration = 0.3
        save_openness = 0.4
        trust_risk = 0.2
        if primary_reason == "route_quality":
            frustration = 0.7
            trust_risk = 0.4
        elif primary_reason == "price":
            save_openness = 0.6
        elif primary_reason == "low_usage":
            save_openness = 0.5
        elif primary_reason in ("webhook", "user_initiated"):
            frustration = 0.2
            save_openness = 0.3
        elif primary_reason == "job_change":
            save_openness = 0.1

        churn_risk = 0.5 + frustration * 0.3 - save_openness * 0.2

        cohort_rows.append({
            "primary_reason": primary_reason,
            "plan_tier": plan_tier,
            "student_type": "logistics",
            "features": {
                "frustration_level": round(frustration, 2),
                "trust_risk": round(trust_risk, 2),
                "save_openness": round(save_openness, 2),
                "churn_risk_score": round(churn_risk, 2),
                "feature_requests": [],
                "bug_signals": [],
                "tags": [f"fleet_{fleet_size}", f"sessions_{num_sessions}"],
            },
            # Pass through raw data for row_to_persona
            "num_sessions": num_sessions,
            "fleet_size": fleet_size,
            "total_routes_created": routes_created,
            "total_routes_optimized": _int(row.get("total_routes_optimized", 0)),
            "total_stops_planned": _int(row.get("total_stops_planned", 0)),
            "first_used": str(row.get("first_used", "")),
            "last_used": str(row.get("last_used", "")),
            "payment_interval": interval,
            "revenue": _float(row.get("revenue", 0)),
            "cancel_reason": raw_reason,
            "cancel_note": raw_note,
        })

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": cohort_rows, "client_id": "zeo_auto", "total": len(cohort_rows)}
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(cohort_rows)} rows to {output_path}")

    # Print reason distribution
    from collections import Counter
    reasons = Counter(r["primary_reason"] for r in cohort_rows)
    print("\nReason distribution:")
    for reason, count in reasons.most_common():
        print(f"  {reason:20s} {count:3d}")


def _int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
