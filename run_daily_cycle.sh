#!/bin/bash
# Daily online→offline→deploy cycle.
# Schedule: crontab -e → 0 2 * * * /path/to/run_daily_cycle.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

STATE_DIR="${BANDIT_STATE_DIR:-serving_data/policy}"
EVAL_COHORT="${EVAL_COHORT:-data/jungle_ai_seed_profiles.json}"
ITERATIONS="${DAILY_ITERATIONS:-50}"
MODE="${DAILY_MODE:-random}"
LOG_DIR="serving_logs"
LOG="$LOG_DIR/daily_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR" "$STATE_DIR"

echo "=== Daily cycle started at $(date) ===" | tee "$LOG"

# Step 1: Backup current state
if [ -f "$STATE_DIR/policy_state.json" ]; then
    cp "$STATE_DIR/policy_state.json" "$STATE_DIR/policy_state_backup_$(date +%Y%m%d).json"
    echo "Backed up policy state." | tee -a "$LOG"
fi

# Step 2: Run optimizer with current live state
python3 -m cta_autoresearch.policy_optimizer \
    --data-dir "$STATE_DIR" \
    --output-dir optimizer_output \
    --load-state "$STATE_DIR/policy_state.json" \
    --eval-cohort "$EVAL_COHORT" \
    --iterations "$ITERATIONS" \
    --mode "$MODE" \
    --intensity 1.5 2>&1 | tee -a "$LOG"

# Step 3: Server hot-reloads via mtime detection — no restart needed
echo "=== Optimization complete. Server will hot-reload on next request. ===" | tee -a "$LOG"
echo "=== Daily cycle finished at $(date) ===" | tee -a "$LOG"
