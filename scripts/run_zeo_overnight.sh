#!/bin/bash
# Overnight autoresearch pipeline for Zeo Auto — leave running, check results in the morning
# Usage: OPENAI_API_KEY=<key> nohup bash scripts/run_zeo_overnight.sh &
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export CLIENT_ID=zeo_auto

DATA_DIR="serving_data/zeo_policy"
OUTPUT_DIR="optimizer_output/zeo_auto"
EVAL_COHORT="data/zeo_eval_cohort.json"
LOG="$OUTPUT_DIR/overnight.log"

mkdir -p "$OUTPUT_DIR"
echo "=== Zeo overnight autoresearch started at $(date) ===" | tee "$LOG"

# Phase 1: Clean slate, high exploration, 200 iterations
echo "" | tee -a "$LOG"
echo "=== Phase 1: Exploration (200 iters, intensity=3.0) ===" | tee -a "$LOG"
rm -rf "$DATA_DIR" "$OUTPUT_DIR/results.tsv"
python -m cta_autoresearch.policy_optimizer \
  --mode agent \
  --model gpt-4o \
  --iterations 200 \
  --bootstrap-traffic 300 \
  --bootstrap-save-probability 0.45 \
  --intensity 3.0 \
  --seed 42 \
  --eval-cohort "$EVAL_COHORT" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG"

cp "$OUTPUT_DIR/results.tsv" "$OUTPUT_DIR/results_phase1.tsv" 2>/dev/null || true
cp "$DATA_DIR/policy/policy_state.json" "$OUTPUT_DIR/policy_state_phase1.json"

# Phase 2: Continue from Phase 1 state, lower intensity for fine-tuning
echo "" | tee -a "$LOG"
echo "=== Phase 2: Fine-tuning (200 iters, intensity=1.5) ===" | tee -a "$LOG"
python -m cta_autoresearch.policy_optimizer \
  --mode agent \
  --model gpt-4o \
  --iterations 200 \
  --bootstrap-traffic 0 \
  --bootstrap-save-probability 0.45 \
  --intensity 1.5 \
  --seed 777 \
  --load-state "$OUTPUT_DIR/policy_state_phase1.json" \
  --eval-cohort "$EVAL_COHORT" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG"

cp "$OUTPUT_DIR/results.tsv" "$OUTPUT_DIR/results_phase2.tsv" 2>/dev/null || true
cp "$DATA_DIR/policy/policy_state.json" "$OUTPUT_DIR/policy_state_phase2.json"

# Phase 3: Polish — tight mutations, continue from Phase 2
echo "" | tee -a "$LOG"
echo "=== Phase 3: Polish (100 iters, intensity=1.0) ===" | tee -a "$LOG"
python -m cta_autoresearch.policy_optimizer \
  --mode agent \
  --model gpt-4o \
  --iterations 100 \
  --bootstrap-traffic 0 \
  --bootstrap-save-probability 0.45 \
  --intensity 1.0 \
  --seed 2024 \
  --load-state "$OUTPUT_DIR/policy_state_phase2.json" \
  --eval-cohort "$EVAL_COHORT" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG"

cp "$DATA_DIR/policy/policy_state.json" "$OUTPUT_DIR/policy_state_final.json"

echo "" | tee -a "$LOG"
echo "=== Zeo overnight autoresearch completed at $(date) ===" | tee -a "$LOG"
echo "Final policy: $OUTPUT_DIR/policy_state_final.json" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
