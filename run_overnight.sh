#!/bin/bash
# Overnight autoresearch pipeline — leave running, check results in the morning
# Usage: nohup bash run_overnight.sh &
#
# FIX: Phases now compound — Phase 2+ uses --load-state to build on previous
# phase's output instead of re-warm-starting from scratch each time.

set -e

cd "$(dirname "$0")"
export PYTHONPATH=src

LOG="optimizer_output/overnight.log"
mkdir -p optimizer_output

echo "=== Overnight autoresearch started at $(date) ===" | tee "$LOG"

# Phase 1: Clean slate, high exploration, 200 iterations
echo "" | tee -a "$LOG"
echo "=== Phase 1: Exploration (200 iters, intensity=3.0) ===" | tee -a "$LOG"
rm -rf optimizer_data optimizer_output/results.tsv

python3 -m cta_autoresearch.policy_optimizer \
  --mode agent \
  --model gpt-5.4-2026-03-05 \
  --iterations 200 \
  --bootstrap-traffic 300 \
  --bootstrap-save-probability 0.45 \
  --intensity 3.0 \
  --seed 42 \
  --warm-start-file .context/warm_start_rows_enriched_457.json \
  --eval-cohort .context/warm_start_rows_enriched_457.json \
  --data-dir optimizer_data \
  --output-dir optimizer_output 2>&1 | tee -a "$LOG"

# Save Phase 1 snapshot
cp optimizer_output/results.tsv optimizer_output/results_phase1.tsv 2>/dev/null || true
cp optimizer_data/policy/policy_state.json optimizer_output/policy_state_phase1.json

# Phase 2: CONTINUE from Phase 1 state, lower intensity for fine-tuning
# Key fix: --load-state loads Phase 1's optimized priors instead of re-warm-starting
echo "" | tee -a "$LOG"
echo "=== Phase 2: Fine-tuning (200 iters, intensity=1.5) ===" | tee -a "$LOG"

python3 -m cta_autoresearch.policy_optimizer \
  --mode agent \
  --model gpt-5.4-2026-03-05 \
  --iterations 200 \
  --bootstrap-traffic 0 \
  --bootstrap-save-probability 0.45 \
  --intensity 1.5 \
  --seed 777 \
  --load-state optimizer_output/policy_state_phase1.json \
  --eval-cohort .context/warm_start_rows_enriched_457.json \
  --data-dir optimizer_data \
  --output-dir optimizer_output 2>&1 | tee -a "$LOG"

# Save Phase 2 snapshot
cp optimizer_output/results.tsv optimizer_output/results_phase2.tsv 2>/dev/null || true
cp optimizer_data/policy/policy_state.json optimizer_output/policy_state_phase2.json

# Phase 3: Polish — tight mutations, continue from Phase 2
echo "" | tee -a "$LOG"
echo "=== Phase 3: Polish (100 iters, intensity=1.0) ===" | tee -a "$LOG"

python3 -m cta_autoresearch.policy_optimizer \
  --mode agent \
  --model gpt-5.4-2026-03-05 \
  --iterations 100 \
  --bootstrap-traffic 0 \
  --bootstrap-save-probability 0.45 \
  --intensity 1.0 \
  --seed 2024 \
  --load-state optimizer_output/policy_state_phase2.json \
  --eval-cohort .context/warm_start_rows_enriched_457.json \
  --data-dir optimizer_data \
  --output-dir optimizer_output 2>&1 | tee -a "$LOG"

# Save final state
cp optimizer_data/policy/policy_state.json optimizer_output/policy_state_final.json

echo "" | tee -a "$LOG"
echo "=== Overnight autoresearch completed at $(date) ===" | tee -a "$LOG"
echo "Final policy: optimizer_output/policy_state_final.json" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
