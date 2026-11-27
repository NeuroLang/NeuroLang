#!/bin/sh
# S4 of ../PROTOCOL.md: the four folds, one after the other (memory: one engine at a time).
# usage, from tools/biomedsql_benchmark: sh strategies/results/run_s4.sh BiomedSQL.csv DATA_DIR RESULTS_DIR
# RESULTS_DIR holds s4/folds.json and s4/guidance_f.txt (strategies/induce_guidance.py).
for f in 0 1 2 3; do
  T=$(python3 -c "import json;print(','.join(json.load(open('$3/s4/folds.json'))[$f]))")
  uv run --package neurolang-apps --extra llm python strategies/run.py "$1" "$2" S4_$f 1 "$3/S4_$f.jsonl" \
    --guidance "$3/s4/guidance_$f.txt" --templates "$T" > "$3/S4_$f.log" 2>&1
done
