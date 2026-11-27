#!/bin/sh
# The test run of PROTOCOL.md: conditions A, B, C, three repeats each (one process per condition).
# usage: sh run_test.sh BiomedSQL.csv DATA_DIR RESULTS_DIR
for c in A B C; do
  (for run in 1 2 3; do
     uv run --package neurolang-apps --extra llm python run_agent.py "$1" "$2" $c test $run "$3/test_${c}_${run}.jsonl" > "$3/test_${c}_${run}.log" 2>&1
   done) &
done
wait
