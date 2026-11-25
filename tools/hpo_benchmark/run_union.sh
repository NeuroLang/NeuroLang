#!/bin/sh
# The Sonnet arms of PROTOCOL_UNION.md, then the analysis. Needs
# ANTHROPIC_API_INRIA_KEY in the environment; about $2 for 184 patients.
set -e
cd "$(dirname "$0")"
: "${ANTHROPIC_API_INRIA_KEY:?export ANTHROPIC_API_INRIA_KEY first}"
uv run --package neurolang-apps --extra llm python union_hybrid.py run
