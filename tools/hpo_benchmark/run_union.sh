#!/bin/sh
# The Sonnet arms of PROTOCOL_UNION.md, then the analysis; about $2 for 184
# patients. With HPO_LLM_ROUTE=openrouter, through OpenRouter
# (OPENROUTER_API_KEY); otherwise directly (ANTHROPIC_API_INRIA_KEY).
set -e
cd "$(dirname "$0")"
if [ "$HPO_LLM_ROUTE" = openrouter ]; then
  : "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first}"
else
  : "${ANTHROPIC_API_INRIA_KEY:?export ANTHROPIC_API_INRIA_KEY first}"
fi
uv run --package neurolang-apps --extra llm python union_hybrid.py run
