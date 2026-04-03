# User Testing

## Validation Surface

**Primary surface:** Browser (React web application)
- Dev: http://localhost:3100 (Vite dev server, proxies API to :8888)
- Prod build: http://localhost:8888/sparklis/ (served by Tornado)

**API surface:** HTTP/WebSocket endpoints on http://localhost:8888/v2/

**Tool:** `agent-browser` for all browser-based assertions.

## Validation Concurrency

Machine: ~36GB RAM, 11 CPU cores.

### agent-browser surface
- The React dev server uses ~200MB. Each agent-browser instance ~300MB.
- Baseline memory usage: ~6GB.
- Usable headroom: (36 - 6) * 0.7 = ~21GB.
- 5 instances = ~1.7GB (well within budget).
- **Max concurrent: 5**

## Testing Setup Requirements

### Services to start
1. Tornado backend on port 8888 (with Neurosynth + Destrieux engines)
2. React dev server on port 3100 (or use prod build served by Tornado)

### Engine readiness
- Engines take 30-60 seconds to initialize on first start
- Healthcheck: `curl -sf http://localhost:8888/v2/engines` returns non-empty JSON array
- Schema check: `curl -sf http://localhost:8888/v2/schema/neurosynth` returns symbols

### Data requirements
- `~/neurolang_data/` must contain downloaded datasets (auto-downloaded on first engine start)
- No external credentials or accounts needed

### Known constraints
- Engine initialization is slow; validation should wait for healthcheck before testing
- Neurosynth queries with spatial smoothing can take 10+ seconds
- The `compute_datalog_program_for_autocompletion` API requires a valid program prefix; empty strings cause parser errors
- The v2_handlers.py schema endpoint iterates `engine.symbols` (which is a QuerySymbolsProxy) using `for name in engine.symbols:` and then `engine.symbols[name]` to get each symbol. Do NOT call `.items()` on it.

## Flow Validator Guidance: agent-browser

This section covers isolation rules for browser-based validation subagents.

### Isolation boundaries
- Each subagent uses a separate browser session (use `--session <session-id>__<suffix>` format)
- The React dev server at http://localhost:3100 is shared; subagents should not navigate away from the app
- No shared mutable state exists in the app at this stage (no user accounts, no persistent server-side state per user)
- Subagents may run concurrently without interference

### Resources
- App URL: http://localhost:3100
- API base: http://localhost:8888 (proxied as /api/ from the dev server if configured, otherwise direct)
- Chrome is at `/Applications/Google Chrome.app`

### Test setup
- Ensure Tornado backend is running: `curl -sf http://localhost:8888/v2/engines`
- Ensure React dev server is running: `curl -sf http://localhost:3100`
- No login or credentials required

### Screenshot evidence
- Save screenshots to the evidence directory provided in the subagent prompt
- Use descriptive filenames matching the assertion ID (e.g., `VAL-FOUND-001-app-loaded.png`)

### Known UI behaviors (foundation milestone)
- Engine list should appear in the sidebar showing "neurosynth" and "destrieux"
- Clicking an engine in the sidebar should highlight it and update the main content area
- The layout should have a top nav bar, sidebar, and main content area
