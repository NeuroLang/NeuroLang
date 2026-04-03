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
