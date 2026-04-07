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

### Known UI behaviors (execution milestone)
- CodeMirror 6 editor has `data-language='datalog'` and shows syntax highlighting (teal for identifiers/predicates, blue for operators like `:-`, gray for punctuation)
- Visual builder and code editor are bidirectionally synced; clicking a predicate in the browser updates both simultaneously
- The Run Query button shows `◌ Running...` with a red `✗ Cancel` button visible during query execution
- Destrieux queries execute very fast (<100ms) — to capture loading state, use JavaScript WebSocket interception with artificial delay
- After a query with no matching rows completes, `No results found` message appears in element with class `no-results-message`
- Example queries panel is in the sidebar showing examples from `/v2/examples/:engine`; clicking an example title loads it into the code editor
- Keyboard shortcut `Cmd+Enter` (Mac) / `Ctrl+Enter` submits the query when the code editor is focused
- **FIXED (execution milestone round 2)**: Cancel query now correctly shows "⊘ Query cancelled." instead of "WebSocketError". The `isCancellingRef` guard was added to `ExecutionContext.tsx` in commit f1fe82ab.
- **FIXED (execution milestone round 3)**: `responses.py serializeVBR` KeyError was fixed by changing `image_row[1]` to `image_row.iloc[1]` (commit c76b94e9). Results table now renders correctly for VBR-column queries.
- **FIXED (fix-pagination-controls)**: Pagination controls now correctly appear for result sets with >50 rows. Root cause: `QuerySocketHandler.send_query_update()` in `app.py` was calling `QueryResults(..., get_values=True)` without a `length` parameter, defaulting to 50 rows. Fixed by passing `length=10000` to send all rows to the frontend, which does client-side pagination. Verified with real Destrieux backend data (74 LeftSulcus rows → "Page 1 of 2"). Affects VAL-EXEC-008.
- **NOTE**: The "Sulcal identification" Destrieux example query (`~exists()` syntax) fails to parse with `UnexpectedTokenError on AMPERSAND`. Use a simpler custom query instead: `LeftSulcus(name_) :- destrieux(name_, region) & startswith("L S", name_)`
- **NOTE**: Symbol selector is a native HTML `<select>` element. In agent-browser, trigger change via JS: `select.value = 'symbol'; select.dispatchEvent(new Event('change', { bubbles: true }))`
- **CodeMirror editor content**: Cannot use standard `Control+A` + `fill` pattern to replace editor content. Use either React context `setDatalogText()` or click example query buttons to set query content.

### Known UI behaviors (query-builder milestone)
- After selecting an engine, the predicate browser shows Relations, Functions, and Probabilistic groups
- Search/filter in the predicate browser is client-side (no network call)
- Clicking a predicate from the browser adds it with auto-generated variable names (v, v1, v2, ...)
- The visual query renders as "Find [vars] where [predicates]" (readable NL form)
- Undo/Redo buttons are in the visual query builder toolbar
- Suggestions panel shows context-aware chips after predicates are added (debounced, ~1-2s delay after predicate add)
- Clicking a suggestion chip adds the predicate WITH proper placeholder variable names
- Variable binding: clicking a variable token opens inline rename textbox; renaming to match another variable applies 'vqb-var-token--shared' CSS class with blue highlight on all occurrences
- **Suggestion chips accessibility note**: suggestion chips (class `suggestions-chip`) do not appear in `agent-browser snapshot -i` accessibility tree; use JavaScript eval to click them: `document.querySelector('button.suggestions-chip')?.click()`

### IPv6 note (Vite dev server)
- The Vite dev server on port 3100 may only listen on IPv6 (::1) — use `http://[::1]:3100` if `http://localhost:3100` fails in agent-browser
- To start the dev server: `cd /Users/dwasserm/sources/NeuroLang/neurolang/utils/server/neurolang-sparklis && PORT=3100 npm run dev`

### Vite dev server WebSocket proxy issue with large payloads
- The Vite dev server's WebSocket proxy causes Playwright (agent-browser) to navigate to `about:blank` when the backend sends large WebSocket responses (VBR brain region data ~435MB).
- **Workaround**: Build the production sparklis app (`cd neurolang/utils/server/neurolang-sparklis && npm run build`) and serve it via `npx vite preview --port 3150 --host 0.0.0.0` which also proxies API calls. The production build served this way avoids the proxy crash.
- **Alternative**: Use text-only queries that do not return VBR/NIfTI data to avoid the large payload issue.
- This issue DOES affect brain visualization tests (VAL-BRAIN-002 through VAL-BRAIN-005) as they need VBR query results.

### Brain visualization tests (brain-viz milestone)
- **Use port 3150 (vite preview of prod build) for ALL brain-viz tests** — not port 3100 (dev server), which will crash with VBR payloads.
- The prod build is pre-built in `neurolang/utils/server/neurolang-sparklis/dist/`.
- Start vite preview: `cd /Users/dwasserm/sources/NeuroLang/neurolang/utils/server/neurolang-sparklis && npx vite preview --port 3150 --host 0.0.0.0 &`
- Verify: `curl -sf http://localhost:3150` returns HTML, `curl -sf http://localhost:3150/v2/engines` returns engine list.

### VBR queries for brain-viz testing
- For ExplicitVBR (non-probabilistic regions): use `LeftRegion(s, agg_create_region(r)) :- destrieux(s, r) & startswith("L S", s)` on Destrieux engine
- For ExplicitVBROverlay (probabilistic): use `LeftSulcusOverlay(s, agg_create_region_overlay(0, 0, 0, p)) :- destrieux(s, r) & startswith("L S", s) & (p == 1.0)` — but NOTE: simple overlay queries may return no results; the Neurosynth CBMA query produces ExplicitVBROverlay but takes minutes.
- **Recommended VBR query** (fast, returns VBR data): `LeftRegion(s, region) :- destrieux(s, region) & startswith("L S", s)` — this returns 'destrieux_region' type in row_type.
- Check column types in results: `row_type` array in the SymbolData indicates 'ExplicitVBR' or 'ExplicitVBROverlay' which shows "View in brain" buttons.
- The BrainViewer canvas has `data-testid="brain-viewer-canvas"` and the container has `data-testid="brain-viewer"`.
- The OverlayManager appears only when overlays > 0, with `data-testid="overlay-manager"`.
- Each overlay item has `data-testid="overlay-item"` and remove button has `data-testid="overlay-remove-btn"`.
- Color bar has `data-testid="color-bar"` and `data-testid="color-bar-gradient"`.
- MNI coordinate display has `data-testid="brain-viewer-coords"` with individual `data-testid="brain-viewer-coord-x"` etc.
- The "View in brain" button in DataTable: `aria-label="View in brain"`, located in rows where VBR columns exist.
- Niivue renders inside a `<canvas>` element; the 3-slice view is configured with `sliceType: 3` (MULTIPLANAR).
- Note: Niivue canvas rendering is WebGL-based — take screenshot evidence, don't rely on DOM inspection for the rendered atlas.

### Known UI behaviors (brain-viz milestone)
- Brain viewer is positioned below the fold at default 1024x768 viewport — use 1440x900 viewport for brain-viz tests
- BrainViewer canvas: `data-testid="brain-viewer-canvas"`, container: `data-testid="brain-viewer"`, loading: `data-testid="brain-viewer-loading"`, error: `data-testid="brain-viewer-error"`
- MNI coordinates: `data-testid="brain-viewer-coords"`, individual axes: `data-testid="brain-viewer-coord-x/y/z"`, initial values are x=0.0, y=0.0, z=0.0
- Clicking the canvas moves the crosshair and updates the coordinate display via Niivue onLocationChange callback
- OverlayManager only renders when overlays > 0: `data-testid="overlay-manager"`, items: `data-testid="overlay-item"`, remove: `data-testid="overlay-remove-btn"`, clear all: `data-testid="overlay-clear-all-btn"`
- Color bar: `data-testid="color-bar"`, gradient: `data-testid="color-bar-gradient"` — appears only for isProbabilistic=true overlays
- **KNOWN BUG (brain-viz round 1)**: DataTable 'View in brain' button never appears because condition checks `typeof value === 'string'` but VBR data is returned as object `{min, max, q95, image, hash, center, idx}`. The button condition needs to be updated to handle object format. Blocks VAL-BRAIN-002, 003, 004, 005.
- **KNOWN BUG (brain-viz round 1)**: BrainViewer uses Niivue methods `setOverlayList` and `addOverlay` that don't exist in Niivue 0.43.7. Correct API: `nv.addVolume(nvImage)`. Blocks canvas overlay rendering. Blocks VAL-BRAIN-002, 003, 004.
