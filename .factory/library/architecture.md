# Architecture

## System Overview

The NeuroLang Sparklis GUI is a **hybrid query builder** that combines visual point-and-click query construction (inspired by Sparklis for SPARQL) with a traditional code editor, layered on top of the NeuroLang probabilistic Datalog engine.

## Components

### 1. Tornado Backend (Python, port 8888)
- **Existing code** at `neurolang/utils/server/app.py`: serves both the old frontend (`/`) and new frontend (`/sparklis/`)
- **New v2 handlers** at `neurolang/utils/server/v2_handlers.py`: schema introspection, suggestions, query execution
- **NeurolangQueryManager** (`queries.py`): manages a pool of NeuroLang engine instances per engine type (Neurosynth, Destrieux). Submits queries to a ThreadPoolExecutor.
- **Engine configurations** (`engines.py`): `NeurosynthEngineConf` and `DestrieuxEngineConf` define how to create engines with their datasets.

### 2. React Frontend (TypeScript, port 3100 dev)
- Located at `neurolang/utils/server/neurolang-sparklis/`
- Built with Vite + React 18 + TypeScript
- Key UI areas:
  - **Top navbar**: App title, settings
  - **Sidebar**: Engine selector, predicate browser with search
  - **Main area** (split panels): Visual query builder + Code editor (top), Results table + Brain viewer (bottom)

### 3. NeuroLang Engine (Python library)
- `NeurolangPDL`: the probabilistic Datalog frontend class
- Key introspection APIs:
  - `nl.symbols`: iterator over symbol names
  - `nl.predicate_parameter_names(name)`: returns tuple of parameter names
  - `nl.compute_datalog_program_for_autocompletion(complete_code, partial_code)`: returns valid next tokens
  - `nl.execute_datalog_program(code)`: executes a program, returns results or None
  - `nl.solve_all()`: solves the current program

## Data Flow

1. **Schema introspection**: Frontend fetches GET /v2/schema/:engine -> backend acquires engine from pool -> iterates `engine.symbols` -> returns JSON with names, types, parameters, docstrings
2. **Suggestions**: Frontend POSTs partial program to /v2/suggest/:engine -> backend calls `compute_datalog_program_for_autocompletion` -> returns valid next tokens grouped by category
3. **Query execution**: Frontend sends query via WebSocket /v1/statementsocket (or new /v2/query) -> backend submits to ThreadPoolExecutor -> sends status updates -> returns results as JSON
4. **Brain images**: Query results containing ExplicitVBR/ExplicitVBROverlay are serialized as base64-encoded NIfTI by `responses.py` -> frontend decodes and passes to Niivue

## Key Invariants

- The v2 API endpoints share the same `NeurolangQueryManager` instance as v1 endpoints
- Engine pool is initialized at startup; schema endpoints must wait for engines to be ready
- All v2 endpoints inherit CORS headers from `JSONRequestHandler`
- The React app in dev mode (port 3100) proxies API calls to the Tornado backend (port 8888)
- The production build is served as static files from `/sparklis/` by Tornado

## API Response Shape

The `/v2/engines` endpoint returns an **envelope format**, not a plain array:
```json
{ "status": "ok", "data": ["neurosynth", "destrieux"] }
```
Frontend workers must handle this envelope. See `EngineSelector.tsx` for a defensive pattern that handles both the envelope and plain array cases.

## WebSocket Protocol (/v1/statementsocket)

The query execution WebSocket uses the following message shapes:

**Outgoing (frontend → backend):**
```json
{"query": "ans(x) :- PeakReported(x, y, z, s).", "engine": "neurosynth"}
```

**Incoming (backend → frontend):**
```json
{
  "status": "running" | "done" | "cancelled",
  "data": {
    "uuid": "...",
    "running": true | false,
    "done": true | false,
    "cancelled": true | false,
    "errorName": "SyntaxError | ...",
    "message": "...",
    "errorDoc": "...",
    "line_info": {"line": 1, "column": 5},
    "results": { "symbol_name": {"columns": [...], "values": [[...]], "size": N} }
  }
}
```

See `responses.py:QueryResults` for the server side and `ExecutionContext.tsx:ExecutionMessage` for the TypeScript interface.

**Note:** `line_info` is only present when the error is a `ParserError` with position info. The field is absent (not null) when unavailable.

**Note:** `responses.py:set_error_details` has a latent `UnboundLocalError` for exotic `ParserError` subclasses that have none of the three expected position-info attributes (`tokenizer`, `buf`, `line`/`column`). Standard `ParserError` instances are safe.

**Note:** `QuerySocketHandler.send_query_update()` in `app.py` instantiates `QueryResults(self.uuid, future, get_values=True, length=10000)`. The `length=10000` cap means up to 10,000 rows are sent over WebSocket (vs. the default 50). Client-side pagination in `ResultsPanel.tsx` / `DataTable` is computed from `values.length`, so all rows must be sent for pagination controls to appear correctly. **VBR warning**: When a query returns ExplicitVBR/ExplicitVBROverlay columns, each row's NIfTI image is base64-encoded and included inline in the WebSocket message. At `length=10000`, a 148-row Destrieux query can produce ~435MB WebSocket payloads that may crash the Vite dev proxy. For validation with VBR queries, use the production build or reduce `length` for VBR-heavy workloads. See also AGENTS.md "Vite WebSocket proxy limitation".

## Bidirectional Sync Architecture (QueryContext)

The `QueryContext` (`src/context/QueryContext.tsx`) manages a synchronized state between the visual query builder and the code editor using:

- **`isBuilderUpdate` ref**: A `useRef<boolean>` set to `true` synchronously just before any builder-initiated text update, then reset to `false` immediately after. The debounced `setDatalogText` handler checks `if (!isBuilderUpdate.current)` to suppress re-parsing on builder-originated changes — preventing infinite loops.
- **`PARSE_DEBOUNCE_MS = 500`**: Code-editor changes are debounced 500ms before attempting `parseDatalog()`. On success, `model.reset(parsed)` updates the visual builder. On failure, `isSynced = false` shows a desync indicator.
- **Undo/redo paths**: `undo()` and `redo()` also set/reset `isBuilderUpdate` to prevent the resulting `setDatalogText` call from scheduling a parse.

**The pattern must be applied consistently**: Any code path that programmatically sets `datalogText` must wrap it with `isBuilderUpdate.current = true / false` to prevent the debounced parse from `model.reset()`-ing state.

## CodeMirror 6 Patterns

- **Detecting programmatic vs. user edits**: The `EditorView.updateListener` should check `if (update.docChanged && !isExternalUpdate.current)` to avoid firing `onChange` for programmatic updates. The ref `isExternalUpdate` must be passed into `buildExtensions()` closure so it is accessible from inside the listener. Do NOT rely on setting a ref before `view.dispatch()` and reading it after — the listener fires synchronously during dispatch, so the ref must be in scope before the listener is created.
  - Alternatively, check `update.transactions.some(tr => tr.annotation(Transaction.userEvent) !== undefined)` to distinguish user transactions from programmatic ones (programmatic dispatches have no `userEvent` annotation).
- **Dynamic `readOnly`**: `EditorState.readOnly.of(value)` is immutable once set in the extensions array. To support dynamic changes, wrap it in a `Compartment` and reconfigure it in a `useEffect([readOnly])`. See [CodeMirror docs on Compartments](https://codemirror.net/docs/ref/#state.Compartment).
- **Agent-browser limitation**: The CodeMirror 6 editor renders as a `div[contenteditable]` with class `cm-content`. Standard `agent-browser` `fill` or `keyboard-type` commands do not reliably set CodeMirror content because CodeMirror intercepts DOM mutations. For validation, prefer building queries via the predicate browser (which calls the QueryModel API) rather than typing in the CodeMirror editor directly.

## Stale Closure Pattern in ExecutionContext.tsx

`submitQuery` in `ExecutionContext.tsx` uses `useCallback` with an empty deps array (`[]`). This means any state variable (e.g., `executionStatus`) captured in closures inside `submitQuery` (including WebSocket event handlers like `ws.onclose`, `ws.onerror`) will reflect the **initial render's value** for the lifetime of the component, not the current state.

**The correct pattern** for reading current state inside a `useCallback` with empty deps is to **mirror the state in a `useRef`** and read the ref inside the callback:

```tsx
const executionStatusRef = useRef(executionStatus);
useEffect(() => { executionStatusRef.current = executionStatus; }, [executionStatus]);
// Inside the empty-dep useCallback:
if (executionStatusRef.current === 'running') { ... }
```

This is the same principle as `isCancellingRef` in `ExecutionContext.tsx` and `isBuilderUpdate` in `QueryContext.tsx`. Any future worker adding state checks inside `useCallback` with `[]` deps must use this pattern.

**Known pre-existing stale closure:** `ws.onclose` in `submitQuery` checks `executionStatus === 'running'` directly (not via a ref), making the `ConnectionClosed` error path likely unreachable. This is non-blocking because `ws.onerror` handles the same failure scenario correctly.

## parseDatalog() — Intentionally Limited Parser

`QueryModel.parseDatalog()` (in `QueryModel.ts`) is a **regex-based parser supporting only the canonical format** produced by `serializeToDatalog()`:
```
ans(v, v1, v2) :- Pred1(v, v1), Pred2(v1, v2).
```
It returns `null` (desync) for:
- Constants or literals in arguments
- Negation (`~`)
- Arithmetic or comparison operators
- Alternative head names (only `ans` is expected)
- Any syntax that deviates from the simple conjunctive query form

This is by design. The visual builder only represents simple conjunctive queries — complex Datalog is write-only in the code editor, and will desync the visual builder. This is expected and not a bug.

## Query Coordination Architecture (NEW)

The **Query Coordination** feature enables the visual query builder to extract and display predicates from complex datalog code that cannot be fully parsed by `parseDatalog()`.

### Components

1. **Enhanced Parser** (`QueryModel.ts`): New `coordinateFromCode(text)` method that:
   - Extracts predicate instances using more permissive regex patterns
   - Ignores unsupported syntax (negation, aggregates, string literals, probabilistic annotations)
   - Returns partial results with a list of warnings about unparsable content
   - Preserves variable names and creates shared variable color mappings

2. **Coordination State** (`QueryContext.tsx`): Extended with:
   - `coordinationStatus`: 'none' | 'full' | 'partial' | 'failed'
   - `coordinationWarnings`: string[] of issues encountered during coordination
   - `coordinate()` function to trigger manual coordination

3. **UI Components**:
   - **Coordinate Button** (in `VisualQueryBuilder.tsx` toolbar): Triggers coordination
   - **Coordination Status Indicator**: Shows success/partial/warning state
   - **Warnings Panel**: Displays list of unparsable constructs when partial success

### Data Flow

**Manual Coordination:**
1. User clicks "Coordinate" button → calls `coordinate()` from QueryContext
2. `coordinate()` reads current `datalogText`
3. Calls `model.coordinateFromCode(datalogText)` → returns `{ state, warnings }`
4. If state extracted: `model.reset(state)` + set coordination status
5. If warnings exist: display warnings panel
6. Visual builder updates with extracted predicates

**Automatic Coordination (Example Load):**
1. User clicks example → `handleLoadExample(query)` executes
2. `model.reset()` clears builder
3. `setDatalogText(query)` sets code editor
4. After debounce, if `parseDatalog()` fails → automatically call `coordinate()`
5. Shows partial results with warnings for complex examples

### Key Invariants

- Coordination is **opt-in** for manual editing (doesn't auto-parse on every keystroke)
- Coordination **preserves undo/redo history** like other mutations
- Visual builder actions **always resync** and override coordination state
- Coordination **never loses user work** — it's a one-way sync (code → builder)
- Warnings are **informational only** — don't block showing parseable predicates

## Backend Quirks

- **Backend parser rejects trailing period**: The NeuroLang Datalog parser expects queries **without** a trailing period (`.`). For example, `ans(x) :- PeakReported(x, y, z, s)` is valid, but `ans(x) :- PeakReported(x, y, z, s).` raises `UnexpectedTokenError`. The visual query builder's `serializeToDatalog()` appends a period — the `submitQuery` function in `ExecutionContext.tsx` must strip it before sending. Any worker submitting queries to the backend must be aware of this constraint.
- **`get_atlas(engine_key)` raises `IndexError`** (not `KeyError`) when the engine key is unknown. The underlying `queries.py` implementation does a list comprehension and accesses `config[0]`, so an empty list raises `IndexError`. Use `except (IndexError, KeyError)` when catching 404 cases for atlas-related endpoints.
- **`NeurolangEngineSet.engine()` yields `None`** (not raising) when timeout is set and the semaphore cannot be acquired. Always check `if engine is None` and return HTTP 503 when acquiring engines directly. Example pattern in `v2_handlers.py V2SchemaHandler.get()`.
- **`JSONRequestHandler` has two definitions**: One in `neurolang/utils/server/app.py` (for v1 handlers) and one in `neurolang/utils/server/base_handlers.py` (for v2 handlers). New v2 handlers should import from `base_handlers.py`. Do NOT import from `app.py` as this creates a circular import.
- **`compute_datalog_program_for_autocompletion` mutates engine state**: This method internally calls `self.program_ir.walk()` which mutates the engine's `program_ir`. Unlike `execute_datalog_program`, the suggestion handler (`V2SuggestHandler`) does NOT use `with engine.scope:` for isolation. Future workers implementing autocomplete-like handlers should consider wrapping calls in `with engine.scope:` (which calls `push_scope()/pop_scope()`) to prevent rule accumulation across requests. See `queries.py:215-238` for the correct pattern.
- **Schema API returns integer ordinal parameter names for some predicates**: For predicates like `PeakReported`, `nl.predicate_parameter_names(name)` returns `('0', '1', '2', '3')` instead of semantic names. This is because the underlying symbol was defined without named parameters. Frontend code that uses parameter names to generate variable names should handle this case (e.g., fall back to a generic prefix like `v`, `v1`, `v2`). This affects the visual query builder's auto-variable-naming logic.
- **`/v2/suggest/:engine` response data may contain non-array scalar fields**: The `data` object in the suggestions response may include a `message` field (string) alongside category arrays when a parser error occurs. Frontend code iterating over `data` keys to render suggestion categories must guard against non-array values with `Array.isArray(value)`. See `SuggestionsPanel.tsx` for the defensive pattern.

## Nilearn Atlas Quirks

- **Nilearn >= 0.10 changed the Destrieux atlas label format**: Prior to 0.10, `fetch_atlas_destrieux_2009()` returned labels as a list of `(int, bytes)` tuples. From 0.10 onwards (confirmed with 0.13.1), labels are a plain list of strings. The fix in `engines.py` (`load_destrieux_atlas`) uses `isinstance(raw_labels[0], str)` to detect the format at runtime.
- **Expected nilearn warning about missing atlas regions**: When loading the Destrieux atlas, nilearn prints a `UserWarning` about regions `L Medial_wall` (index 42) and `R Medial_wall` (index 117) being present in the look-up table but missing from the atlas image. This is expected behavior from nilearn and does not indicate a bug.

## Niivue Integration Quirks

- **`sliceType: 3` is the MULTIPLANAR constant**: `BrainViewer.tsx` initializes Niivue with `sliceType: 3` to display all three orthogonal views (axial, sagittal, coronal) simultaneously. The Niivue enum `SLICE_TYPE.MULTIPLANAR = 3`. Use this value for three-panel display; other values (0=axial, 1=coronal, 2=sagittal, 4=render) show a single view.
- **`NVImage.loadFromBase64` requires a file extension in the `name` field**: Niivue's format detection uses the `name` field to infer the image format. Without an extension (e.g., `name: 'atlas'`), Niivue falls back to DICOM detection and throws `RangeError` when given NIfTI data. Always use `.nii` extension: `name: 'atlas.nii'`. This caused a real bug in the initial niivue-atlas-viewer implementation (commit `a97f5389`).
- **`setOverlayList` may not exist on Niivue instances**: The Niivue API for managing overlays after initial load is not stable across versions. `BrainViewer.tsx` uses a duck-typed check: `if (typeof niivueRef.current.setOverlayList === 'function')`. If absent, the fallback uses internal properties `overlaysVox` and `volumes?.slice(1)`. Use the same duck-typed approach for any code manipulating overlays dynamically.
- **Coordinate update testing is a no-op in jsdom**: `BrainViewer.tsx` exposes a `onLocationHandlerReady` callback prop to inject the Niivue location-change handler for testing. However, in jsdom (no WebGL), the Niivue canvas initialization path resolves asynchronously, meaning `capturedHandler` is always `null` in test environments. Tests that check coordinate updates guard with `if (!capturedHandler) return`, making them trivially pass without exercising the coordinate logic. For coordinate behavior, rely on manual verification; do not add new tests expecting the handler to fire in jsdom.
