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
