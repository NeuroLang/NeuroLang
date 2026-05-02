# neurolang-sparklis — New Standalone Package Design

**Date:** 2026-05-02  
**Status:** Approved

---

## Overview

Extract the Sparklis React/TypeScript web interface and its Tornado server layer from the
`neurolang` repository into a standalone Python package `neurolang-sparklis`, hosted at
`~/sources/neurolang-sparklis`. The new package declares `neurolang` as a runtime dependency
(engine only) and owns the entire web stack: Tornado server, REST API handlers, engine
configuration, and the React frontend.

Environment management uses `uv` throughout.

---

## Repository Structure

```
~/sources/neurolang-sparklis/
├── pyproject.toml                  # build system, runtime deps, CLI entry points
├── hatch_build.py                  # npm build hook; copies frontend/dist → neurolang_sparklis/dist
├── uv.lock                         # locked dependency manifest (committed)
├── README.md
├── neurolang_sparklis/
│   ├── __init__.py
│   ├── app.py                      # Tornado app (moved from neurolang.utils.server.app)
│   ├── engines.py                  # engine configs (moved from neurolang.utils.server.engines)
│   ├── queries.py                  # query manager (moved from neurolang.utils.server.queries)
│   ├── responses.py                # response serialisation (moved from neurolang.utils.server.responses)
│   ├── queries.yaml                # predefined queries (moved)
│   ├── cli.py                      # CLI entry points (see CLI section)
│   └── dist/                       # pre-built frontend bundle (committed as fallback)
│       ├── index.html
│       └── assets/
│           ├── index-*.js
│           └── index-*.css
├── frontend/                       # React/TypeScript source
│   ├── package.json                # reconstructed from node_modules/.package-lock.json
│   ├── vite.config.ts              # reconstructed
│   ├── tsconfig.json               # reconstructed
│   ├── index.html                  # minimal reconstructed entry point
│   └── src/
│       ├── context/                # copied from neurolang-sparklis/src/context/
│       │   ├── __tests__/
│       │   │   └── QueryContext.test.tsx
│       │   ├── QueryContext.tsx    # (source to be restored)
│       │   └── useQuery.ts         # (source to be restored)
│       └── models/
│           └── QueryModel.ts       # (restored from .bak)
└── tests/                          # Python tests (moved from neurolang.utils.server.tests)
    ├── __init__.py
    ├── test_app.py
    ├── test_engines.py
    ├── test_queries.py
    └── test_responses.py
```

`.gitignore` excludes: `frontend/node_modules/`, `frontend/dist/`, `__pycache__/`, `.venv/`,
`*.egg-info/`.

`neurolang_sparklis/dist/` is **committed** to git as a pre-built fallback so the package
installs and works even without npm.

---

## Python Package (`pyproject.toml`)

Build backend: `hatchling` (consistent with parent repo).

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "neurolang-sparklis"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "neurolang",          # engine only — must be installed separately (not on PyPI);
                          # install with: uv pip install -e /path/to/neurolang
    "tornado",
    "nibabel",
    "nilearn",
    "matplotlib",
    "pandas",
    "numpy",
    "seaborn",
    "pyyaml",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-asyncio",
    "pytest-timeout",
]

[project.scripts]
neuro-sparklis       = "neurolang_sparklis.cli:main"
neuro-sparklis-serve = "neurolang_sparklis.cli:serve_static"

[tool.hatch.build.targets.wheel]
include = [
    "neurolang_sparklis/**",
]
```

---

## Build Hook (`hatch_build.py`)

Mirrors the parent repo's pattern:

1. Check if `npm` is available in PATH.
2. If yes: run `npm install && npm run build` from `frontend/`.
3. Copy `frontend/dist/` → `neurolang_sparklis/dist/` (overwrite).
4. If npm unavailable or build fails: log a warning and leave the committed
   `neurolang_sparklis/dist/` untouched so the install still succeeds.

---

## CLI (`neurolang_sparklis/cli.py`)

### `neuro-sparklis` — all-in-one

Entry point: `neurolang_sparklis.cli:main`

- Accepts the same options as the current `neuro-server`:
  `--port` (default 8888), `--data-dir` (default `~/neurolang_data`), `--npm-build`.
- Overrides the static path in `app.py` to point at
  `neurolang_sparklis/dist/` (bundled with the package) instead of
  `neurolang-web/dist/`.
- Starts the full Tornado application (engines, query manager, REST API, static serving).

### `neuro-sparklis-serve` — static frontend only

Entry point: `neurolang_sparklis.cli:serve_static`

- Options: `--port` (default 8888), `--backend-url` (default `http://localhost:8888`).
- Starts a minimal Tornado `StaticFileHandler` pointing at `neurolang_sparklis/dist/`.
- Does **not** start any NeuroLang engine; expects a separately-running backend.
- At startup, `cli.py` writes a `config.js` into `neurolang_sparklis/dist/` that sets
  `window.NEUROLANG_API_URL = "<backend-url>"`. The frontend reads this global at
  initialisation time. `index.html` includes `<script src="/config.js"></script>` before
  the main bundle.

---

## Moved Server Code — Import Changes

All relative imports in the moved files are updated to absolute:

| Old import | New import |
|---|---|
| `from .engines import …` | `from neurolang_sparklis.engines import …` |
| `from .queries import …` | `from neurolang_sparklis.queries import …` |
| `from .responses import …` | `from neurolang_sparklis.responses import …` |
| `from ...frontend import …` | `from neurolang.frontend import …` |
| `from ...regions import …` | `from neurolang.regions import …` |
| `from ...type_system import …` | `from neurolang.type_system import …` |
| `from ...expressions import …` | `from neurolang.expressions import …` |
| `from ..relational_algebra_set import …` | `from neurolang.utils.relational_algebra_set import …` |

---

## Frontend Reconstruction (`frontend/`)

The `package.json` is reconstructed from `node_modules/.package-lock.json` with these
known versions:

**Runtime dependencies:**
- `react: 18.3.1`, `react-dom: 18.3.1`
- `@codemirror/language: 6.12.3`, `@codemirror/state: 6.6.0`, `@codemirror/view: 6.41.0`
- `@lezer/common: 1.5.1`, `@lezer/highlight: 1.2.3`, `@lezer/lr: 1.4.8`
- `@niivue/niivue: 0.43.7`
- `rxjs: 7.8.2`

**Dev dependencies:**
- `vite: 8.0.7`, `typescript: 5.9.3`, `tailwindcss: 3.4.19`
- `vitest: 4.1.3`, `@testing-library/react`, `@testing-library/jest-dom`

`vite.config.ts` configures:
- Dev proxy: API calls to `/v2/` forwarded to `http://localhost:8888`
- Build output: `dist/`

`QueryModel.ts` is restored from `QueryModel.ts.bak`.
`QueryContext.tsx` and `useQuery.ts` are scaffolded from the test file imports.

---

## Development Workflow

```bash
# Clone and set up
cd ~/sources/neurolang-sparklis
uv sync                       # creates .venv, installs all deps including neurolang

# Start the full stack
uv run neuro-sparklis --port 8888

# Start frontend only (requires running neurolang backend)
uv run neuro-sparklis-serve --port 3100 --backend-url http://localhost:8888

# Frontend development
cd frontend
npm install
npm run dev                   # dev server on port 3100 with HMR
npm test                      # vitest
npm run build                 # production build → frontend/dist/

# Run Python tests
uv run pytest tests/ -v
```

---

## What Remains in `neurolang`

The `neurolang.utils.server` module is **removed** from the parent repo after this package
is created. The `neuro-server` entry point in `neurolang` is deprecated in favour of
`neuro-sparklis`. The `neurolang-web` frontend remains in the parent repo (different UI,
unrelated to this work).

---

## Success Criteria

1. `uv sync` in the new repo succeeds.
2. `uv run neuro-sparklis` starts and serves the Sparklis UI at `http://localhost:8888`.
3. `uv run neuro-sparklis-serve` starts and serves static files.
4. All moved Python tests pass: `uv run pytest tests/ -v`.
5. `neurolang_sparklis/dist/` is present and committed so the package works without npm.
6. `npm run build` from `frontend/` succeeds and updates `neurolang_sparklis/dist/`.
