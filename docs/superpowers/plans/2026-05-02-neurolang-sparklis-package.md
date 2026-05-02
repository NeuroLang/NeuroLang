# neurolang-sparklis Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone Python package `neurolang-sparklis` at `~/sources/neurolang-sparklis` that bundles the Sparklis React frontend and the full Tornado web server, with `neurolang` (from GitHub) as the engine-only dependency.

**Architecture:** Move `neurolang.utils.server` (app, engines, queries, responses, tests) into a new repo as `neurolang_sparklis`, update all imports to absolute, bundle the pre-built frontend in `neurolang_sparklis/dist/`, and expose two CLI commands (`neuro-sparklis` all-in-one, `neuro-sparklis-serve` static-only). A `hatch_build.py` hook rebuilds the frontend on `pip install` when npm is available.

**Tech Stack:** Python 3.8+, hatchling, uv, Tornado, React 18, TypeScript, Vite 8, Vitest, Tailwind CSS, CodeMirror 6, Niivue.

---

## File Map

### New repo root (`~/sources/neurolang-sparklis/`)

| File | Action | Purpose |
|---|---|---|
| `pyproject.toml` | Create | Package metadata, deps, CLI entry points |
| `hatch_build.py` | Create | npm build hook |
| `.gitignore` | Create | Exclude node_modules, .venv, dist artifacts |
| `README.md` | Create | Setup and usage docs |
| `neurolang_sparklis/__init__.py` | Create | Package init |
| `neurolang_sparklis/app.py` | Move+edit | Tornado app (from `neurolang.utils.server.app`) |
| `neurolang_sparklis/engines.py` | Move+edit | Engine configs |
| `neurolang_sparklis/queries.py` | Move+edit | Query manager |
| `neurolang_sparklis/responses.py` | Move+edit | Response serialisation |
| `neurolang_sparklis/queries.yaml` | Move | Predefined queries |
| `neurolang_sparklis/cli.py` | Create | Two CLI entry points |
| `neurolang_sparklis/dist/` | Copy | Pre-built frontend fallback (committed) |
| `tests/__init__.py` | Create | Test package |
| `tests/test_app.py` | Move+edit | Tornado app tests |
| `tests/test_engines.py` | Move+edit | Engine tests |
| `tests/test_queries.py` | Move+edit | Query manager tests |
| `tests/test_responses.py` | Move+edit | Response serialisation tests |
| `frontend/package.json` | Create | Reconstructed frontend package |
| `frontend/vite.config.ts` | Create | Vite config with dev proxy |
| `frontend/tsconfig.json` | Create | TypeScript config |
| `frontend/index.html` | Create | Entry HTML with config.js injection |
| `frontend/src/models/QueryModel.ts` | Create | Restored from .bak |
| `frontend/src/context/QueryContext.tsx` | Create | Scaffolded from test imports |
| `frontend/src/context/useQuery.ts` | Create | Scaffolded from test imports |
| `frontend/src/context/__tests__/QueryContext.test.tsx` | Move | Existing test |
| `frontend/src/main.tsx` | Create | React entry point |
| `frontend/src/App.tsx` | Create | Minimal root component |

---

## Task 1: Initialise the git repo and scaffold the directory structure

**Files:**
- Create: `~/sources/neurolang-sparklis/` (git repo)
- Create: `~/sources/neurolang-sparklis/.gitignore`

- [ ] **Step 1: Create the repo**

```bash
cd ~/sources
mkdir neurolang-sparklis
cd neurolang-sparklis
git init
```

Expected: `Initialized empty Git repository in ~/sources/neurolang-sparklis/.git/`

- [ ] **Step 2: Create .gitignore**

Create `~/sources/neurolang-sparklis/.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
dist/
build/

# Frontend
frontend/node_modules/
frontend/dist/

# Editors
.DS_Store
*.swp
```

- [ ] **Step 3: Create directory skeleton**

```bash
cd ~/sources/neurolang-sparklis
mkdir -p neurolang_sparklis/dist/assets
mkdir -p tests
mkdir -p frontend/src/context/__tests__
mkdir -p frontend/src/models
mkdir -p frontend/public
```

- [ ] **Step 4: Initial commit**

```bash
cd ~/sources/neurolang-sparklis
git add .gitignore
git commit -m "chore: initial repo scaffold"
```

---

## Task 2: Create `pyproject.toml` and `hatch_build.py`

**Files:**
- Create: `~/sources/neurolang-sparklis/pyproject.toml`
- Create: `~/sources/neurolang-sparklis/hatch_build.py`

- [ ] **Step 1: Write `pyproject.toml`**

Create `~/sources/neurolang-sparklis/pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "neurolang-sparklis"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "neurolang @ git+https://github.com/NeuroLang/NeuroLang.git",
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
    "tornado",
]

[project.scripts]
neuro-sparklis       = "neurolang_sparklis.cli:main"
neuro-sparklis-serve = "neurolang_sparklis.cli:serve_static"

[tool.hatch.build.targets.wheel]
include = [
    "neurolang_sparklis/**",
]

[tool.hatch.build.hooks.custom]
path = "hatch_build.py"
```

- [ ] **Step 2: Write `hatch_build.py`**

Create `~/sources/neurolang-sparklis/hatch_build.py`:

```python
"""
hatch_build.py

Build hook: runs `npm install && npm run build` from the `frontend/`
directory and copies the output to `neurolang_sparklis/dist/`.
Falls back gracefully if npm is unavailable.
"""
import logging
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent / "frontend"
DIST_SRC = FRONTEND_DIR / "dist"
DIST_DST = Path(__file__).parent / "neurolang_sparklis" / "dist"


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        npm = shutil.which("npm")
        if npm is None:
            logger.warning(
                "npm not found — skipping frontend build. "
                "Using pre-built neurolang_sparklis/dist/ as fallback."
            )
            return

        try:
            subprocess.run(
                [npm, "install"],
                cwd=FRONTEND_DIR,
                check=True,
            )
            subprocess.run(
                [npm, "run", "build"],
                cwd=FRONTEND_DIR,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Frontend build failed (%s) — using pre-built fallback.", exc
            )
            return

        if DIST_SRC.exists():
            if DIST_DST.exists():
                shutil.rmtree(DIST_DST)
            shutil.copytree(DIST_SRC, DIST_DST)
            logger.info("Frontend dist copied to %s", DIST_DST)
```

- [ ] **Step 3: Commit**

```bash
cd ~/sources/neurolang-sparklis
git add pyproject.toml hatch_build.py
git commit -m "chore: add pyproject.toml and hatch build hook"
```

---

## Task 3: Bootstrap the uv environment

**Files:** (no source files — environment only)

- [ ] **Step 1: Initialise uv and sync**

```bash
cd ~/sources/neurolang-sparklis
uv sync
```

Expected: uv creates `.venv/`, resolves deps, generates `uv.lock`.
Note: `neurolang` will be cloned from GitHub — this may take a minute.

- [ ] **Step 2: Verify environment**

```bash
cd ~/sources/neurolang-sparklis
uv run python -c "import neurolang; print('neurolang OK')"
uv run python -c "import tornado; print('tornado OK')"
```

Expected: both lines print `OK`.

- [ ] **Step 3: Commit lockfile**

```bash
cd ~/sources/neurolang-sparklis
git add uv.lock
git commit -m "chore: add uv.lock"
```

---

## Task 4: Move and adapt the Python server source files

**Files:**
- Create: `neurolang_sparklis/__init__.py`
- Create: `neurolang_sparklis/engines.py` (from `neurolang/utils/server/engines.py`)
- Create: `neurolang_sparklis/queries.py` (from `neurolang/utils/server/queries.py`)
- Create: `neurolang_sparklis/responses.py` (from `neurolang/utils/server/responses.py`)
- Create: `neurolang_sparklis/queries.yaml` (from `neurolang/utils/server/queries.yaml`)
- Create: `neurolang_sparklis/app.py` (from `neurolang/utils/server/app.py`)

The import rewrites to apply in every file:

| Old | New |
|---|---|
| `from .engines import` | `from neurolang_sparklis.engines import` |
| `from .queries import` | `from neurolang_sparklis.queries import` |
| `from .responses import` | `from neurolang_sparklis.responses import` |
| `from ...frontend` | `from neurolang.frontend` |
| `from ...frontend.` | `from neurolang.frontend.` |
| `from ...regions` | `from neurolang.regions` |
| `from ...type_system` | `from neurolang.type_system` |
| `from ...expressions` | `from neurolang.expressions` |
| `from ...commands` | `from neurolang.commands` |
| `from ..relational_algebra_set` | `from neurolang.utils.relational_algebra_set` |
| `from ...exceptions` | `from neurolang.exceptions` |

- [ ] **Step 1: Create `neurolang_sparklis/__init__.py`**

```python
"""neurolang-sparklis: Sparklis web interface for NeuroLang."""
```

- [ ] **Step 2: Copy and rewrite `engines.py`**

Copy `~/sources/NeuroLang/neurolang/utils/server/engines.py` to
`~/sources/neurolang-sparklis/neurolang_sparklis/engines.py`, then apply the
import rewrites listed above.

The only relative imports in `engines.py` are:
```python
from neurolang.frontend import NeurolangDL, NeurolangPDL
from neurolang.frontend.neurosynth_utils import StudyID
from neurolang.regions import (...)
```
These are already absolute — no changes needed to those lines. Verify there
are no remaining `from .` or `from ...` patterns:

```bash
grep -n "from \." ~/sources/neurolang-sparklis/neurolang_sparklis/engines.py
```

Expected: no output.

- [ ] **Step 3: Copy and rewrite `queries.py`**

Copy `~/sources/NeuroLang/neurolang/utils/server/queries.py` to
`~/sources/neurolang-sparklis/neurolang_sparklis/queries.py`.

Rewrite these imports at the top of the file:

```python
# OLD:
from .engines import (...)
from ..relational_algebra_set import (...)
from ...commands import CommandsMixin
from ...expressions import Command
from ...frontend.query_resolution_expressions import Symbol
from ...type_system import get_args, is_leq_informative

# NEW:
from neurolang_sparklis.engines import (...)
from neurolang.utils.relational_algebra_set import (...)
from neurolang.commands import CommandsMixin
from neurolang.expressions import Command
from neurolang.frontend.query_resolution_expressions import Symbol
from neurolang.type_system import get_args, is_leq_informative
```

Verify:
```bash
grep -n "from \." ~/sources/neurolang-sparklis/neurolang_sparklis/queries.py
```
Expected: no output.

- [ ] **Step 4: Copy and rewrite `responses.py`**

Copy `~/sources/NeuroLang/neurolang/utils/server/responses.py` to
`~/sources/neurolang-sparklis/neurolang_sparklis/responses.py`.

Rewrite:
```python
# OLD:
from neurolang.frontend.query_resolution_expressions import Symbol
from neurolang.regions import EmptyRegion, ExplicitVBR, ExplicitVBROverlay
from neurolang.type_system import get_args
from neurolang.utils.relational_algebra_set import (...)
from neurolang.exceptions import ParserError
```
These are already absolute (`neurolang.*`), so no changes needed. Verify:

```bash
grep -n "from \." ~/sources/neurolang-sparklis/neurolang_sparklis/responses.py
```
Expected: no output.

- [ ] **Step 5: Copy `queries.yaml`**

```bash
cp ~/sources/NeuroLang/neurolang/utils/server/queries.yaml \
   ~/sources/neurolang-sparklis/neurolang_sparklis/queries.yaml
```

- [ ] **Step 6: Copy and rewrite `app.py`**

Copy `~/sources/NeuroLang/neurolang/utils/server/app.py` to
`~/sources/neurolang-sparklis/neurolang_sparklis/app.py`.

Rewrite these imports:
```python
# OLD:
from neurolang.regions import ExplicitVBR, ExplicitVBROverlay
from .engines import DestrieuxEngineConf, NeurosynthEngineConf
from .queries import NeurolangQueryManager
from .responses import (
    CustomQueryResultsEncoder,
    QueryResults,
    base64_encode_nifti,
)

# NEW:
from neurolang.regions import ExplicitVBR, ExplicitVBROverlay
from neurolang_sparklis.engines import DestrieuxEngineConf, NeurosynthEngineConf
from neurolang_sparklis.queries import NeurolangQueryManager
from neurolang_sparklis.responses import (
    CustomQueryResultsEncoder,
    QueryResults,
    base64_encode_nifti,
)
```

Also update the static path near the top of `app.py`. Change:
```python
static_path = str(Path(__file__).resolve().parent / "neurolang-web" / "dist")
```
to:
```python
static_path = str(Path(__file__).resolve().parent / "dist")
```

Verify:
```bash
grep -n "from \." ~/sources/neurolang-sparklis/neurolang_sparklis/app.py
```
Expected: no output.

- [ ] **Step 7: Smoke-test imports**

```bash
cd ~/sources/neurolang-sparklis
uv run python -c "from neurolang_sparklis.engines import NeurosynthEngineConf; print('engines OK')"
uv run python -c "from neurolang_sparklis.responses import QueryResults; print('responses OK')"
uv run python -c "from neurolang_sparklis.queries import NeurolangQueryManager; print('queries OK')"
uv run python -c "from neurolang_sparklis.app import Application; print('app OK')"
```

Expected: all four lines print `OK`.

- [ ] **Step 8: Commit**

```bash
cd ~/sources/neurolang-sparklis
git add neurolang_sparklis/
git commit -m "feat: move tornado server source into neurolang_sparklis"
```

---

## Task 5: Copy pre-built frontend dist as committed fallback

**Files:**
- Create: `neurolang_sparklis/dist/index.html`
- Create: `neurolang_sparklis/dist/assets/index-*.js`
- Create: `neurolang_sparklis/dist/assets/index-*.css`

- [ ] **Step 1: Copy dist files**

```bash
cp ~/sources/NeuroLang/neurolang/utils/server/neurolang-sparklis/dist/index.html \
   ~/sources/neurolang-sparklis/neurolang_sparklis/dist/index.html

cp ~/sources/NeuroLang/neurolang/utils/server/neurolang-sparklis/dist/assets/* \
   ~/sources/neurolang-sparklis/neurolang_sparklis/dist/assets/
```

- [ ] **Step 2: Add config.js script tag to index.html**

Edit `~/sources/neurolang-sparklis/neurolang_sparklis/dist/index.html`.

Change:
```html
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>NeuroLang Sparklis</title>
```
to:
```html
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>NeuroLang Sparklis</title>
    <script src="/config.js"></script>
```

- [ ] **Step 3: Verify dist is complete**

```bash
ls ~/sources/neurolang-sparklis/neurolang_sparklis/dist/
ls ~/sources/neurolang-sparklis/neurolang_sparklis/dist/assets/
```

Expected: `index.html` and `assets/` with `.js` and `.css` files.

- [ ] **Step 4: Commit**

```bash
cd ~/sources/neurolang-sparklis
git add neurolang_sparklis/dist/
git commit -m "feat: add pre-built frontend dist as committed fallback"
```

---

## Task 6: Create the CLI (`neurolang_sparklis/cli.py`)

**Files:**
- Create: `~/sources/neurolang-sparklis/neurolang_sparklis/cli.py`

- [ ] **Step 1: Write `cli.py`**

Create `~/sources/neurolang-sparklis/neurolang_sparklis/cli.py`:

```python
"""
cli.py

Two entry points:
  neuro-sparklis        — all-in-one: starts the full NeuroLang Tornado server
                          serving the Sparklis frontend from the bundled dist/.
  neuro-sparklis-serve  — static frontend only: serves dist/ with a minimal
                          Tornado static file server; writes a config.js so
                          the frontend knows the backend URL.
"""
import logging
import os
from pathlib import Path

import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options

DIST_DIR = Path(__file__).resolve().parent / "dist"

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# neuro-sparklis  (all-in-one)
# ---------------------------------------------------------------------------

def main():
    """Start the full NeuroLang Tornado server with the Sparklis frontend."""
    # Import here so startup errors are reported clearly
    from neurolang_sparklis.app import make_app  # noqa: F401 — defined below

    tornado.options.parse_command_line()
    _write_config_js(DIST_DIR, f"http://localhost:{options.port}")

    app = make_app(static_path=str(DIST_DIR))
    app.listen(options.port)
    LOG.info("neuro-sparklis running on http://localhost:%d", options.port)
    tornado.ioloop.IOLoop.current().start()


# ---------------------------------------------------------------------------
# neuro-sparklis-serve  (static frontend only)
# ---------------------------------------------------------------------------

define(
    "backend_url",
    default="http://localhost:8888",
    help="URL of the running NeuroLang backend (used by the frontend)",
    type=str,
)


def serve_static():
    """Serve only the Sparklis static frontend."""
    tornado.options.parse_command_line()
    _write_config_js(DIST_DIR, options.backend_url)

    app = tornado.web.Application(
        [
            (
                r"/(.*)",
                tornado.web.StaticFileHandler,
                {
                    "path": str(DIST_DIR),
                    "default_filename": "index.html",
                },
            ),
        ]
    )
    port = options.port
    app.listen(port)
    LOG.info(
        "neuro-sparklis-serve on http://localhost:%d → backend %s",
        port,
        options.backend_url,
    )
    tornado.ioloop.IOLoop.current().start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config_js(dist_dir: Path, backend_url: str) -> None:
    """Write a config.js into dist_dir so the frontend can read the API URL."""
    config_js = dist_dir / "config.js"
    config_js.write_text(
        f'window.NEUROLANG_API_URL = "{backend_url}";\n', encoding="utf-8"
    )
    LOG.debug("Wrote %s", config_js)
```

- [ ] **Step 2: Expose `make_app` in `app.py`**

Open `~/sources/neurolang-sparklis/neurolang_sparklis/app.py` and find the
`main()` function (or equivalent startup code). Add a `make_app` factory
function just before it:

```python
def make_app(static_path: str = None) -> Application:
    """Create and return the Tornado Application.

    Parameters
    ----------
    static_path : str, optional
        Path to the directory of static files to serve.
        Defaults to the bundled ``dist/`` directory.
    """
    import os
    from pathlib import Path as _Path

    if static_path is None:
        static_path = str(_Path(__file__).resolve().parent / "dist")

    # patch the module-level variable used by Application.__init__
    import neurolang_sparklis.app as _self
    _self.static_path = static_path

    data_dir = options.data_dir
    queries_file = _Path(__file__).resolve().parent / "queries.yaml"
    nqm = NeurolangQueryManager(
        [NeurosynthEngineConf(data_dir), DestrieuxEngineConf(data_dir)],
        queries_file=str(queries_file),
    )
    return Application(nqm)
```

- [ ] **Step 3: Verify CLI entry points are importable**

```bash
cd ~/sources/neurolang-sparklis
uv run python -c "from neurolang_sparklis.cli import main, serve_static; print('cli OK')"
```

Expected: `cli OK`

- [ ] **Step 4: Commit**

```bash
cd ~/sources/neurolang-sparklis
git add neurolang_sparklis/cli.py neurolang_sparklis/app.py
git commit -m "feat: add CLI entry points (neuro-sparklis, neuro-sparklis-serve)"
```

---

## Task 7: Move and adapt the Python tests

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_app.py`
- Create: `tests/test_engines.py`
- Create: `tests/test_queries.py`
- Create: `tests/test_responses.py`

- [ ] **Step 1: Copy test files**

```bash
NLSRC=~/sources/NeuroLang/neurolang/utils/server/tests
DEST=~/sources/neurolang-sparklis/tests

touch $DEST/__init__.py
cp $NLSRC/test_app.py $DEST/test_app.py
cp $NLSRC/test_engines.py $DEST/test_engines.py
cp $NLSRC/test_queries.py $DEST/test_queries.py
cp $NLSRC/test_responses.py $DEST/test_responses.py
```

- [ ] **Step 2: Rewrite imports in all test files**

In each test file, change relative imports of the form `from ..app import`,
`from ..queries import`, etc. to absolute:

```python
# OLD pattern (in all test files):
from ..app import Application
from ..queries import NeurolangQueryManager
from ..engines import (...)
from ..responses import (...)

# NEW:
from neurolang_sparklis.app import Application
from neurolang_sparklis.queries import NeurolangQueryManager
from neurolang_sparklis.engines import (...)
from neurolang_sparklis.responses import (...)
```

Also update any `from neurolang.*` imports — these should already be
absolute, but verify:

```bash
grep -n "from \.\." ~/sources/neurolang-sparklis/tests/*.py
```

Expected: no output.

- [ ] **Step 3: Run the tests**

```bash
cd ~/sources/neurolang-sparklis
uv run pytest tests/ -v --timeout=60 -x
```

Expected: tests collected and passing (or clearly skipped). Fix any import
errors before moving on.

- [ ] **Step 4: Commit**

```bash
cd ~/sources/neurolang-sparklis
git add tests/
git commit -m "feat: move and adapt server tests"
```

---

## Task 8: Reconstruct the frontend scaffolding

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`

- [ ] **Step 1: Write `frontend/package.json`**

```json
{
  "name": "neurolang-sparklis",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --port 3100",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "typecheck": "tsc --noEmit",
    "lint": "eslint src"
  },
  "dependencies": {
    "@codemirror/language": "6.12.3",
    "@codemirror/state": "6.6.0",
    "@codemirror/view": "6.41.0",
    "@lezer/common": "1.5.1",
    "@lezer/highlight": "1.2.3",
    "@lezer/lr": "1.4.8",
    "@niivue/niivue": "0.43.7",
    "react": "18.3.1",
    "react-dom": "18.3.1",
    "rxjs": "7.8.2"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.4.0",
    "@testing-library/react": "^16.0.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "autoprefixer": "^10.4.0",
    "eslint": "^9.0.0",
    "jsdom": "^25.0.0",
    "postcss": "^8.4.0",
    "tailwindcss": "3.4.19",
    "typescript": "5.9.3",
    "vite": "8.0.7",
    "vitest": "4.1.3"
  }
}
```

- [ ] **Step 2: Write `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Write `frontend/vite.config.ts`**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
  },
  server: {
    port: 3100,
    proxy: {
      '/v1': 'http://localhost:8888',
      '/v2': 'http://localhost:8888',
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
})
```

- [ ] **Step 4: Write `frontend/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>NeuroLang Sparklis</title>
    <script src="/config.js"></script>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 5: Write `frontend/tailwind.config.js`**

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: { extend: {} },
  plugins: [],
}
```

- [ ] **Step 6: Write `frontend/postcss.config.js`**

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

- [ ] **Step 7: Write `frontend/src/test-setup.ts`**

```typescript
import '@testing-library/jest-dom'
```

- [ ] **Step 8: Write minimal `frontend/src/main.tsx`**

```typescript
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

- [ ] **Step 9: Write minimal `frontend/src/App.tsx`**

```typescript
import React from 'react'

export default function App() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold">NeuroLang Sparklis</h1>
    </div>
  )
}
```

- [ ] **Step 10: Write minimal `frontend/src/index.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 11: Commit scaffolding**

```bash
cd ~/sources/neurolang-sparklis
git add frontend/
git commit -m "feat: scaffold frontend package.json, vite config, tsconfig"
```

---

## Task 9: Restore the frontend source files

**Files:**
- Create: `frontend/src/models/QueryModel.ts`
- Create: `frontend/src/context/QueryContext.tsx`
- Create: `frontend/src/context/useQuery.ts`
- Move: `frontend/src/context/__tests__/QueryContext.test.tsx`

- [ ] **Step 1: Restore `QueryModel.ts` from the .bak file**

```bash
cp ~/sources/NeuroLang/neurolang/utils/server/neurolang-sparklis/src/models/QueryModel.ts.bak \
   ~/sources/neurolang-sparklis/frontend/src/models/QueryModel.ts
```

- [ ] **Step 2: Copy the existing test file**

```bash
cp ~/sources/NeuroLang/neurolang/utils/server/neurolang-sparklis/src/context/__tests__/QueryContext.test.tsx \
   ~/sources/neurolang-sparklis/frontend/src/context/__tests__/QueryContext.test.tsx
```

- [ ] **Step 3: Scaffold `QueryContext.tsx`**

The test file imports these from `QueryContext`:
- `QueryProvider` (component)
- `QueryContextValue` (type with: `coordinationStatus`, `coordinationWarnings`,
  `state`, `model`, `canUndo`, `setDatalogText`, `coordinate`, `refresh`)

Create `frontend/src/context/QueryContext.tsx`:

```typescript
import React, { createContext, useContext, useState, useCallback } from 'react'
import { QueryModel, QueryState, parseDatalog, serializeToDatalog } from '../models/QueryModel'

export type CoordinationStatus = 'none' | 'full' | 'partial' | 'failed'

export interface QueryContextValue {
  /** Current immutable query state */
  state: QueryState
  /** Mutable query builder with undo/redo */
  model: QueryModel
  /** Whether undo is available */
  canUndo: boolean
  /** Datalog text in the code editor */
  datalogText: string
  /** Set the datalog text (does not trigger coordination) */
  setDatalogText: (text: string) => void
  /** Parse datalogText and sync to model state */
  coordinate: () => void
  /** Re-serialize model state to datalogText */
  refresh: () => void
  /** Status of the last coordination attempt */
  coordinationStatus: CoordinationStatus
  /** Warnings produced during the last coordination */
  coordinationWarnings: string[]
}

const QueryContext = createContext<QueryContextValue | null>(null)

export function QueryProvider({ children }: { children: React.ReactNode }) {
  const [model] = useState(() => new QueryModel())
  const [state, setState] = useState<QueryState>(model.state)
  const [datalogText, setDatalogText] = useState('')
  const [coordinationStatus, setCoordinationStatus] =
    useState<CoordinationStatus>('none')
  const [coordinationWarnings, setCoordinationWarnings] = useState<string[]>([])

  const canUndo = model.canUndo

  const coordinate = useCallback(() => {
    const parsed = parseDatalog(datalogText)
    if (parsed === null) {
      setCoordinationStatus('failed')
      setCoordinationWarnings([
        `Could not parse Datalog text: "${datalogText.slice(0, 60)}"`,
      ])
      return
    }
    model.reset(parsed)
    setState({ ...model.state })
    setCoordinationStatus('full')
    setCoordinationWarnings([])
  }, [datalogText, model])

  const refresh = useCallback(() => {
    setState({ ...model.state })
    setDatalogText(serializeToDatalog(model.state))
  }, [model])

  const value: QueryContextValue = {
    state,
    model,
    canUndo,
    datalogText,
    setDatalogText,
    coordinate,
    refresh,
    coordinationStatus,
    coordinationWarnings,
  }

  return (
    <QueryContext.Provider value={value}>{children}</QueryContext.Provider>
  )
}

export function useQueryContext(): QueryContextValue {
  const ctx = useContext(QueryContext)
  if (!ctx) throw new Error('useQueryContext must be used inside QueryProvider')
  return ctx
}
```

- [ ] **Step 4: Scaffold `useQuery.ts`**

The test imports `useQuery` from `./useQuery` and uses it to get the full
`QueryContextValue`. Create `frontend/src/context/useQuery.ts`:

```typescript
import { useQueryContext } from './QueryContext'
export { useQueryContext as useQuery }
```

- [ ] **Step 5: Commit**

```bash
cd ~/sources/neurolang-sparklis
git add frontend/src/
git commit -m "feat: restore QueryModel and scaffold QueryContext, useQuery"
```

---

## Task 10: Verify the frontend builds and tests pass

- [ ] **Step 1: Install frontend deps**

```bash
cd ~/sources/neurolang-sparklis/frontend
npm install
```

Expected: `node_modules/` created, no errors.

- [ ] **Step 2: Run frontend tests**

```bash
cd ~/sources/neurolang-sparklis/frontend
npm test -- --run
```

Expected: `QueryContext – coordination` suite passes (4 tests).

If tests fail, check the `QueryContext.tsx` scaffold — most likely the test
expects `model.addPredicate` to work and `refresh()` to sync back to
datalogText. Verify `refresh()` calls `setState` and `setDatalogText`.

- [ ] **Step 3: Run TypeScript typecheck**

```bash
cd ~/sources/neurolang-sparklis/frontend
npm run typecheck
```

Expected: no errors. Fix any type errors (usually unused vars or missing
`resetIdCounter` export — it's a static method in the class, accessed as
`QueryModel.resetIdCounter()`).

Check that `QueryModel.ts` exports `resetIdCounter` as a static method. Open
the file and confirm:

```typescript
export class QueryModel {
  // ...
  static resetIdCounter(): void {
    _idCounter = 0
  }
  // ...
}
```

If it only exports a standalone `resetIdCounter` function, update the test
import or add the static method to the class.

- [ ] **Step 4: Run the frontend build**

```bash
cd ~/sources/neurolang-sparklis/frontend
npm run build
```

Expected: `frontend/dist/` created with `index.html` and `assets/`.

- [ ] **Step 5: Copy fresh build into package dist**

```bash
cp -r ~/sources/neurolang-sparklis/frontend/dist/* \
      ~/sources/neurolang-sparklis/neurolang_sparklis/dist/
```

Re-add the `<script src="/config.js"></script>` tag to
`neurolang_sparklis/dist/index.html` if the copy overwrote it (see Task 5
Step 2).

- [ ] **Step 6: Commit**

```bash
cd ~/sources/neurolang-sparklis
git add neurolang_sparklis/dist/ frontend/src/
git commit -m "feat: frontend tests passing and dist updated from source build"
```

---

## Task 11: Write the README and final verification

**Files:**
- Create: `~/sources/neurolang-sparklis/README.md`

- [ ] **Step 1: Write README**

Create `~/sources/neurolang-sparklis/README.md`:

```markdown
# neurolang-sparklis

Sparklis web interface for [NeuroLang](https://github.com/NeuroLang/NeuroLang).

Bundles the React/TypeScript Sparklis frontend and the Tornado REST server.
`neurolang` itself (the probabilistic logic engine) is a dependency, installed
directly from GitHub.

## Requirements

- Python 3.8+
- [uv](https://github.com/astral-sh/uv)
- Node.js ≥ 18 (optional — only needed to rebuild the frontend)

## Install

```bash
uv sync
```

This installs all Python dependencies including `neurolang` from GitHub,
and (if `npm` is available) rebuilds the frontend.

## Usage

### All-in-one (backend + frontend)

```bash
uv run neuro-sparklis --port 8888 --data-dir ~/neurolang_data
```

Open http://localhost:8888 in your browser.

### Frontend only (requires a running NeuroLang backend)

```bash
uv run neuro-sparklis-serve --port 3100 --backend-url http://localhost:8888
```

## Development

### Python

```bash
uv run pytest tests/ -v
```

### Frontend

```bash
cd frontend
npm install
npm run dev      # dev server on port 3100 with HMR
npm test         # vitest
npm run build    # rebuild dist/
```
```

- [ ] **Step 2: Run full Python test suite**

```bash
cd ~/sources/neurolang-sparklis
uv run pytest tests/ -v --timeout=60
```

Expected: all tests pass (or are explicitly skipped for slow/network tests).

- [ ] **Step 3: Run frontend tests one more time**

```bash
cd ~/sources/neurolang-sparklis/frontend
npm test -- --run
```

Expected: all passing.

- [ ] **Step 4: Verify CLI entry points are installed**

```bash
cd ~/sources/neurolang-sparklis
uv run neuro-sparklis --help
uv run neuro-sparklis-serve --help
```

Expected: both print usage/help without errors.

- [ ] **Step 5: Final commit**

```bash
cd ~/sources/neurolang-sparklis
git add README.md
git commit -m "docs: add README with install and usage instructions"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - Repo at `~/sources/neurolang-sparklis` ✓ Task 1
  - `pyproject.toml` with GitHub dep ✓ Task 2
  - `hatch_build.py` ✓ Task 2
  - `uv` environment ✓ Task 3
  - All server files moved + imports rewritten ✓ Task 4
  - Pre-built dist committed ✓ Task 5
  - CLI with both modes ✓ Task 6
  - Tests moved + adapted ✓ Task 7
  - Frontend scaffolded ✓ Task 8
  - Frontend source restored ✓ Task 9
  - Build + tests verified ✓ Task 10
  - README ✓ Task 11
  - `config.js` injection for `--backend-url` ✓ Task 5+6
- [x] **No placeholders** — all steps include concrete code or commands
- [x] **Type consistency** — `QueryContextValue`, `QueryModel`, `CoordinationStatus` used consistently across Tasks 9–10
- [x] **`resetIdCounter`** — noted in Task 10 Step 3 as a potential issue (static vs standalone function)
