# NeuroLang — Agent Instructions

## Quick Commands

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run all tests
python -m pytest neurolang -v --timeout=600

# Run tests with coverage
python -m pytest neurolang --cov=neurolang --cov-report=xml --timeout=600

# Run slow tests (normally skipped)
python -m pytest --runslow

# Run single test file
python -m pytest path/to/test_file.py -v
```

## Build & Setup

- **Python**: Requires ≥ 3.12 (enforced in `pyproject.toml`)
- **Package manager**: `uv` (see `uv.lock`)
- **Build backend**: hatchling + hatch-vcs (version from git tags)
- **Custom build hook** (`hatch_build.py`):
  - Runs `npm install && npm run build -- --mode dev` in `neurolang/utils/server/neurolang-web/` if `dist/` doesn't exist
  - If `NEUROLANG_DASK=1`, mutates `neurolang/config/config.ini` to set `[RAS] backend = dask`

## Architecture

```
neurolang/
├── frontend/          # User-facing APIs (NeurolangDL, NeurolangPDL)
├── datalog/           # Datalog engine, ontologies, magic sets
├── probabilistic/     # Probabilistic solvers, CP-Logic, WMC
├── relational_algebra/ # RA operations, provenance, optimizers
├── logic/             # FOL, unification, transformations
├── utils/             # Helpers, server (tornado), dask integration
└── config/            # config.ini (backend selection)
```

**Main entry points**:
- `neurolang.frontend.NeurolangDL` — deterministic queries
- `neurolang.frontend.NeurolangPDL` — probabilistic queries
- `neurolang.neurolang_compiler.NeurolangCompiler` — low-level compiler

## Testing Quirks

- **Slow tests**: Skipped by default; use `--runslow` flag
- **Dask backend**: Requires JVM via jpype. The JVM must start *before* importing neurolang to avoid segfaults (`conftest.py` handles this)
- **Faulthandler**: Disabled in `conftest.py` to prevent false segfault reports from jpype
- **Dask context**: Auto-cleared after each test via fixture in `conftest.py`
- **Timeout**: Tests have 600s timeout; CI enforces 30-minute job limit

## Configuration

**`neurolang/config/config.ini`**:
```ini
[RAS]
backend = pandas  # or 'dask' or 'sql'

[PROBABILISTIC_SOLVER]
check_unate = True
```

**Runtime backend switch**:
```python
from neurolang.config import config
config.set_query_backend("dask")  # Call BEFORE importing other neurolang modules
```

## Frontend Components

- **neurolang-web**: React frontend in `neurolang/utils/server/neurolang-web/` (built via npm)
- **neurolang-sparklis**: TypeScript component in `neurolang/utils/server/neurolang-sparklis/`
- Server CLI: `neuro-server` (from `neurolang.utils.server.app:main`)

## CI / CD

- **GitHub Actions** (`.github/workflows/tests.yml`):
  - Tests on Python 3.12, 3.13, 3.14 (ubuntu-latest)
  - Coverage upload on Python 3.12 only
  - 30-minute timeout per job

## Gotchas

1. **npm optional**: Build skips frontend if `npm` not found or `dist/` exists
2. **Dask requires Java**: Maven + Java ≥ 8 needed for dask-sql backend
3. **Version from git**: `__version__` auto-generated from git tags via hatch-vcs
4. **Import order**: With dask, ensure jpype JVM starts before neurolang import (handled by `conftest.py` in tests)
