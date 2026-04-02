# Architecture

NeuroLang is a Python library for probabilistic logic programming for neuroscience research. It is a pure Python package (no C extensions in the core -- though the `pysdd` dependency has C extensions).

## Package Structure

- `neurolang/` - Main package
  - `__init__.py` - Exports + version via `_version.py` (versioneer, to be migrated)
  - `frontend/` - Datalog parser and AST (uses `lark` parser library)
  - `probabilistic/` - Probabilistic reasoning (uses `pysdd` for SDD compilation)
  - `type_system/` - Type system (uses `typing_inspect`)
  - `utils/` - Utilities including:
    - `server/` - Tornado-based web server (`neuro-server` console script)
    - `server/neurolang-web/` - Vue.js frontend app (npm-built, distributed as package data)
    - `relational_algebra_set/` - Optional dask-sql backend (uses `sqlalchemy`)

## Build System (Current - Pre-Migration)

- `setup.cfg` - Primary config: metadata, dependencies, extras (dev/doc/server), entry points, package data
- `setup.py` - Custom commands: NPMBuildCommand, DevelopCommand (--dask), InstallCommand (--dask)
- `versioneer.py` + `neurolang/_version.py` - Git-based versioning (v0.18, uses distutils)
- `pyproject.toml` - Currently only has `[tool.black]` and `[tool.isort]`
- `MANIFEST.in` - Includes versioneer files
- `.coveragerc` - Coverage config

## Key Dependencies

| Package | Current Constraint | Issue |
|---------|-------------------|-------|
| lark | >=1.1.2,<1.1.3 | Overly restrictive, latest is 1.3.1 |
| nilearn | >=0.9.0,<=0.10.2 | Old, latest is 0.13.1 |
| sqlalchemy | <2.0.0 | Used only in optional dask backend |
| pytest | <=6.2.5 | Very old constraint |
| numpydoc | <1.2.0 | Old constraint |
| typing_inspect | unpinned | Needs verification on 3.12+ |
| pysdd | unpinned | C extension, needs 3.12+ compat check |

## Versioning

Currently uses `versioneer` v0.18 which depends on `distutils` (removed in Python 3.12). Migration target: `setuptools-scm` with `importlib.metadata` for runtime version access.

## Test Infrastructure

- `pytest` with custom markers (`slow`), conftest.py has dask-related fixtures
- `make test` runs pytest with coverage
- Coverage config in `.coveragerc`
- CI: CircleCI testing on Python 3.8, 3.9, 3.10

## Entry Points

- `neuro-server` -> `neurolang.utils.server.app:main`
