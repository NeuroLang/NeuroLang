# Design: Migrate from setuptools to Hatchling

**Date:** 2026-04-17
**Status:** Approved

## Overview

Replace `setuptools` + `setuptools-scm` with `hatchling` + `hatch-vcs` as the
build backend. Move all custom install-time logic (npm frontend build, dask
config mutation) into a single `hatch_build.py` build hook.

## `pyproject.toml` changes

- `[build-system]`: change `requires` to `["hatchling", "hatch-vcs"]` and
  `build-backend` to `"hatchling.build"`.
- `[tool.setuptools.*]` sections removed; replaced with:
  - `[tool.hatch.build.targets.wheel]` — explicit `packages = ["neurolang"]`
    and `artifacts` / `include` for data files.
  - `[tool.hatch.build.hooks.custom]` — wires in `hatch_build.py`.
- `[tool.setuptools_scm]` removed; replaced with:
  - `[tool.hatch.version]` using `source = "vcs"`.
  - `[tool.hatch.version.raw-options]` with `version_file =
    "neurolang/_version_scm.py"`.

## `hatch_build.py` — custom build hook

Single file at the repo root. Implements
`hatchling.builders.hooks.plugin.interface.BuildHookInterface`.

`initialize()` runs two optional steps:

1. **npm build** — locates `npm`, runs `npm install` then
   `npm run build -- --mode dev` in
   `neurolang/utils/server/neurolang-web/`. Skips if `dist/` already exists.
   Emits a warning (does not fail) if `npm` is not found.

2. **dask config** — if env var `NEUROLANG_DASK=1` is set, reads
   `neurolang/config/config.ini` and writes `[RAS] backend = dask` before the
   wheel packages the file.

Usage:
```bash
pip install .                        # standard install, npm build attempted
NEUROLANG_DASK=1 pip install .       # also switches backend to dask
```

## `setup.py` removal

`setup.py` is deleted. No backward-compat shim.

## Unchanged

- All `[tool.pytest]`, `[tool.black]`, `[tool.isort]`, `[tool.coverage]`,
  `[tool.codespell]` sections in `pyproject.toml`.
- CI workflows.
- All `[project.*]` metadata.
