# User Testing Guide: NeuroLang Build Migration

## Overview

This is a pure Python library project. There are NO web services, no database, and no server to start. All testing is done via shell commands.

## Testing Tool

All assertions use `shell` as the testing tool. No browser automation or TUI testing needed.

## No Services Required

This project has no services to start or stop. `services.yaml` only contains build/test commands.

## Test Commands

- Install: `uv pip install -e ".[dev]"`
- Multi-version test: `uv run --python 3.X pytest neurolang/ -x -q`
- Build: `uv build`

## Validation Concurrency

### Surface: shell

Max concurrent validators: 3 (shell commands are CPU/disk bound, machine has 11 cores and 36GB RAM, but uv installs may share pip cache, so limit to 3 groups)

## Environment Notes

- `uv` is at `/Users/dwasserm/.local/bin/uv`
- Python 3.12 at `/opt/homebrew/bin/python3.12`
- Python 3.13 at `/opt/homebrew/bin/python3.13`
- Python 3.14 at `/opt/homebrew/bin/python3.14`
- Machine: macOS, 36GB RAM, 11 CPU cores, ~20K free pages (16KB pages = ~320MB free)
- Pre-existing test failures on master are NOT this mission's responsibility

## Flow Validator Guidance: shell

- All assertions are verified by running shell commands in the repo root
- Assertions don't mutate shared state (they're all read-only checks of the repo)
- Multiple subagents can run concurrently safely (no shared mutable state for these assertions)
- Use `/Users/dwasserm/.local/bin/uv` for all uv commands
- Repo root: `/Users/dwasserm/sources/NeuroLang`
- No isolation resources needed (read-only shell checks)

## Known Issues / Setup Notes

- `setup.cfg` still contains `[versioneer]` section (not `[metadata]` or `[options]`) — this is acceptable
- `neurolang/_version.pya` exists (renamed from `_version.py`) — this is expected
- `MANIFEST.in` only contains `include pyproject.toml` (versioneer refs removed) — expected
