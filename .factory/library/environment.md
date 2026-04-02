# Environment

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Python Versions

Target: 3.12, 3.13, 3.14 (the three latest stable Python versions as of April 2026).

Available on this machine:
- `/opt/homebrew/bin/python3.12`
- `/opt/homebrew/bin/python3.13`
- `/opt/homebrew/bin/python3.14` (also the default `python3`)

## uv

- Installed at `/Users/dwasserm/.local/bin/uv` (v0.6.11)
- ALL environment management must go through uv. No direct pip, virtualenv, conda.
- Use `uv venv --python 3.X` to create version-specific venvs
- Use `uv pip install` for package installation
- Use `uv run --python 3.X` to run commands in a specific Python version
- Use `uv build` to build sdist/wheel

## Machine

- macOS (darwin 25.3.0), ARM64
- 36 GB RAM, 11 CPU cores

## Known Quirks

- `pysdd` has C extensions that need compilation. PySDD 1.0.6 supports Python 3.8+. May need build tools (Xcode command line tools) for compilation on macOS.
- The project has a `.claude/worktrees/` directory from a previous Claude session -- ignore this entirely.
- `examples_old/cmba.json` is a huge trace file -- do not grep through it.
- The `dask_sql` / dask backend is optional and not actively tested in CI. The sqlalchemy dependency is only used there.
