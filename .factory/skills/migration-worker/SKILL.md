---
name: migration-worker
description: Worker for Python build system migration, dependency updates, and compatibility fixes
---

# Migration Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features involving:
- Migrating build configuration (setup.cfg/setup.py -> pyproject.toml)
- Updating dependency version constraints
- Replacing legacy tooling (versioneer -> setuptools-scm)
- Fixing Python version compatibility issues (distutils removal, typing changes)
- Running test suites across multiple Python versions and fixing failures

## Required Skills

None.

## Work Procedure

1. **Read the feature description thoroughly.** Understand exactly what files need to change and what the expected outcome is.

2. **Investigate before changing.** Read all files that will be modified. Understand the current state. Use Grep to find all references to things being changed (e.g., if removing versioneer, find ALL files that reference it).

3. **Make changes incrementally.** Don't try to do everything in one massive edit. Make one logical change at a time and verify it works before moving on.

4. **CRITICAL: All environment management through uv.** Never use `pip install` directly, never use `virtualenv` or `python -m venv`. Always use:
   - `uv pip install -e ".[dev]"` for installing
   - `uv run --python 3.X <command>` for running under a specific Python version
   - `uv venv --python 3.X .venv-3.X` for creating version-specific venvs
   - `uv build` for building packages

5. **Verify each change.** After modifying pyproject.toml or dependencies:
   - Run `uv pip install -e ".[dev]"` to check install works
   - Run `python -c "import neurolang"` to check import works
   - Run a small subset of tests to check nothing is broken

6. **For test suite runs across Python versions:**
   - Create separate venvs: `uv venv --python 3.12 .venv-312`, etc.
   - Install in each: `VIRTUAL_ENV=.venv-312 uv pip install -e ".[dev]"`
   - Run tests in each: `uv run --python 3.12 pytest neurolang/ -x -q`
   - If tests fail, analyze whether failures are:
     a) Caused by our migration (must fix)
     b) Pre-existing on master (document but don't fix)
     c) Caused by dependency version changes (may need to adjust constraints)

7. **Run all validators before completing:**
   - `uv pip install -e ".[dev]"` succeeds
   - `uv run python -c "import neurolang; print(neurolang.__version__)"` works
   - `uv run pytest neurolang/ -x -q` passes (or pre-existing failures documented)

8. **Commit your work** with a clear commit message describing what was changed.

## Example Handoff

```json
{
  "salientSummary": "Created pyproject.toml with all metadata, deps, and extras migrated from setup.cfg. Replaced versioneer with setuptools-scm. Updated lark constraint from <1.1.3 to >=1.1.2, nilearn from <=0.10.2 to >=0.10.0, sqlalchemy from <2.0.0 to >=1.4. Verified uv pip install succeeds and import works on Python 3.12.",
  "whatWasImplemented": "Full pyproject.toml with [build-system] using setuptools+setuptools-scm, [project] metadata, dependencies with updated version bounds, [project.optional-dependencies] for dev/doc/server extras, [project.scripts] for neuro-server entry point, [tool.*] sections for black/isort/pytest. Removed versioneer.py and neurolang/_version.py. Updated neurolang/__init__.py to use importlib.metadata.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "uv pip install -e '.[dev]'", "exitCode": 0, "observation": "All deps installed successfully"},
      {"command": "uv run python -c 'import neurolang; print(neurolang.__version__)'", "exitCode": 0, "observation": "Prints 0.0.1.dev123+gabcdef"},
      {"command": "uv run pytest neurolang/tests/test_basic.py -x -q", "exitCode": 0, "observation": "5 passed"},
      {"command": "grep distutils setup.py", "exitCode": 1, "observation": "No distutils references remain"}
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": []
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- A dependency cannot be installed on one of the target Python versions and no version constraint adjustment fixes it
- The test suite has widespread failures that appear to be fundamental incompatibilities (not just a few fixable issues)
- The migration requires changes to the package's public API or behavior
- Circular dependency or conflicting version requirements between dependencies
