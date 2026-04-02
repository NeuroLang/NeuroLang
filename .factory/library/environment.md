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
- `problog` is NOT listed in pyproject.toml dependencies (neither core nor dev extras). In the original setup.cfg it was in dev extras but was dropped during migration. Tests in `neurolang/probabilistic/cplogic/` and `neurolang/tests/test_relational_algebra_provenance.py` etc. will fail with ModuleNotFoundError. These are pre-existing failures. Do NOT add problog to core `[project] dependencies` — it should remain absent or go into dev extras only.
- `pandas 3.x` changed StringArray behavior; tests in `neurolang/datalog/tests/test_chase.py` and `test_instance.py` fail with "setting an array element with a sequence". These are pre-existing failures from the pandas version upgrade.
- `numpy 2.x` changed `np.bool` to `np.bool_` and 0d array behavior; some probabilistic tests fail. Pre-existing.

## Full Test Suite Results (test-fixes milestone - FINAL)

Tested with: `pytest neurolang/ -q --ignore=neurolang/utils/server --ignore=neurolang/probabilistic/cplogic/tests --ignore=neurolang/probabilistic/tests/test_marg_query_resolution.py --ignore=neurolang/probabilistic/tests/test_probabilistic_solvers.py --ignore=neurolang/tests/test_relational_algebra_provenance.py --ignore=neurolang/tests/test_relational_algebra_semiring.py`

The ignored test files above fail to import because problog is not installed (pre-existing).

### Python 3.12 ✓
- **984 passed**, 0 failed, 16 skipped, 1 xpassed
- ALL fixes applied: numpy nonzero, pandas StringArray, Python 3.14 annotations, nilearn labels, Union typing

### Python 3.13 ✓
- **984 passed**, 0 failed, 16 skipped, 1 xpassed
- Identical to Python 3.12 results

### Python 3.14 ✓
- **984 passed**, 0 failed, 16 skipped, 1 xpassed
- All previous 104 failures are now fixed

**Key note**: When tests are run with `-x` (stop on first failure), the baseline validator stops at the first problog import error. Use `--ignore` flags above to get meaningful test results.

## Python 3.12+ Compatibility Fixes Applied

The following changes were made to support Python 3.12/3.13/3.14:

1. **neurolang/type_system/__init__.py**: Added `NEW_TYPING_312` flag; `typing._Immutable` was removed in Python 3.12 -- conditionally import only for Python < 3.12.
2. **neurolang/CD_relations.py**: Replaced `from scipy.linalg import kron` with `from numpy import kron` -- `scipy.linalg.kron` was removed in scipy 1.17+.
3. **neurolang/commands.py** and **neurolang/frontend/neurosynth_utils.py**: Added try/except for `nilearn.datasets.utils._fetch_files` → `nilearn.datasets._utils.fetch_files` (nilearn >= 0.11).
4. **neurolang/utils/server/engines.py**: Added compatibility import for `_nilearn_fetch_files` replacing `datasets.utils._fetch_files`.
5. **neurolang/frontend/datalog/standard_syntax.py**: Fixed invalid Python escape sequences `\/` and `\.` in Lark grammar string (were SyntaxWarning in 3.12, would be SyntaxError in 3.14).
6. **neurolang/expressions.py**: Added `'__annotations__'` to `Expression.__super_attributes__` explicitly; `Constant.__init_callable_formula__` now copies `__annotations__` on Python 3.14+ where `functools.WRAPPER_ASSIGNMENTS` no longer includes it.
7. **neurolang/utils/relational_algebra_set/pandas.py**: Fixed `RelationalAlgebraSet.add()` to use `pd.concat` instead of `.loc` assignment for pandas 3.x StringArray compatibility.
8. **neurolang/probabilistic/dalvi_suciu_lift.py**: Added `np.atleast_1d()` in `connected_components()` to fix `ValueError: Calling nonzero on 0d arrays` in numpy 2.x.
9. **neurolang/tests/test_regions.py**: Updated `destrieux` fixture to handle both old (tuple list) and new (string list) nilearn labels format.
10. **neurolang/type_system/__init__.py**: Added `_has_type_args()` helper using `isinstance(args, tuple)` check; used in `is_parametrical`/`is_parameterized` for Python 3.14 where bare `Union.__args__` is a `member_descriptor` not a tuple.
