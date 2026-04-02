# User Testing

## Validation Surface

This mission has a single validation surface: **CLI/shell commands**. There is no web UI, API, or interactive application to test. All validation is done by running shell commands:

1. `uv pip install -e ".[dev]"` -- package installs
2. `python -c "import neurolang; print(neurolang.__version__)"` -- package is importable
3. `pytest neurolang/` -- test suite passes
4. `uv build` -- sdist/wheel builds
5. File content checks (grep, cat) for build config correctness

### Tool

All assertions use `shell` (Execute tool). No agent-browser or tuistory needed.

## Validation Concurrency

Max concurrent validators: **5** (shell commands are lightweight, no heavy processes).

Each validator just runs shell commands -- no dev servers, no browsers, no heavy processes. Resource consumption is negligible per validator.
