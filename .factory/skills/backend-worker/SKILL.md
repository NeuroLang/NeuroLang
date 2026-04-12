---
name: backend-worker
description: Builds Tornado API endpoints and backend logic for the NeuroLang Sparklis GUI
---

# Backend Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features that are purely backend: new Tornado API endpoints, backend logic, data serialization, Python tests. No frontend changes.

## Required Skills

None.

## Work Procedure

1. **Read the feature description** carefully. Understand what API endpoints are needed, their inputs/outputs, and how they interact with the NeurolangQueryManager.

2. **Check existing code**: Read `neurolang/utils/server/app.py`, `queries.py`, `engines.py`, `responses.py` to understand patterns. New handlers go in `neurolang/utils/server/v2_handlers.py`. New routes are registered in `app.py`.

3. **Write tests first (RED)**:
   - Create test files in `neurolang/utils/server/tests/`
   - Use pytest with tornado's testing utilities (AsyncHTTPTestCase or similar)
   - Test: HTTP status codes, response shapes, error cases, edge cases
   - Run: `.venv/bin/python -m pytest neurolang/utils/server/tests/ -x -v --timeout=120`

4. **Implement (GREEN)**:
   - Create Tornado RequestHandler subclasses inheriting from `JSONRequestHandler`
   - Add routes in `app.py` under the `/v2/` prefix
   - Use `write_json_reponse()` for consistent JSON output
   - Handle errors with appropriate HTTP status codes
   - Add type hints and docstrings

5. **Run validators**:
   - `.venv/bin/python -m pytest neurolang/utils/server/tests/ -x -v --timeout=120`
   - Manual curl checks against a running server

6. **Manual verification**:
   - Start the server: `cd /Users/dwasserm/sources/NeuroLang && .venv/bin/python -m neurolang.utils.server.app --port=8888`
   - Wait for engines to initialize (check logs for "Added a created engine")
   - Test each endpoint with curl:
     ```
     curl -s http://localhost:8888/v2/engines | python3 -m json.tool
     curl -s http://localhost:8888/v2/schema/neurosynth | python3 -m json.tool | head -50
     ```
   - Record observations in the handoff.
   - Stop the server.

## Example Handoff

```json
{
  "salientSummary": "Implemented GET /v2/engines, GET /v2/schema/:engine, and GET /v2/atlas/:engine endpoints. Pytest: 8 tests passing. Manually verified all endpoints return correct data with curl against a running server.",
  "whatWasImplemented": "Three new v2 API endpoints in v2_handlers.py: V2EnginesHandler returns engine list, V2SchemaHandler returns symbols with parameter names/types/docstrings grouped by category, V2AtlasHandler returns base64-encoded atlas image. Routes registered in app.py.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": ".venv/bin/python -m pytest neurolang/utils/server/tests/test_v2_handlers.py -x -v", "exitCode": 0, "observation": "8 tests passing: engines list, schema for neurosynth, schema for destrieux, schema 404 for invalid engine, atlas returns base64, atlas 404 for invalid, schema includes functions, schema includes probabilistic symbols"}
    ],
    "interactiveChecks": [
      {"action": "curl http://localhost:8888/v2/engines", "observed": "Returns JSON array with 'neurosynth' and 'destrieux'"},
      {"action": "curl http://localhost:8888/v2/schema/neurosynth", "observed": "Returns JSON with symbols including PeakReported (params: x,y,z,id), Study (params: id), TermInStudyTFIDF (params: term,tfidf,study)"},
      {"action": "curl http://localhost:8888/v2/schema/nonexistent", "observed": "Returns 404"}
    ]
  },
  "tests": {
    "added": [
      {"file": "neurolang/utils/server/tests/test_v2_handlers.py", "cases": [
        {"name": "test_engines_list", "verifies": "GET /v2/engines returns both engine types"},
        {"name": "test_schema_neurosynth", "verifies": "GET /v2/schema/neurosynth returns symbols with params"},
        {"name": "test_schema_invalid_engine", "verifies": "GET /v2/schema/nonexistent returns 404"},
        {"name": "test_atlas_returns_base64", "verifies": "GET /v2/atlas/neurosynth returns base64 image data"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- The NeurolangQueryManager doesn't expose the needed data and modifying the core library would be required
- Engine initialization fails or datasets can't be downloaded
- Existing v1 endpoints break due to the changes
- The feature requires frontend changes that aren't part of this feature's scope
