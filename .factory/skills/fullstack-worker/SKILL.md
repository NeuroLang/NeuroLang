---
name: fullstack-worker
description: Builds features spanning both Tornado backend and React frontend for the NeuroLang Sparklis GUI
---

# Fullstack Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features that require both backend API changes AND frontend UI changes. Examples: wiring a new API endpoint to a React component, implementing end-to-end query execution flow, connecting brain visualization to query results.

## Required Skills

- `agent-browser`: For manual verification of the full-stack flow. After implementation, start both servers and use agent-browser to verify the end-to-end behavior.

## Work Procedure

1. **Read the feature description** carefully. Identify the backend and frontend pieces. Plan the API contract (request/response shapes) first.

2. **Check existing code**: Read both backend (`neurolang/utils/server/`) and frontend (`neurolang/utils/server/neurolang-sparklis/src/`) to understand patterns.

3. **Backend first -- Write tests (RED)**:
   - Create/update Python test files in `neurolang/utils/server/tests/`
   - Test API endpoints: status codes, response shapes, error cases
   - Run: `.venv/bin/python -m pytest neurolang/utils/server/tests/ -x -v --timeout=120`

4. **Backend -- Implement (GREEN)**:
   - Create/modify handlers in `v2_handlers.py`, register routes in `app.py`
   - Run tests to confirm they pass

5. **Frontend -- Write tests (RED)**:
   - Create/update test files for React components
   - Test: rendering with mock API data, user interactions, state changes
   - Run: `cd neurolang/utils/server/neurolang-sparklis && npm test -- --run`

6. **Frontend -- Implement (GREEN)**:
   - Create/modify React components that consume the backend API
   - Add TypeScript types matching the API response shapes
   - Run tests to confirm they pass

7. **Run all validators**:
   - `.venv/bin/python -m pytest neurolang/utils/server/tests/ -x -v --timeout=120`
   - `cd neurolang/utils/server/neurolang-sparklis && npx tsc --noEmit`
   - `cd neurolang/utils/server/neurolang-sparklis && npm test -- --run`

8. **Manual verification with agent-browser**:
   - Start backend: `cd /Users/dwasserm/sources/NeuroLang && .venv/bin/python -m neurolang.utils.server.app --port=8888 &`
   - Wait for engine initialization
   - Start frontend: `cd /Users/dwasserm/sources/NeuroLang/neurolang/utils/server/neurolang-sparklis && npm run dev &`
   - Use `agent-browser` to test the full-stack flow:
     - API data loads in the UI correctly
     - User interactions trigger correct API calls
     - Results display properly
     - Error states handled
   - Record all observations.
   - Stop all processes you started.

## Example Handoff

```json
{
  "salientSummary": "Wired query execution from React frontend to Tornado backend via WebSocket. Built QueryExecutor component that sends Datalog to /v1/statementsocket, displays loading state, and renders results in a paginated DataTable. Backend: no new endpoints needed (reuses existing WebSocket). Frontend: 3 new components, 12 tests. Verified end-to-end with agent-browser: typed a Destrieux query, saw loading spinner, results appeared in table with 5 columns.",
  "whatWasImplemented": "QueryExecutor component manages WebSocket connection to /v1/statementsocket. Sends {query, engine} message on submit. Handles status updates (running, done, error, cancelled). DataTable component renders results with sortable columns and pagination (50 rows/page). ResultsPanel wraps DataTable with symbol selector dropdown. All components in src/components/execution/.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": ".venv/bin/python -m pytest neurolang/utils/server/tests/ -x -v", "exitCode": 0, "observation": "All backend tests pass"},
      {"command": "cd neurolang/utils/server/neurolang-sparklis && npx tsc --noEmit", "exitCode": 0, "observation": "No type errors"},
      {"command": "cd neurolang/utils/server/neurolang-sparklis && npm test -- --run", "exitCode": 0, "observation": "12 tests passing: WebSocket connection, message sending, loading state, results rendering, symbol switching, sorting, pagination, error display, cancel flow, empty results"}
    ],
    "interactiveChecks": [
      {"action": "Navigated to http://localhost:3100, selected Destrieux engine, typed union of regions query in editor, clicked Run", "observed": "Loading spinner appeared, then results table showed with columns: name, region. 5 rows visible."},
      {"action": "Clicked column header 'name'", "observed": "Rows sorted alphabetically by name"},
      {"action": "Typed invalid query 'foo bar', clicked Run", "observed": "Error message displayed: 'Unexpected token' with line/column info"}
    ]
  },
  "tests": {
    "added": [
      {"file": "src/components/execution/__tests__/QueryExecutor.test.tsx", "cases": [
        {"name": "sends query via WebSocket", "verifies": "WebSocket message sent with query and engine"},
        {"name": "shows loading state", "verifies": "Spinner shown while query running"},
        {"name": "renders results on success", "verifies": "DataTable rendered with result data"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- The feature depends on API endpoints or components from other features that don't exist yet
- The WebSocket protocol or API contract needs changes that would break other features
- Engine initialization fails or the backend can't start
- A significant architectural decision is needed (e.g., state management approach, routing strategy)
