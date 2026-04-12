---
name: frontend-worker
description: Builds React UI components, pages, and frontend logic for the NeuroLang Sparklis GUI
---

# Frontend Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features that are purely frontend: React components, pages, styling, client-side state management, frontend tests. No backend changes.

## Required Skills

- `agent-browser`: For manual verification of UI components. Invoke when the feature produces visible UI changes -- after implementation is complete, start the dev server and use agent-browser to verify the UI renders correctly and interactions work.

## Work Procedure

1. **Read the feature description** carefully. Understand what UI components/pages are needed, what data they consume, and what user interactions they support.

2. **Check existing code**: Read relevant existing components in `neurolang/utils/server/neurolang-sparklis/src/` to understand patterns, imports, and conventions.

3. **Write tests first (RED)**:
   - Create test files in `__tests__/` directories or `*.test.tsx` files alongside components.
   - Use Vitest + React Testing Library.
   - Test: rendering, user interactions (clicks, typing), state changes, edge cases (empty data, loading, error states).
   - Run tests to confirm they fail: `cd neurolang/utils/server/neurolang-sparklis && npm test -- --run`

4. **Implement (GREEN)**:
   - Create/modify React components using functional components + hooks.
   - Use TypeScript with proper type definitions.
   - Follow existing patterns for API calls, state management, and styling.
   - Run tests to confirm they pass.

5. **Run validators**:
   - `cd neurolang/utils/server/neurolang-sparklis && npx tsc --noEmit` (typecheck)
   - `cd neurolang/utils/server/neurolang-sparklis && npm run lint` (lint, if configured)
   - `cd neurolang/utils/server/neurolang-sparklis && npm test -- --run` (all tests)

6. **Manual verification with agent-browser**:
   - Start the dev server: `cd neurolang/utils/server/neurolang-sparklis && npm run dev` (port 3100)
   - Also start the backend if the feature consumes API data: `cd /Users/dwasserm/sources/NeuroLang && .venv/bin/python -m neurolang.utils.server.app --port=8888 &`
   - Use `agent-browser` to navigate to the relevant page/component and verify:
     - Component renders correctly
     - Interactions work (clicks, typing, navigation)
     - Edge cases handled (empty state, loading, errors)
   - **CodeMirror 6 tip:** CM6 editors (class `cm-content`, contenteditable) do NOT respond reliably to agent-browser `fill` or keyboard-type commands. To set query content during manual verification, use the predicate browser or example queries panel. To read editor content, use JavaScript eval: `document.querySelector(".cm-content")?.textContent`.
   - Record observations in the handoff.
   - Stop all processes you started.

## Example Handoff

```json
{
  "salientSummary": "Built the predicate browser panel component with search/filter, grouped by category (relations, functions, probabilistic). Vitest: 6 tests passing. Verified with agent-browser: predicates render in groups, search filters correctly, clicking a predicate fires the onSelect callback.",
  "whatWasImplemented": "PredicateBrowser component at src/components/PredicateBrowser.tsx with PredicateCard, CategoryGroup subcomponents. Accepts symbols array prop, groups by type, includes search input with debounced filtering. Styles via CSS modules.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "cd neurolang/utils/server/neurolang-sparklis && npx tsc --noEmit", "exitCode": 0, "observation": "No type errors"},
      {"command": "cd neurolang/utils/server/neurolang-sparklis && npm test -- --run", "exitCode": 0, "observation": "6 tests passing: renders categories, filters on search, shows empty state, click fires callback, highlights probabilistic symbols, keyboard navigation"}
    ],
    "interactiveChecks": [
      {"action": "Opened http://localhost:3100, selected Neurosynth engine", "observed": "Predicate browser showed 15+ symbols grouped into Relations (PeakReported, Study, etc.), Functions (agg_create_region, etc.), Probabilistic (SelectedStudy)"},
      {"action": "Typed 'Peak' in search box", "observed": "List filtered to show only PeakReported"},
      {"action": "Clicked PeakReported", "observed": "Predicate highlighted, onSelect callback logged in console"}
    ]
  },
  "tests": {
    "added": [
      {"file": "src/components/__tests__/PredicateBrowser.test.tsx", "cases": [
        {"name": "renders symbol categories", "verifies": "Symbols are grouped by type"},
        {"name": "filters symbols on search", "verifies": "Search input filters displayed symbols"},
        {"name": "shows empty state when no matches", "verifies": "Message shown when search has no results"},
        {"name": "click fires onSelect", "verifies": "Clicking a predicate card calls the callback"},
        {"name": "highlights probabilistic symbols", "verifies": "Probabilistic symbols have distinct styling"},
        {"name": "keyboard navigation works", "verifies": "Arrow keys navigate between predicates"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- The feature requires new backend API endpoints that don't exist yet
- TypeScript types for API responses are missing and the API shape is unclear
- A dependency (npm package) needs to be added to package.json and it's unclear which version/package to use
- The feature requires changes to the Vite config or build setup that might affect other features
