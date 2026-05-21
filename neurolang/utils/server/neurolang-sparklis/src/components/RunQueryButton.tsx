/**
 * RunQueryButton.tsx
 *
 * A button component that:
 *   - In idle/done/error/cancelled state: shows "Run Query" with a play icon
 *   - In running state: shows a loading spinner AND a "Cancel" button
 *
 * On click (when idle/done/error/cancelled):
 *   - Reads the current Datalog text from QueryContext
 *   - Reads the selected engine from EngineContext
 *   - Calls ExecutionContext.submitQuery()
 *
 * On cancel click:
 *   - Calls ExecutionContext.cancelQuery()
 */
import React from 'react'
import { useQuery } from '../context/useQuery'
import { useEngine } from '../context/useEngine'
import { useExecution } from '../context/useExecution'

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function RunQueryButton(): React.ReactElement {
  const { datalogText } = useQuery()
  const { selectedEngine } = useEngine()
  const { executionStatus, submitQuery, cancelQuery } = useExecution()

  const isRunning = executionStatus === 'running'
  const canRun = !!selectedEngine && !!datalogText.trim()

  const handleRun = () => {
    if (!canRun || isRunning) return
    submitQuery(datalogText, selectedEngine!)
  }

  const handleCancel = () => {
    cancelQuery()
  }

  if (isRunning) {
    return (
      <div className="run-query-controls" data-testid="run-query-controls">
        <button
          className="run-query-btn run-query-btn--running"
          disabled
          aria-label="Query is running"
          data-testid="run-query-btn-running"
        >
          <span className="run-query-spinner" aria-hidden="true" />
          Running…
        </button>
        <button
          className="run-query-btn run-query-btn--cancel"
          onClick={handleCancel}
          aria-label="Cancel running query"
          data-testid="cancel-query-btn"
        >
          ✕ Cancel
        </button>
      </div>
    )
  }

  return (
    <div className="run-query-controls" data-testid="run-query-controls">
      <button
        className={`run-query-btn run-query-btn--run${!canRun ? ' run-query-btn--disabled' : ''}`}
        onClick={handleRun}
        disabled={!canRun}
        aria-label="Run query"
        data-testid="run-query-btn"
        title={!selectedEngine ? 'Select an engine first' : !datalogText.trim() ? 'Enter a query first' : 'Run query (Ctrl+Enter)'}
      >
        ▶ Run Query
      </button>
    </div>
  )
}

export default RunQueryButton
