/**
 * QueryHistoryRecorder.tsx
 *
 * An invisible component that watches ExecutionContext for completed queries
 * (status === 'done') and records them into QueryHistoryContext.
 *
 * Placed inside the provider tree in App.tsx so it can access both contexts.
 */
import { useEffect, useRef } from 'react'
import { useExecution } from '../context/useExecution'
import { useQueryHistory } from '../context/useQueryHistory'

function QueryHistoryRecorder(): null {
  const { executionStatus, queryResults, lastQuery, lastEngine } = useExecution()
  const { addEntry } = useQueryHistory()

  /**
   * Track the previously-seen status to fire addEntry exactly once
   * on the transition from 'running' → 'done'.
   */
  const prevStatusRef = useRef(executionStatus)

  useEffect(() => {
    const prevStatus = prevStatusRef.current
    prevStatusRef.current = executionStatus

    if (prevStatus === 'running' && executionStatus === 'done') {
      if (!lastQuery || !lastEngine) return

      // Summarise the results (count rows across all result symbols).
      let rowCount = 0
      if (queryResults) {
        for (const symbolData of Object.values(queryResults)) {
          const data = symbolData as { values?: unknown[] } | null
          if (data && Array.isArray(data.values)) {
            rowCount += data.values.length
          }
        }
      }
      const resultSummary = rowCount === 1 ? '1 row' : `${rowCount} rows`

      addEntry({
        query: lastQuery,
        engine: lastEngine,
        resultSummary,
      })
    }
  }, [executionStatus, lastQuery, lastEngine, queryResults, addEntry])

  return null
}

export default QueryHistoryRecorder
