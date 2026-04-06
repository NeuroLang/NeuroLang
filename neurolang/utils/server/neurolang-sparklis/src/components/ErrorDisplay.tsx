/**
 * ErrorDisplay.tsx
 *
 * Displays query execution errors with:
 *   - The error name (type)
 *   - The error message
 *   - Optional line/column info (for parser errors)
 *
 * When line info is available, it fires an onHighlightError callback so the
 * parent (Layout) can highlight the error location in the CodeEditor.
 */
import React, { useEffect } from 'react'
import { useExecution } from '../context/useExecution'

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ErrorDisplayProps {
  /**
   * Called when a parser error with line info is detected.
   * The parent can use this to highlight the error in the CodeEditor.
   * @param line 1-based line number
   * @param col  1-based column number
   */
  onHighlightError?: (line: number, col: number) => void
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function ErrorDisplay({ onHighlightError }: ErrorDisplayProps): React.ReactElement | null {
  const { executionStatus, queryError } = useExecution()

  // Notify parent about line info so it can highlight the error in the editor.
  useEffect(() => {
    if (
      executionStatus === 'error' &&
      queryError?.line_info &&
      onHighlightError
    ) {
      onHighlightError(queryError.line_info.line, queryError.line_info.col)
    }
  }, [executionStatus, queryError, onHighlightError])

  if (executionStatus === 'cancelled') {
    return (
      <div
        className="error-display error-display--cancelled"
        data-testid="query-cancelled-message"
        role="status"
        aria-live="polite"
      >
        <span className="error-display-icon" aria-hidden="true">⊘</span>
        <span className="error-display-text">Query cancelled.</span>
      </div>
    )
  }

  if (executionStatus !== 'error' || !queryError) {
    return null
  }

  const { errorName, message, errorDoc, line_info } = queryError

  // Extract a short class name from the Python class string
  // e.g. "<class 'neurolang.exceptions.ParserError'>" → "ParserError"
  const shortName = errorName.replace(/.*'([^']+)'\s*>?/, '$1').split('.').pop() ?? errorName

  return (
    <div
      className="error-display error-display--error"
      data-testid="query-error-display"
      role="alert"
      aria-live="assertive"
    >
      <div className="error-display-header">
        <span className="error-display-icon" aria-hidden="true">⚠</span>
        <span
          className="error-display-name"
          data-testid="error-name"
          title={errorName}
        >
          {shortName}
        </span>
      </div>

      <p className="error-display-message" data-testid="error-message">
        {message}
      </p>

      {errorDoc && errorDoc !== message && (
        <p className="error-display-doc" data-testid="error-doc">
          {errorDoc}
        </p>
      )}

      {line_info && (
        <p
          className="error-display-location"
          data-testid="error-location"
        >
          Line {line_info.line}, column {line_info.col}
          {line_info.text ? `: ${line_info.text}` : ''}
        </p>
      )}
    </div>
  )
}

export default ErrorDisplay
