/**
 * ErrorBoundary.tsx
 *
 * React error boundary that catches rendering errors anywhere in the component
 * tree and displays a friendly error message with a "Reload" button instead of
 * crashing the whole app.
 */
import React from 'react'

interface ErrorBoundaryState {
  hasError: boolean
  errorMessage: string | null
}

interface ErrorBoundaryProps {
  children: React.ReactNode
  /** Optional fallback to render instead of the default error UI. */
  fallback?: React.ReactNode
}

class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, errorMessage: null }
  }

  static getDerivedStateFromError(error: unknown): ErrorBoundaryState {
    const errorMessage =
      error instanceof Error ? error.message : String(error)
    return { hasError: true, errorMessage }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // Log the error for debugging purposes
    console.error('[ErrorBoundary] Caught rendering error:', error, info)
  }

  handleReload = (): void => {
    window.location.reload()
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="error-boundary" role="alert" data-testid="error-boundary">
          <div className="error-boundary-icon" aria-hidden="true">⚠️</div>
          <h1 className="error-boundary-title">Something went wrong</h1>
          <p className="error-boundary-message">
            An unexpected error occurred while rendering the application.
          </p>
          {this.state.errorMessage && (
            <pre className="error-boundary-details" data-testid="error-boundary-details">
              {this.state.errorMessage}
            </pre>
          )}
          <button
            className="error-boundary-reload-btn"
            onClick={this.handleReload}
            data-testid="error-boundary-reload"
          >
            Reload Page
          </button>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
