import React, { useCallback, useEffect, useState } from 'react'
import { useEngine } from '../context/useEngine'
import { useQuery } from '../context/useQuery'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Example {
  id: string
  title: string
  shortTitle: string
  query: string
  description: string
}

interface ExamplesResponse {
  status: string
  data: Example[]
}

// ---------------------------------------------------------------------------
// ExampleItem sub-component
// ---------------------------------------------------------------------------

interface ExampleItemProps {
  example: Example
  onLoad: (query: string) => void
}

function ExampleItem({
  example,
  onLoad,
}: ExampleItemProps): React.ReactElement {
  const [descriptionOpen, setDescriptionOpen] = useState(false)

  const handleToggleDescription = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      setDescriptionOpen((prev) => !prev)
    },
    [],
  )

  return (
    <li className="example-item">
      <div className="example-item-row">
        <button
          className="example-item-title"
          onClick={() => onLoad(example.query)}
          title={example.title}
          aria-label={`Load example: ${example.shortTitle}`}
        >
          {example.shortTitle}
        </button>
        <button
          className="example-item-info-btn"
          onClick={handleToggleDescription}
          aria-expanded={descriptionOpen}
          aria-label={`${descriptionOpen ? 'Hide' : 'Show'} description for ${example.shortTitle}`}
          title={descriptionOpen ? 'Hide description' : 'Show description'}
        >
          {descriptionOpen ? '▼' : '▶'}
        </button>
      </div>
      {descriptionOpen && (
        <div className="example-item-description">{example.description}</div>
      )}
    </li>
  )
}

// ---------------------------------------------------------------------------
// ExampleQueries main component
// ---------------------------------------------------------------------------

function ExampleQueries(): React.ReactElement {
  const { selectedEngine } = useEngine()
  const { setDatalogText, refresh, model } = useQuery()

  const [examples, setExamples] = useState<Example[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [panelOpen, setPanelOpen] = useState(true)

  // Fetch examples when the engine changes
  useEffect(() => {
    if (!selectedEngine) {
      setExamples([])
      setLoading(false)
      setError(null)
      return
    }

    setLoading(true)
    setError(null)

    fetch(`/v2/examples/${selectedEngine}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(
            `Failed to load examples: ${res.status}`,
          )
        }
        return res.json() as Promise<ExamplesResponse>
      })
      .then((data) => {
        setExamples(Array.isArray(data.data) ? data.data : [])
        setLoading(false)
      })
      .catch((err: unknown) => {
        const message =
          err instanceof Error
            ? err.message
            : 'Failed to load examples'
        setError(message)
        setLoading(false)
      })
  }, [selectedEngine])

  const handleLoadExample = useCallback(
    (query: string) => {
      // Reset the visual builder and update the code editor
      model.reset()
      refresh()
      // Set the datalog text (triggers bidirectional sync parsing)
      setDatalogText(query)
    },
    [model, refresh, setDatalogText],
  )

  // Don't render anything if no engine is selected
  if (!selectedEngine) {
    return <div className="example-queries" />
  }

  return (
    <div className="example-queries">
      {/* Panel header / toggle */}
      <button
        className="example-queries-header"
        onClick={() => setPanelOpen((prev) => !prev)}
        aria-expanded={panelOpen}
        aria-label={panelOpen ? 'Collapse examples panel' : 'Expand examples panel'}
      >
        <span className="example-queries-title">Examples</span>
        <span
          className={`example-queries-chevron ${panelOpen ? 'example-queries-chevron--open' : ''}`}
        >
          ▶
        </span>
      </button>

      {panelOpen && (
        <div className="example-queries-body">
          {loading && (
            <p className="example-queries-loading">Loading examples...</p>
          )}
          {error && (
            <p className="example-queries-error">{error}</p>
          )}
          {!loading && !error && examples.length === 0 && (
            <p className="example-queries-empty">No examples available</p>
          )}
          {!loading && !error && examples.length > 0 && (
            <ul className="example-list">
              {examples.map((example) => (
                <ExampleItem
                  key={example.id}
                  example={example}
                  onLoad={handleLoadExample}
                />
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

export default ExampleQueries
