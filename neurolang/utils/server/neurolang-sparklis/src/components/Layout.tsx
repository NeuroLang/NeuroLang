import React, { useCallback, useState } from 'react'
import EngineSelector from './EngineSelector'
import ExampleQueries from './ExampleQueries'
import PredicateBrowser from './PredicateBrowser'
import VisualQueryBuilder from './VisualQueryBuilder'
import SuggestionsPanel from './SuggestionsPanel'
import CodeEditor from './CodeEditor'
import RunQueryButton from './RunQueryButton'
import ErrorDisplay from './ErrorDisplay'
import ResultsPanel from './ResultsPanel'
import BrainViewer from './BrainViewer'
import OverlayManager from './OverlayManager'
import QueryHistory from './QueryHistory'
import { useEngine } from '../context/useEngine'
import { useQuery } from '../context/useQuery'
import { useSchema } from '../context/useSchema'
import { useExecution } from '../context/useExecution'
import { type SchemaSymbol } from './PredicateBrowser'

function MainContent(): React.ReactElement {
  const { selectedEngine } = useEngine()
  const { model, refresh, datalogText, setDatalogText } = useQuery()
  const { lookupSymbol } = useSchema()
  const { submitQuery, executionStatus } = useExecution()

  // Track the error line for highlighting in the CodeEditor
  const [errorLine, setErrorLine] = useState<number | null>(null)

  const handleSuggestionSelect = useCallback(
    (suggestion: string) => {
      // If the suggestion matches a known predicate name, look it up in the
      // schema and add it with proper placeholder variables (same as the
      // predicate browser click handler).  Otherwise fall back to adding it
      // with an empty parameter list.
      const symbol = lookupSymbol(suggestion)
      if (symbol) {
        model.addPredicate(symbol.name, symbol.params)
      } else {
        model.addPredicate(suggestion, [])
      }
      refresh()
    },
    [model, refresh, lookupSymbol],
  )

  const handleEditorChange = useCallback(
    (text: string) => {
      // Update the shared Datalog text when the user types directly
      setDatalogText(text)
      // Clear error highlight when user edits
      setErrorLine(null)
    },
    [setDatalogText],
  )

  const handleSubmit = useCallback(() => {
    if (!selectedEngine || !datalogText.trim()) return
    setErrorLine(null)
    submitQuery(datalogText, selectedEngine)
  }, [datalogText, selectedEngine, submitQuery])

  const handleHighlightError = useCallback((line: number) => {
    setErrorLine(line)
  }, [])

  // Clear error line when a new run starts
  const isRunning = executionStatus === 'running'
  React.useEffect(() => {
    if (isRunning) {
      setErrorLine(null)
    }
  }, [isRunning])

  if (!selectedEngine) {
    return (
      <div className="content-placeholder">
        <h1>Welcome to NeuroLang Sparklis</h1>
        <p>Select an engine from the sidebar to get started.</p>
      </div>
    )
  }

  return (
    <div className="content-engine-selected">
      {/* Split view: Visual Query Builder + Code Editor side by side */}
      <div className="query-split-view">
        <div className="query-split-panel query-split-panel--builder">
          <VisualQueryBuilder />
        </div>
        <div className="query-split-panel query-split-panel--editor">
          <div className="code-editor-panel">
            <div className="code-editor-panel-header">
              <span className="code-editor-panel-label">Datalog Editor</span>
              <span className="code-editor-panel-hint">Ctrl+Enter to run</span>
            </div>
            <div className="code-editor-panel-body">
              <CodeEditor
                value={datalogText}
                onChange={handleEditorChange}
                onSubmit={handleSubmit}
                errorLine={errorLine}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Run Query Button and Error/Cancel Display */}
      <div className="execution-controls">
        <RunQueryButton />
        <ErrorDisplay onHighlightError={handleHighlightError} />
      </div>

      <SuggestionsPanel onSuggestionSelect={handleSuggestionSelect} />

      {/* Bottom section: Results table (left) + Brain Viewer (right) */}
      <div className="results-and-viewer">
        <div className="results-column">
          <ResultsPanel />
        </div>
        <div className="brain-viewer-column">
          <BrainViewer />
          <OverlayManager />
        </div>
      </div>
    </div>
  )
}

function SidebarPredicateBrowser(): React.ReactElement {
  const { model, refresh } = useQuery()

  function handleSelect(symbol: SchemaSymbol): void {
    model.addPredicate(symbol.name, symbol.params)
    refresh()
  }

  return <PredicateBrowser onSelect={handleSelect} />
}

function Layout(): React.ReactElement {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="app-container">
      {/* Top Navbar */}
      <header className="navbar" role="banner">
        <div className="navbar-brand">
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen((prev) => !prev)}
            aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            ☰
          </button>
          <span className="navbar-title">NeuroLang Sparklis</span>
        </div>
        <nav className="navbar-nav" aria-label="Main navigation">
          <a href="#" className="nav-link">
            Documentation
          </a>
          <a href="#" className="nav-link">
            About
          </a>
        </nav>
      </header>

      {/* Main Layout */}
      <div className="main-layout">
        {/* Sidebar */}
        <aside
          className={`sidebar ${sidebarOpen ? 'sidebar--open' : 'sidebar--collapsed'}`}
          aria-hidden={!sidebarOpen}
        >
          <div className="sidebar-section">
            <h2 className="sidebar-section-title">Engines</h2>
            <EngineSelector />
          </div>
          <div className="sidebar-section">
            <h2 className="sidebar-section-title">Predicates</h2>
            <SidebarPredicateBrowser />
          </div>
          <div className="sidebar-section">
            <ExampleQueries />
          </div>
          <div className="sidebar-section">
            <QueryHistory />
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="main-content">
          <MainContent />
        </main>
      </div>
    </div>
  )
}

export default Layout
