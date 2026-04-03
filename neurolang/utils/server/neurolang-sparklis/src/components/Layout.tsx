import React, { useCallback, useState } from 'react'
import EngineSelector from './EngineSelector'
import PredicateBrowser from './PredicateBrowser'
import VisualQueryBuilder from './VisualQueryBuilder'
import SuggestionsPanel from './SuggestionsPanel'
import { useEngine } from '../context/useEngine'
import { useQuery } from '../context/useQuery'
import { useSchema } from '../context/useSchema'
import { type SchemaSymbol } from './PredicateBrowser'

function MainContent(): React.ReactElement {
  const { selectedEngine } = useEngine()
  const { model, refresh } = useQuery()
  const { lookupSymbol } = useSchema()

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
      <VisualQueryBuilder />
      <SuggestionsPanel onSuggestionSelect={handleSuggestionSelect} />
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
