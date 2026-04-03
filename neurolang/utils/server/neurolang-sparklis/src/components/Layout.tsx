import React, { useState } from 'react'
import EngineSelector from './EngineSelector'
import PredicateBrowser from './PredicateBrowser'
import VisualQueryBuilder from './VisualQueryBuilder'
import { useEngine } from '../context/useEngine'
import { useQuery } from '../context/useQuery'
import { type SchemaSymbol } from './PredicateBrowser'

function MainContent(): React.ReactElement {
  const { selectedEngine } = useEngine()

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
