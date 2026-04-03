import React from 'react'

function App(): React.ReactElement {
  return (
    <div className="app-container">
      {/* Top Navbar */}
      <header className="navbar">
        <div className="navbar-brand">
          <span className="navbar-title">NeuroLang Sparklis</span>
        </div>
        <nav className="navbar-nav">
          <a href="#" className="nav-link">Documentation</a>
          <a href="#" className="nav-link">About</a>
        </nav>
      </header>

      {/* Main Layout */}
      <div className="main-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <h2 className="sidebar-section-title">Engines</h2>
            <p className="sidebar-placeholder">Select an engine to begin</p>
          </div>
          <div className="sidebar-section">
            <h2 className="sidebar-section-title">Predicates</h2>
            <p className="sidebar-placeholder">Engine not selected</p>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="main-content">
          <div className="content-placeholder">
            <h1>Welcome to NeuroLang Sparklis</h1>
            <p>Select an engine from the sidebar to get started.</p>
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
