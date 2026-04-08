import React from 'react'
import { EngineProvider } from './context/EngineContext'
import { QueryProvider } from './context/QueryContext'
import { SchemaProvider } from './context/SchemaContext'
import { ExecutionProvider } from './context/ExecutionContext'
import { BrainOverlayProvider } from './context/BrainOverlayContext'
import { QueryHistoryProvider } from './context/QueryHistoryContext'
import { ConnectionProvider } from './context/ConnectionContext'
import ErrorBoundary from './components/ErrorBoundary'
import Layout from './components/Layout'
import QueryHistoryRecorder from './components/QueryHistoryRecorder'
import PermalinkLoader from './components/PermalinkLoader'

function App(): React.ReactElement {
  return (
    <ErrorBoundary>
      <ConnectionProvider>
        <EngineProvider>
          <SchemaProvider>
            <QueryProvider>
              <ExecutionProvider>
                <BrainOverlayProvider>
                  <QueryHistoryProvider>
                    <PermalinkLoader />
                    <QueryHistoryRecorder />
                    <Layout />
                  </QueryHistoryProvider>
                </BrainOverlayProvider>
              </ExecutionProvider>
            </QueryProvider>
          </SchemaProvider>
        </EngineProvider>
      </ConnectionProvider>
    </ErrorBoundary>
  )
}

export default App
