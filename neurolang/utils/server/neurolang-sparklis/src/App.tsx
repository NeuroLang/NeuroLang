import React from 'react'
import { EngineProvider } from './context/EngineContext'
import { QueryProvider } from './context/QueryContext'
import { SchemaProvider } from './context/SchemaContext'
import { ExecutionProvider } from './context/ExecutionContext'
import { BrainOverlayProvider } from './context/BrainOverlayContext'
import { QueryHistoryProvider } from './context/QueryHistoryContext'
import Layout from './components/Layout'
import QueryHistoryRecorder from './components/QueryHistoryRecorder'

function App(): React.ReactElement {
  return (
    <EngineProvider>
      <SchemaProvider>
        <QueryProvider>
          <ExecutionProvider>
            <BrainOverlayProvider>
              <QueryHistoryProvider>
                <QueryHistoryRecorder />
                <Layout />
              </QueryHistoryProvider>
            </BrainOverlayProvider>
          </ExecutionProvider>
        </QueryProvider>
      </SchemaProvider>
    </EngineProvider>
  )
}

export default App
