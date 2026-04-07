import React from 'react'
import { EngineProvider } from './context/EngineContext'
import { QueryProvider } from './context/QueryContext'
import { SchemaProvider } from './context/SchemaContext'
import { ExecutionProvider } from './context/ExecutionContext'
import { BrainOverlayProvider } from './context/BrainOverlayContext'
import Layout from './components/Layout'

function App(): React.ReactElement {
  return (
    <EngineProvider>
      <SchemaProvider>
        <QueryProvider>
          <ExecutionProvider>
            <BrainOverlayProvider>
              <Layout />
            </BrainOverlayProvider>
          </ExecutionProvider>
        </QueryProvider>
      </SchemaProvider>
    </EngineProvider>
  )
}

export default App
