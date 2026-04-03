import React from 'react'
import { EngineProvider } from './context/EngineContext'
import { QueryProvider } from './context/QueryContext'
import { SchemaProvider } from './context/SchemaContext'
import Layout from './components/Layout'

function App(): React.ReactElement {
  return (
    <EngineProvider>
      <SchemaProvider>
        <QueryProvider>
          <Layout />
        </QueryProvider>
      </SchemaProvider>
    </EngineProvider>
  )
}

export default App
