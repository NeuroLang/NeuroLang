import React from 'react'
import { EngineProvider } from './context/EngineContext'
import { QueryProvider } from './context/QueryContext'
import Layout from './components/Layout'

function App(): React.ReactElement {
  return (
    <EngineProvider>
      <QueryProvider>
        <Layout />
      </QueryProvider>
    </EngineProvider>
  )
}

export default App
