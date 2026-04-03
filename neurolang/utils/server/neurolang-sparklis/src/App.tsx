import React from 'react'
import { EngineProvider } from './context/EngineContext'
import Layout from './components/Layout'

function App(): React.ReactElement {
  return (
    <EngineProvider>
      <Layout />
    </EngineProvider>
  )
}

export default App
