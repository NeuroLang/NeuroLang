import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import App from '../App'

describe('App', () => {
  it('renders without crashing', () => {
    render(<App />)
    expect(screen.getByText('NeuroLang Sparklis')).toBeInTheDocument()
  })

  it('renders the navbar with title', () => {
    render(<App />)
    const navbar = screen.getByRole('banner')
    expect(navbar).toBeInTheDocument()
    expect(screen.getByText('NeuroLang Sparklis')).toBeInTheDocument()
  })

  it('renders sidebar with engine section', () => {
    render(<App />)
    expect(screen.getByText('Engines')).toBeInTheDocument()
    expect(screen.getByText('Select an engine to begin')).toBeInTheDocument()
  })

  it('renders sidebar with predicates section', () => {
    render(<App />)
    expect(screen.getByText('Predicates')).toBeInTheDocument()
    expect(screen.getByText('Engine not selected')).toBeInTheDocument()
  })

  it('renders main content area with welcome message', () => {
    render(<App />)
    expect(screen.getByRole('main')).toBeInTheDocument()
    expect(screen.getByText(/Welcome to NeuroLang Sparklis/i)).toBeInTheDocument()
  })
})
