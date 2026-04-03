/**
 * VisualQueryBuilder.test.tsx
 *
 * Tests for the VisualQueryBuilder component.
 * Covers: rendering, undo/redo buttons, remove predicate, variable coloring.
 */
import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, beforeEach } from 'vitest'
import { QueryProvider } from '../../context/QueryContext'
import { useQuery } from '../../context/useQuery'
import VisualQueryBuilder from '../VisualQueryBuilder'
import { resetIdCounter } from '../../models/QueryModel'

// Helper: render VisualQueryBuilder wrapped in QueryProvider
function renderBuilder(): ReturnType<typeof render> {
  return render(
    <QueryProvider>
      <VisualQueryBuilder />
    </QueryProvider>,
  )
}

// Helper: wrapper component that lets tests add predicates via context
function BuilderWithAdder({
  predicateName,
  params,
}: {
  predicateName: string
  params: string[]
}): React.ReactElement {
  const { model, refresh } = useQuery()
  return (
    <div>
      <button
        data-testid="add-btn"
        onClick={() => {
          model.addPredicate(predicateName, params)
          refresh()
        }}
      >
        Add {predicateName}
      </button>
      <VisualQueryBuilder />
    </div>
  )
}

function renderWithAdder(
  predicateName: string,
  params: string[],
): ReturnType<typeof render> {
  return render(
    <QueryProvider>
      <BuilderWithAdder predicateName={predicateName} params={params} />
    </QueryProvider>,
  )
}

// Helper: wrapper that adds two predicates with a shared variable
function TwoPredicateBuilder(): React.ReactElement {
  const { model, refresh, state } = useQuery()
  return (
    <div>
      <button
        data-testid="add-peak"
        onClick={() => {
          model.addPredicate('PeakReported', ['x', 'y', 'z', 'study'])
          refresh()
        }}
      >
        Add PeakReported
      </button>
      <button
        data-testid="add-study"
        onClick={() => {
          const studyVar =
            state.predicates[0]?.params[3]?.varName ?? 's'
          model.addPredicate('Study', ['study'], { study: studyVar })
          refresh()
        }}
      >
        Add Study
      </button>
      <VisualQueryBuilder />
    </div>
  )
}

function renderTwoPredicates(): ReturnType<typeof render> {
  return render(
    <QueryProvider>
      <TwoPredicateBuilder />
    </QueryProvider>,
  )
}

beforeEach(() => {
  resetIdCounter()
})

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

describe('VisualQueryBuilder – rendering', () => {
  it('renders the toolbar label', () => {
    renderBuilder()
    expect(screen.getByText('Visual Query Builder')).toBeInTheDocument()
  })

  it('renders undo and redo buttons', () => {
    renderBuilder()
    expect(screen.getByRole('button', { name: 'Undo last query step' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Redo last undone query step' })).toBeInTheDocument()
  })

  it('shows placeholder when no predicates', () => {
    renderBuilder()
    expect(
      screen.getByText(/Click a predicate in the sidebar/i),
    ).toBeInTheDocument()
  })

  it('does not show Datalog preview when no predicates', () => {
    renderBuilder()
    expect(screen.queryByText('Datalog:')).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Adding predicates
// ---------------------------------------------------------------------------

describe('VisualQueryBuilder – adding predicates', () => {
  it('shows predicate name after adding', () => {
    renderWithAdder('Study', ['study'])
    fireEvent.click(screen.getByTestId('add-btn'))
    expect(screen.getByText('Study')).toBeInTheDocument()
  })

  it('renders "Find … where …" sentence structure', () => {
    renderWithAdder('Study', ['study'])
    fireEvent.click(screen.getByTestId('add-btn'))
    expect(screen.getByText('Find')).toBeInTheDocument()
    expect(screen.getByText('where')).toBeInTheDocument()
  })

  it('renders remove button for the predicate', () => {
    renderWithAdder('Study', ['study'])
    fireEvent.click(screen.getByTestId('add-btn'))
    expect(
      screen.getByRole('button', { name: /Remove Study/i }),
    ).toBeInTheDocument()
  })

  it('shows Datalog preview after adding a predicate', () => {
    renderWithAdder('Study', ['study'])
    fireEvent.click(screen.getByTestId('add-btn'))
    expect(screen.getByText('Datalog:')).toBeInTheDocument()
  })

  it('Datalog preview contains valid Datalog text', () => {
    renderWithAdder('Study', ['study'])
    fireEvent.click(screen.getByTestId('add-btn'))
    const code = screen.getByRole('code')
    expect(code.textContent).toMatch(/ans\(.+\) :- Study\(.+\)\.$/)
  })
})

// ---------------------------------------------------------------------------
// Removing predicates
// ---------------------------------------------------------------------------

describe('VisualQueryBuilder – removing predicates', () => {
  it('removes the predicate block when remove button is clicked', () => {
    renderWithAdder('Study', ['study'])
    fireEvent.click(screen.getByTestId('add-btn'))
    expect(screen.getByText('Study')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Remove Study/i }))
    expect(screen.queryByText('Study')).not.toBeInTheDocument()
  })

  it('shows placeholder again after all predicates are removed', () => {
    renderWithAdder('Study', ['study'])
    fireEvent.click(screen.getByTestId('add-btn'))
    fireEvent.click(screen.getByRole('button', { name: /Remove Study/i }))
    expect(
      screen.getByText(/Click a predicate in the sidebar/i),
    ).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Undo / Redo
// ---------------------------------------------------------------------------

describe('VisualQueryBuilder – undo/redo', () => {
  it('undo button is disabled when history is empty', () => {
    renderBuilder()
    const undoBtn = screen.getByRole('button', { name: 'Undo last query step' })
    expect(undoBtn).toBeDisabled()
  })

  it('redo button is disabled when redo stack is empty', () => {
    renderBuilder()
    const redoBtn = screen.getByRole('button', { name: 'Redo last undone query step' })
    expect(redoBtn).toBeDisabled()
  })

  it('undo button becomes enabled after adding a predicate', () => {
    renderWithAdder('Study', ['study'])
    fireEvent.click(screen.getByTestId('add-btn'))
    const undoBtn = screen.getByRole('button', { name: 'Undo last query step' })
    expect(undoBtn).not.toBeDisabled()
  })

  it('clicking undo removes the last predicate', () => {
    renderWithAdder('Study', ['study'])
    fireEvent.click(screen.getByTestId('add-btn'))
    expect(screen.getByText('Study')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Undo last query step' }))
    expect(screen.queryByText('Study')).not.toBeInTheDocument()
  })

  it('redo button becomes enabled after undo', () => {
    renderWithAdder('Study', ['study'])
    fireEvent.click(screen.getByTestId('add-btn'))
    fireEvent.click(screen.getByRole('button', { name: 'Undo last query step' }))
    const redoBtn = screen.getByRole('button', { name: 'Redo last undone query step' })
    expect(redoBtn).not.toBeDisabled()
  })

  it('clicking redo restores the undone predicate', () => {
    renderWithAdder('Study', ['study'])
    fireEvent.click(screen.getByTestId('add-btn'))
    fireEvent.click(screen.getByRole('button', { name: 'Undo last query step' }))
    fireEvent.click(screen.getByRole('button', { name: 'Redo last undone query step' }))
    expect(screen.getByText('Study')).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Variable colouring
// ---------------------------------------------------------------------------

describe('VisualQueryBuilder – variable coloring', () => {
  it('shared variables have the vqb-var-token--shared class', () => {
    renderTwoPredicates()
    fireEvent.click(screen.getByTestId('add-peak'))
    fireEvent.click(screen.getByTestId('add-study'))

    // The shared variable (study var) should appear with the shared class
    const sharedTokens = document.querySelectorAll('.vqb-var-token--shared')
    expect(sharedTokens.length).toBeGreaterThan(0)
  })

  it('non-shared variables do NOT have the shared class', () => {
    renderWithAdder('Study', ['study'])
    fireEvent.click(screen.getByTestId('add-btn'))

    // Single predicate → no shared vars
    const sharedTokens = document.querySelectorAll('.vqb-var-token--shared')
    expect(sharedTokens.length).toBe(0)
  })
})
