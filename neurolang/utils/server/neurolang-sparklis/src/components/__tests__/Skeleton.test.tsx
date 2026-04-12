/**
 * Skeleton.test.tsx
 *
 * Tests for skeleton loading placeholder components.
 */
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import {
  Skeleton,
  PredicateBrowserSkeleton,
  ResultsTableSkeleton,
} from '../Skeleton'

describe('Skeleton', () => {
  it('renders a skeleton div', () => {
    const { container } = render(<Skeleton />)
    expect(container.querySelector('.skeleton')).toBeInTheDocument()
  })

  it('applies custom class', () => {
    const { container } = render(<Skeleton className="my-custom" />)
    const el = container.querySelector('.skeleton')
    expect(el).toHaveClass('my-custom')
  })

  it('applies custom width and height via inline style', () => {
    const { container } = render(<Skeleton width="100px" height="20px" />)
    const el = container.querySelector('.skeleton') as HTMLElement
    expect(el.style.width).toBe('100px')
    expect(el.style.height).toBe('20px')
  })

  it('is hidden from screen readers via aria-hidden', () => {
    const { container } = render(<Skeleton />)
    const el = container.querySelector('.skeleton')
    expect(el).toHaveAttribute('aria-hidden', 'true')
  })
})

describe('PredicateBrowserSkeleton', () => {
  it('renders the predicate browser skeleton', () => {
    render(<PredicateBrowserSkeleton />)
    expect(
      screen.getByTestId('predicate-browser-skeleton')
    ).toBeInTheDocument()
  })

  it('is marked as busy for accessibility', () => {
    render(<PredicateBrowserSkeleton />)
    expect(screen.getByTestId('predicate-browser-skeleton')).toHaveAttribute(
      'aria-busy',
      'true'
    )
  })

  it('has an accessible label', () => {
    render(<PredicateBrowserSkeleton />)
    expect(screen.getByLabelText('Loading predicates')).toBeInTheDocument()
  })

  it('renders multiple skeleton items', () => {
    const { container } = render(<PredicateBrowserSkeleton />)
    const skeletons = container.querySelectorAll('.skeleton')
    expect(skeletons.length).toBeGreaterThan(3)
  })
})

describe('ResultsTableSkeleton', () => {
  it('renders the results table skeleton', () => {
    render(<ResultsTableSkeleton />)
    expect(
      screen.getByTestId('results-table-skeleton')
    ).toBeInTheDocument()
  })

  it('is marked as busy for accessibility', () => {
    render(<ResultsTableSkeleton />)
    expect(screen.getByTestId('results-table-skeleton')).toHaveAttribute(
      'aria-busy',
      'true'
    )
  })

  it('has an accessible label', () => {
    render(<ResultsTableSkeleton />)
    expect(screen.getByLabelText('Loading results')).toBeInTheDocument()
  })

  it('renders header and row skeletons', () => {
    const { container } = render(<ResultsTableSkeleton />)
    const headerCells = container.querySelectorAll('.skeleton--table-header-cell')
    const dataCells = container.querySelectorAll('.skeleton--table-cell')
    expect(headerCells.length).toBeGreaterThan(0)
    expect(dataCells.length).toBeGreaterThan(0)
  })
})
