/**
 * Skeleton.tsx
 *
 * Reusable skeleton loading placeholder components for use while data is being
 * fetched.  Each component mirrors the rough shape of the real content.
 */
import React from 'react'

// ---------------------------------------------------------------------------
// Generic skeleton line/block
// ---------------------------------------------------------------------------

interface SkeletonProps {
  className?: string
  width?: string
  height?: string
}

export function Skeleton({ className, width, height }: SkeletonProps): React.ReactElement {
  return (
    <div
      className={`skeleton ${className ?? ''}`}
      style={{ width, height }}
      aria-hidden="true"
    />
  )
}

// ---------------------------------------------------------------------------
// PredicateBrowserSkeleton
// ---------------------------------------------------------------------------

/** Skeleton for the predicate browser panel (3 groups of items). */
export function PredicateBrowserSkeleton(): React.ReactElement {
  return (
    <div
      className="predicate-browser-skeleton"
      aria-busy="true"
      aria-label="Loading predicates"
      data-testid="predicate-browser-skeleton"
    >
      {/* Search bar placeholder */}
      <Skeleton className="skeleton--predicate-search" height="28px" />

      {/* Three category groups */}
      {[1, 2, 3].map((group) => (
        <div key={group} className="skeleton--predicate-group">
          <Skeleton className="skeleton--predicate-group-header" height="22px" />
          {[1, 2, 3].map((item) => (
            <Skeleton key={item} className="skeleton--predicate-item" height="18px" />
          ))}
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// ResultsTableSkeleton
// ---------------------------------------------------------------------------

/** Skeleton for the results table while a query is running. */
export function ResultsTableSkeleton(): React.ReactElement {
  return (
    <div
      className="results-table-skeleton"
      aria-busy="true"
      aria-label="Loading results"
      data-testid="results-table-skeleton"
    >
      {/* Header row */}
      <div className="skeleton--table-header-row">
        {[1, 2, 3, 4].map((col) => (
          <Skeleton key={col} className="skeleton--table-header-cell" height="20px" />
        ))}
      </div>
      {/* Data rows */}
      {[1, 2, 3, 4, 5].map((row) => (
        <div key={row} className="skeleton--table-row">
          {[1, 2, 3, 4].map((col) => (
            <Skeleton key={col} className="skeleton--table-cell" height="16px" />
          ))}
        </div>
      ))}
    </div>
  )
}
