/**
 * QueryContext.tsx
 *
 * React context that exposes a QueryModel instance and its current state to
 * the component tree.  Using a context avoids deep prop-drilling between the
 * PredicateBrowser (sidebar) and the VisualQueryBuilder (main content).
 */
import React, { createContext, useCallback, useReducer, useState } from 'react'
import {
  QueryModel,
  QueryState,
  computeSharedVariables,
} from '../models/QueryModel'

// ---------------------------------------------------------------------------
// Context value type
// ---------------------------------------------------------------------------

export interface QueryContextValue {
  /** The live QueryModel instance (stable reference). */
  model: QueryModel
  /** Reactive snapshot of the current query state. */
  state: QueryState
  /** Map from variable name → colour for shared variables. */
  sharedVarColors: Map<string, string>
  /** Whether undo is available. */
  canUndo: boolean
  /** Whether redo is available. */
  canRedo: boolean
  /** Trigger a re-render after mutating the model. */
  refresh: () => void
  /** Convenience: undo and refresh. */
  undo: () => void
  /** Convenience: redo and refresh. */
  redo: () => void
  /**
   * The raw Datalog text currently shown in the code editor.
   * Kept in sync: visual builder mutations update this; direct code edits
   * also update this (handled by the consumer of the context).
   */
  datalogText: string
  /** Update the raw Datalog text (called by the CodeEditor when user types). */
  setDatalogText: (text: string) => void
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const QueryContext = createContext<QueryContextValue | null>(null)

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/** Singleton model – created once per provider mount. */
function createModel(): QueryModel {
  return new QueryModel()
}

export function QueryProvider({
  children,
}: {
  children: React.ReactNode
}): React.ReactElement {
  // We use useReducer as a simple "force update" mechanism.  Every time the
  // model is mutated we increment a counter to trigger a re-render.
  const [, forceUpdate] = useReducer((n: number) => n + 1, 0)

  // The model is created once and shared for the lifetime of this provider.
  const [model] = React.useState<QueryModel>(createModel)

  // Raw Datalog text for the code editor – kept in sync with the model.
  const [datalogText, setDatalogText] = useState<string>('')

  const refresh = useCallback(() => {
    // After each model mutation, re-serialize to Datalog so the code editor
    // reflects the visual builder's state.
    setDatalogText(model.toDatalog())
    forceUpdate()
  }, [model])

  const undo = useCallback(() => {
    model.undo()
    setDatalogText(model.toDatalog())
    forceUpdate()
  }, [model])

  const redo = useCallback(() => {
    model.redo()
    setDatalogText(model.toDatalog())
    forceUpdate()
  }, [model])

  const state = model.state
  const sharedVarColors = computeSharedVariables(state.predicates)

  const value: QueryContextValue = {
    model,
    state,
    sharedVarColors,
    canUndo: model.canUndo,
    canRedo: model.canRedo,
    refresh,
    undo,
    redo,
    datalogText,
    setDatalogText,
  }

  return (
    <QueryContext.Provider value={value}>{children}</QueryContext.Provider>
  )
}

export default QueryContext
