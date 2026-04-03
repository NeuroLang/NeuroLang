/**
 * QueryContext.tsx
 *
 * React context that exposes a QueryModel instance and its current state to
 * the component tree.  Using a context avoids deep prop-drilling between the
 * PredicateBrowser (sidebar) and the VisualQueryBuilder (main content).
 *
 * Bidirectional sync:
 *   - Visual builder mutations → serialize to Datalog → update code editor
 *   - Code editor changes → debounce 500ms → parse Datalog → if success,
 *     update visual builder; if fail, set isSynced=false (desync indicator)
 */
import React, {
  createContext,
  useCallback,
  useEffect,
  useReducer,
  useRef,
  useState,
} from 'react'
import {
  QueryModel,
  QueryState,
  computeSharedVariables,
  parseDatalog,
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
  /**
   * Update the raw Datalog text (called by the CodeEditor when user types).
   * Triggers debounced parsing to attempt syncing the visual builder.
   */
  setDatalogText: (text: string) => void
  /**
   * Whether the visual builder and the code editor are in sync.
   * Set to false when the editor text cannot be parsed back to a QueryModel.
   * Set to true when:
   *   - The visual builder updates the editor (always valid)
   *   - The editor text is successfully parsed into a QueryModel
   *   - The editor is empty (cleared)
   */
  isSynced: boolean
}

// ---------------------------------------------------------------------------
// Debounce delay for code-to-builder parsing
// ---------------------------------------------------------------------------

const PARSE_DEBOUNCE_MS = 500

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
  const [datalogText, _setDatalogText] = useState<string>('')

  // Whether the visual builder and code editor are in sync.
  const [isSynced, setIsSynced] = useState<boolean>(true)

  // Ref to track whether the current datalogText update originates from the
  // visual builder (to avoid infinite update loops).
  const isBuilderUpdate = useRef<boolean>(false)

  // Debounce timer ref for code-to-builder parsing.
  const parseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ---------------------------------------------------------------------------
  // setDatalogText – called by the CodeEditor when the user types
  // ---------------------------------------------------------------------------
  const setDatalogText = useCallback(
    (text: string) => {
      // Always update the displayed text immediately.
      _setDatalogText(text)

      // If this change was triggered by the visual builder, don't re-parse.
      if (isBuilderUpdate.current) return

      // Cancel any pending parse timer.
      if (parseTimerRef.current !== null) {
        clearTimeout(parseTimerRef.current)
      }

      // Schedule a debounced parse attempt.
      parseTimerRef.current = setTimeout(() => {
        parseTimerRef.current = null

        // Empty text: reset model and mark as synced.
        if (!text.trim()) {
          model.reset()
          setIsSynced(true)
          forceUpdate()
          return
        }

        const parsed = parseDatalog(text)
        if (parsed !== null) {
          // Parsing succeeded → update the visual builder.
          model.reset(parsed)
          setIsSynced(true)
          forceUpdate()
        } else {
          // Parsing failed → show desync indicator.
          setIsSynced(false)
        }
      }, PARSE_DEBOUNCE_MS)
    },
    [model, forceUpdate],
  )

  // Clean up any pending timer on unmount.
  useEffect(() => {
    return () => {
      if (parseTimerRef.current !== null) {
        clearTimeout(parseTimerRef.current)
      }
    }
  }, [])

  // ---------------------------------------------------------------------------
  // refresh – called after visual builder mutations
  // ---------------------------------------------------------------------------
  const refresh = useCallback(() => {
    // Mark that the text update comes from the builder (not user typing).
    isBuilderUpdate.current = true
    const newText = model.toDatalog()
    _setDatalogText(newText)
    setIsSynced(true)
    isBuilderUpdate.current = false
    forceUpdate()
  }, [model])

  const undo = useCallback(() => {
    model.undo()
    isBuilderUpdate.current = true
    _setDatalogText(model.toDatalog())
    setIsSynced(true)
    isBuilderUpdate.current = false
    forceUpdate()
  }, [model])

  const redo = useCallback(() => {
    model.redo()
    isBuilderUpdate.current = true
    _setDatalogText(model.toDatalog())
    setIsSynced(true)
    isBuilderUpdate.current = false
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
    isSynced,
  }

  return (
    <QueryContext.Provider value={value}>{children}</QueryContext.Provider>
  )
}

export default QueryContext
