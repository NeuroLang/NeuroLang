/**
 * CodeEditor.tsx
 *
 * CodeMirror 6-based code editor component for writing Datalog queries.
 *
 * Features:
 *   - Datalog syntax highlighting (keywords, predicates, operators, variables,
 *     strings, numbers, comments)
 *   - Controlled: value syncs with a shared query state via QueryContext
 *   - Editable: user can type Datalog directly
 *   - Ctrl+Enter / Cmd+Enter fires the onSubmit callback
 *   - Error line highlighting: highlights the error line when errorLine is set
 */
import React, { useEffect, useRef, useCallback } from 'react'
import { EditorState } from '@codemirror/state'
import { EditorView, keymap, Decoration, DecorationSet } from '@codemirror/view'
import { StateEffect, StateField } from '@codemirror/state'
import { datalogLanguage } from './datalogLanguage'

// ---------------------------------------------------------------------------
// Error highlight state effect & field
// ---------------------------------------------------------------------------

/** Effect to set (or clear) the error highlight on a specific line. */
const setErrorLineEffect = StateEffect.define<number | null>()

/** State field that holds the error line decorations. */
const errorLineField = StateField.define<DecorationSet>({
  create() {
    return Decoration.none
  },
  update(decorations, tr) {
    // Map existing decorations through document changes
    decorations = decorations.map(tr.changes)

    for (const effect of tr.effects) {
      if (effect.is(setErrorLineEffect)) {
        const lineNum = effect.value
        if (lineNum === null) {
          decorations = Decoration.none
        } else {
          try {
            const line = tr.state.doc.line(lineNum)
            const decoration = Decoration.line({
              class: 'cm-error-line',
            })
            decorations = Decoration.set([decoration.range(line.from)])
          } catch {
            decorations = Decoration.none
          }
        }
      }
    }
    return decorations
  },
  provide: (field) => EditorView.decorations.from(field),
})

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface CodeEditorProps {
  /** Current Datalog text displayed in the editor. */
  value: string
  /** Called when the editor content changes (user typing). */
  onChange?: (value: string) => void
  /** Called when Ctrl+Enter / Cmd+Enter is pressed. */
  onSubmit?: () => void
  /** Optional CSS class added to the wrapper div. */
  className?: string
  /** Whether the editor is read-only. Defaults to false. */
  readOnly?: boolean
  /**
   * 1-based line number to highlight as an error.
   * Set to null/undefined to clear the highlight.
   */
  errorLine?: number | null
}

// ---------------------------------------------------------------------------
// Helpers – build EditorView extensions
// ---------------------------------------------------------------------------

function buildExtensions(
  onChangeRef: React.MutableRefObject<((v: string) => void) | undefined>,
  onSubmitRef: React.MutableRefObject<(() => void) | undefined>,
  readOnly: boolean,
  isExternalUpdate: React.MutableRefObject<boolean>,
) {
  return [
    // Datalog syntax highlighting
    ...datalogLanguage(),

    // Error line highlight field
    errorLineField,

    // Default key bindings (history, etc.) – but we add our own submit binding first
    keymap.of([
      {
        key: 'Mod-Enter',
        run: () => {
          onSubmitRef.current?.()
          return true
        },
      },
    ]),

    // Update listener – calls onChange only for user-initiated doc changes.
    // External value prop updates set isExternalUpdate.current = true before
    // dispatching, so we skip calling onChange for those to prevent feedback
    // loops (which would clear the undo/redo stack via debounced model resets).
    EditorView.updateListener.of((update) => {
      if (update.docChanged && !isExternalUpdate.current) {
        onChangeRef.current?.(update.state.doc.toString())
      }
    }),

    // Read-only flag
    EditorState.readOnly.of(readOnly),

    // Theming: use a clean, light editor that fits the app design
    EditorView.theme({
      '&': {
        fontSize: '0.875rem',
        fontFamily:
          "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace",
        height: '100%',
        backgroundColor: '#f8f9fc',
        border: 'none',
      },
      '.cm-editor': {
        height: '100%',
      },
      '.cm-focused': {
        outline: 'none',
      },
      '.cm-content': {
        padding: '0.75rem',
        minHeight: '6rem',
        caretColor: '#1d4ed8',
        lineHeight: '1.6',
      },
      '.cm-line': {
        padding: '0',
      },
      '.cm-cursor': {
        borderLeftColor: '#1d4ed8',
      },
      '.cm-selectionBackground, &.cm-focused .cm-selectionBackground': {
        backgroundColor: '#dbeafe',
      },
      // Error line highlight
      '.cm-error-line': {
        backgroundColor: '#fee2e2',
        borderLeft: '3px solid #ef4444',
        paddingLeft: '0.5rem',
      },
    }),
  ]
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function CodeEditor({
  value,
  onChange,
  onSubmit,
  className,
  readOnly = false,
  errorLine,
}: CodeEditorProps): React.ReactElement {
  const containerRef = useRef<HTMLDivElement>(null)
  const viewRef = useRef<EditorView | null>(null)

  // Keep the latest callbacks in refs so extensions don't become stale
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange
  const onSubmitRef = useRef(onSubmit)
  onSubmitRef.current = onSubmit

  // Track whether the view is currently being updated from outside
  // to prevent feedback loops
  const isExternalUpdate = useRef(false)

  // ---------------------------------------------------------------------------
  // Mount / unmount
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (!containerRef.current) return

    const state = EditorState.create({
      doc: value,
      extensions: buildExtensions(onChangeRef, onSubmitRef, readOnly, isExternalUpdate),
    })

    const view = new EditorView({
      state,
      parent: containerRef.current,
    })

    viewRef.current = view

    return () => {
      view.destroy()
      viewRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Only on mount

  // ---------------------------------------------------------------------------
  // Sync value changes from outside into the editor
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const view = viewRef.current
    if (!view) return

    const currentDoc = view.state.doc.toString()
    if (currentDoc === value) return // Nothing to do

    // Replace the entire document without firing the onChange listener
    isExternalUpdate.current = true
    view.dispatch({
      changes: {
        from: 0,
        to: currentDoc.length,
        insert: value,
      },
    })
    isExternalUpdate.current = false
  }, [value])

  // ---------------------------------------------------------------------------
  // Sync readOnly changes
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const view = viewRef.current
    if (!view) return

    view.dispatch({
      effects: [],
    })
  }, [readOnly])

  // ---------------------------------------------------------------------------
  // Sync errorLine into the editor decorations
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const view = viewRef.current
    if (!view) return

    view.dispatch({
      effects: setErrorLineEffect.of(errorLine ?? null),
    })
  }, [errorLine])

  // ---------------------------------------------------------------------------
  // Keyboard shortcut handler on the container div (fallback for non-CM events)
  // ---------------------------------------------------------------------------
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      const isMod = e.ctrlKey || e.metaKey
      if (isMod && e.key === 'Enter') {
        e.preventDefault()
        onSubmitRef.current?.()
      }
    },
    [],
  )

  return (
    <div
      ref={containerRef}
      className={`code-editor-wrapper${className ? ` ${className}` : ''}`}
      data-testid="code-editor"
      onKeyDown={handleKeyDown}
      role="textbox"
      aria-multiline="true"
      aria-label="Datalog code editor"
    />
  )
}

export default CodeEditor
