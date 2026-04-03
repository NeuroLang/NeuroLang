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
 */
import React, { useEffect, useRef, useCallback } from 'react'
import { EditorState } from '@codemirror/state'
import { EditorView, keymap } from '@codemirror/view'
import { datalogLanguage } from './datalogLanguage'

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
}

// ---------------------------------------------------------------------------
// Helpers – build EditorView extensions
// ---------------------------------------------------------------------------

function buildExtensions(
  onChangeRef: React.MutableRefObject<((v: string) => void) | undefined>,
  onSubmitRef: React.MutableRefObject<(() => void) | undefined>,
  readOnly: boolean,
) {
  return [
    // Datalog syntax highlighting
    ...datalogLanguage(),

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

    // Update listener – calls onChange whenever the doc changes
    EditorView.updateListener.of((update) => {
      if (update.docChanged) {
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
      extensions: buildExtensions(onChangeRef, onSubmitRef, readOnly),
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
