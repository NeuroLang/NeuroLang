/**
 * DatalogHelpTooltip.tsx
 *
 * A help icon (?) near the query builder that, when clicked, shows a popover
 * explaining basic Datalog syntax:
 *   - :- means "if"
 *   - & means "and"
 *   - ~ means "not"
 *   - PROB for probabilistic queries
 */
import React, { useState, useCallback, useRef, useEffect } from 'react'

const DATALOG_SYNTAX_ITEMS = [
  { syntax: ':-', meaning: 'means "if" (head :- body defines a rule)' },
  { syntax: '&', meaning: 'means "and" (conjunction in rule body)' },
  { syntax: '~', meaning: 'means "not" (negation of a predicate)' },
  { syntax: 'PROB', meaning: 'probabilistic query (returns probabilities)' },
  {
    syntax: 'ans(x) :- Pred(x)',
    meaning: 'basic query: "find x where Pred(x) holds"',
  },
]

function DatalogHelpTooltip(): React.ReactElement {
  const [open, setOpen] = useState(false)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const popoverRef = useRef<HTMLDivElement>(null)

  const toggle = useCallback((): void => {
    setOpen((prev) => !prev)
  }, [])

  // Close popover when clicking outside
  useEffect(() => {
    if (!open) return

    function handleClickOutside(event: MouseEvent): void {
      const target = event.target as Node
      if (
        buttonRef.current &&
        !buttonRef.current.contains(target) &&
        popoverRef.current &&
        !popoverRef.current.contains(target)
      ) {
        setOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [open])

  // Close on Escape
  useEffect(() => {
    if (!open) return

    function handleKeyDown(e: KeyboardEvent): void {
      if (e.key === 'Escape') {
        setOpen(false)
        buttonRef.current?.focus()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [open])

  return (
    <div className="datalog-help-container">
      <button
        ref={buttonRef}
        className="datalog-help-btn"
        onClick={toggle}
        aria-label="Datalog syntax help"
        aria-expanded={open}
        aria-haspopup="dialog"
        data-testid="datalog-help-btn"
        title="Datalog syntax help"
      >
        ?
      </button>

      {open && (
        <div
          ref={popoverRef}
          className="datalog-help-popover"
          role="dialog"
          aria-label="Datalog syntax reference"
          data-testid="datalog-help-popover"
        >
          <div className="datalog-help-popover-header">
            <span className="datalog-help-popover-title">Datalog Syntax</span>
            <button
              className="datalog-help-popover-close"
              onClick={() => setOpen(false)}
              aria-label="Close help"
              data-testid="datalog-help-close"
            >
              ×
            </button>
          </div>
          <dl className="datalog-help-list">
            {DATALOG_SYNTAX_ITEMS.map(({ syntax, meaning }) => (
              <div key={syntax} className="datalog-help-item">
                <dt className="datalog-help-syntax">
                  <code>{syntax}</code>
                </dt>
                <dd className="datalog-help-meaning">{meaning}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </div>
  )
}

export default DatalogHelpTooltip
