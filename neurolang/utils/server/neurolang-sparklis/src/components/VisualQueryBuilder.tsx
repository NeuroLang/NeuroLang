/**
 * VisualQueryBuilder.tsx
 *
 * Sparklis-inspired visual query construction UI.
 *
 * Renders the current QueryModel state as a human-readable, sentence-like
 * display.  Variables that are shared across predicates are highlighted with
 * matching colours.  Each predicate block has a remove button.  Undo / redo
 * buttons allow stepping through query history.
 */
import React from 'react'
import { useQuery } from '../context/useQuery'
import {
  serializeToDatalog,
  type PredicateInstance,
  type PredicateParam,
} from '../models/QueryModel'

// ---------------------------------------------------------------------------
// VariableToken sub-component
// ---------------------------------------------------------------------------

interface VariableTokenProps {
  varName: string
  color?: string
}

function VariableToken({ varName, color }: VariableTokenProps): React.ReactElement {
  const style: React.CSSProperties = color
    ? {
        color,
        borderColor: color,
        backgroundColor: `${color}1a`, // 10% opacity background tint
      }
    : {}

  return (
    <span
      className={`vqb-var-token${color ? ' vqb-var-token--shared' : ''}`}
      style={style}
      title={`Variable: ${varName}`}
    >
      {varName}
    </span>
  )
}

// ---------------------------------------------------------------------------
// PredicateBlock sub-component
// ---------------------------------------------------------------------------

interface PredicateBlockProps {
  instance: PredicateInstance
  sharedVarColors: Map<string, string>
  onRemove: (id: string) => void
}

function PredicateBlock({
  instance,
  sharedVarColors,
  onRemove,
}: PredicateBlockProps): React.ReactElement {
  return (
    <span className="vqb-predicate-block">
      <span className="vqb-predicate-name">{instance.name}</span>
      <span className="vqb-paren">(</span>
      {instance.params.map((param: PredicateParam, idx: number) => (
        <React.Fragment key={param.position}>
          {idx > 0 && <span className="vqb-comma">, </span>}
          <VariableToken
            varName={param.varName}
            color={sharedVarColors.get(param.varName)}
          />
        </React.Fragment>
      ))}
      <span className="vqb-paren">)</span>
      <button
        className="vqb-remove-btn"
        onClick={() => onRemove(instance.id)}
        aria-label={`Remove ${instance.name} from query`}
        title={`Remove ${instance.name}`}
      >
        ×
      </button>
    </span>
  )
}

// ---------------------------------------------------------------------------
// VisualQueryBuilder
// ---------------------------------------------------------------------------

export interface VisualQueryBuilderProps {
  className?: string
}

function VisualQueryBuilder({
  className,
}: VisualQueryBuilderProps): React.ReactElement {
  const { state, sharedVarColors, canUndo, canRedo, undo, redo, model, refresh } =
    useQuery()

  const { predicates } = state
  const hasPredicates = predicates.length > 0

  function handleRemove(id: string): void {
    model.removePredicate(id)
    refresh()
  }

  return (
    <div className={`vqb-container${className ? ` ${className}` : ''}`}>
      {/* Toolbar */}
      <div className="vqb-toolbar">
        <span className="vqb-toolbar-label">Visual Query Builder</span>
        <div className="vqb-toolbar-actions">
          <button
            className="vqb-btn vqb-btn--undo"
            onClick={undo}
            disabled={!canUndo}
            aria-label="Undo last query step"
            title="Undo"
          >
            ↩ Undo
          </button>
          <button
            className="vqb-btn vqb-btn--redo"
            onClick={redo}
            disabled={!canRedo}
            aria-label="Redo last undone query step"
            title="Redo"
          >
            Redo ↪
          </button>
        </div>
      </div>

      {/* Readable sentence rendering */}
      <div className="vqb-sentence" aria-label="Query sentence">
        {!hasPredicates ? (
          <span className="vqb-placeholder">
            Click a predicate in the sidebar to start building your query…
          </span>
        ) : (
          <>
            <span className="vqb-keyword">Find </span>
            {/* Render head variables with colour */}
            {((): React.ReactNode => {
              const headVars = computeHeadVars(predicates)
              return headVars.map((v, i) => (
                <React.Fragment key={v}>
                  {i > 0 && <span className="vqb-comma">, </span>}
                  <VariableToken varName={v} color={sharedVarColors.get(v)} />
                </React.Fragment>
              ))
            })()}
            <span className="vqb-keyword"> where </span>
            {predicates.map((pred, idx) => (
              <React.Fragment key={pred.id}>
                {idx > 0 && <span className="vqb-and"> and </span>}
                <PredicateBlock
                  instance={pred}
                  sharedVarColors={sharedVarColors}
                  onRemove={handleRemove}
                />
              </React.Fragment>
            ))}
          </>
        )}
      </div>

      {/* Datalog serialisation preview */}
      {hasPredicates && (
        <div className="vqb-datalog-preview">
          <span className="vqb-datalog-label">Datalog:</span>
          <code className="vqb-datalog-code">
            {serializeToDatalog(state)}
          </code>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Helper – collect head variables (mirrors the logic in QueryModel)
// ---------------------------------------------------------------------------

function computeHeadVars(predicates: PredicateInstance[]): string[] {
  const counts = new Map<string, number>()
  const order: string[] = []
  for (const pred of predicates) {
    for (const param of pred.params) {
      if (!counts.has(param.varName)) order.push(param.varName)
      counts.set(param.varName, (counts.get(param.varName) ?? 0) + 1)
    }
  }
  const shared = order.filter((v) => (counts.get(v) ?? 0) > 1)
  return shared.length > 0 ? shared : order
}

export default VisualQueryBuilder
