/**
 * QueryModel.ts
 *
 * Represents a Datalog query as a collection of predicate instances with
 * named variables. Supports:
 *   - Adding predicates with auto-generated variable names
 *   - Removing predicates
 *   - Sharing variables across predicates (joins)
 *   - Serialising to valid Datalog text
 *   - Undo / redo via an immutable history stack
 */

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/** A single parameter slot in a predicate instance */
export interface PredicateParam {
  /** Position of this parameter in the predicate's signature (0-indexed) */
  position: number
  /** Human-readable parameter name from the schema (e.g. "x", "study") */
  schemaName: string
  /** The variable name currently bound to this slot (e.g. "x", "y0") */
  varName: string
}

/** One instantiation of a predicate in the query */
export interface PredicateInstance {
  /** Unique id for this instance within the query */
  id: string
  /** Symbol name, e.g. "PeakReported" */
  name: string
  /** Ordered list of parameter slots */
  params: PredicateParam[]
}

/**
 * Immutable snapshot of a Datalog query.
 *
 * A QueryState is a plain-data object so it can be cheaply cloned for the
 * undo / redo history stack.
 */
export interface QueryState {
  /** Ordered list of predicates in the query body */
  predicates: PredicateInstance[]
  /**
   * Counter used to generate fresh variable suffixes.
   * Monotonically increasing across adds – never resets even on removal.
   */
  varCounter: number
}

// ---------------------------------------------------------------------------
// Variable-name generation
// ---------------------------------------------------------------------------

/**
 * Generate a tidy default variable name for a schema parameter name and a
 * disambiguation counter suffix.
 *
 * Rules:
 *   - Keep only alphabetic characters from the schema param name.
 *   - Lowercase the result.
 *   - If the suffix is 0 and the name is a single letter, omit the suffix
 *     (gives "x" instead of "x0") for the first predicate added.
 *
 * Examples:
 *   ("x", 0, 0)      → "x"
 *   ("x", 1, 0)      → "x1"
 *   ("study", 0, 0)  → "s"
 *   ("study", 1, 0)  → "s1"
 */
function genVarName(schemaName: string, instanceSuffix: number): string {
  const letters = schemaName.replace(/[^a-zA-Z]/g, '').toLowerCase()
  const base = letters.length > 0 ? letters[0] : 'v'
  return instanceSuffix === 0 ? base : `${base}${instanceSuffix}`
}

// ---------------------------------------------------------------------------
// Helpers to compute the variable-colour map
// ---------------------------------------------------------------------------

/**
 * Collect every variable name that appears more than once across all
 * predicate instances.  Returns a map from variable name → colour string.
 *
 * Colours are drawn from a fixed palette in a round-robin fashion.
 */
export const VARIABLE_COLORS = [
  '#2563eb', // blue
  '#16a34a', // green
  '#9333ea', // purple
  '#dc2626', // red
  '#d97706', // amber
  '#0891b2', // cyan
  '#be185d', // pink
  '#4f46e5', // indigo
]

export function computeSharedVariables(
  predicates: PredicateInstance[],
): Map<string, string> {
  const counts = new Map<string, number>()

  for (const pred of predicates) {
    for (const param of pred.params) {
      counts.set(param.varName, (counts.get(param.varName) ?? 0) + 1)
    }
  }

  const colorMap = new Map<string, string>()
  let colorIndex = 0
  for (const [varName, count] of counts) {
    if (count > 1) {
      colorMap.set(varName, VARIABLE_COLORS[colorIndex % VARIABLE_COLORS.length])
      colorIndex++
    }
  }
  return colorMap
}

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------

/**
 * Collect the head variables: all distinct variable names across all
 * predicates, ordered by first occurrence, excluding single-letter
 * underscore-style "sink" variables (i.e. keeping only useful ones).
 *
 * For the head we emit every variable that is NOT a singleton (appears in
 * only one predicate and only once there).  This gives natural join
 * variables in the head.
 *
 * Edge case: if every variable is a singleton we still emit all of them
 * (the query isn't a join, but it's still valid Datalog).
 */
function collectHeadVars(predicates: PredicateInstance[]): string[] {
  const counts = new Map<string, number>()
  const order: string[] = []

  for (const pred of predicates) {
    for (const param of pred.params) {
      if (!counts.has(param.varName)) {
        order.push(param.varName)
      }
      counts.set(param.varName, (counts.get(param.varName) ?? 0) + 1)
    }
  }

  const shared = order.filter((v) => (counts.get(v) ?? 0) > 1)
  return shared.length > 0 ? shared : order
}

/**
 * Serialise a QueryState to valid Datalog text.
 *
 * Example output:
 *   ans(x, s) :- PeakReported(x, y, z, s), Study(s).
 *
 * Returns an empty string when there are no predicates.
 */
export function serializeToDatalog(state: QueryState): string {
  if (state.predicates.length === 0) return ''

  const headVars = collectHeadVars(state.predicates)
  const head = `ans(${headVars.join(', ')})`

  const body = state.predicates
    .map((pred) => {
      const args = pred.params.map((p) => p.varName).join(', ')
      return `${pred.name}(${args})`
    })
    .join(', ')

  return `${head} :- ${body}.`
}

/**
 * Produce a human-readable sentence from the query state.
 *
 * Example:
 *   "Find x, s where PeakReported(x, y, z, s) and Study(s)"
 *
 * Returns an empty string when there are no predicates.
 */
export function serializeToReadable(state: QueryState): string {
  if (state.predicates.length === 0) return ''

  const headVars = collectHeadVars(state.predicates)
  const parts = state.predicates.map((pred) => {
    const args = pred.params.map((p) => p.varName).join(', ')
    return `${pred.name}(${args})`
  })

  return `Find ${headVars.join(', ')} where ${parts.join(' and ')}`
}

// ---------------------------------------------------------------------------
// Unique-id generation (deterministic for tests via counter)
// ---------------------------------------------------------------------------

let _idCounter = 0
/** Reset the id counter – useful in tests so ids are deterministic. */
export function resetIdCounter(): void {
  _idCounter = 0
}

function nextId(): string {
  return `pred-${++_idCounter}`
}

// ---------------------------------------------------------------------------
// QueryModel class
// ---------------------------------------------------------------------------

/**
 * Mutable query-builder with undo / redo support.
 *
 * All mutating operations push a new snapshot onto the undo stack.
 * The model holds a reference to the *current* state as well as the
 * history (undo) and future (redo) stacks.
 */
export class QueryModel {
  private _current: QueryState
  private _undoStack: QueryState[]
  private _redoStack: QueryState[]

  constructor(initial?: QueryState) {
    this._current = initial ?? { predicates: [], varCounter: 0 }
    this._undoStack = []
    this._redoStack = []
  }

  // -------------------------------------------------------------------------
  // State access
  // -------------------------------------------------------------------------

  get state(): QueryState {
    return this._current
  }

  get canUndo(): boolean {
    return this._undoStack.length > 0
  }

  get canRedo(): boolean {
    return this._redoStack.length > 0
  }

  // -------------------------------------------------------------------------
  // Private mutation helpers
  // -------------------------------------------------------------------------

  private pushState(next: QueryState): void {
    this._undoStack.push(this._current)
    this._redoStack = []
    this._current = next
  }

  /** Deep-clone a QueryState (params arrays are shallow-cloned sufficiently). */
  private cloneState(): QueryState {
    return {
      predicates: this._current.predicates.map((pred) => ({
        id: pred.id,
        name: pred.name,
        params: pred.params.map((p) => ({ ...p })),
      })),
      varCounter: this._current.varCounter,
    }
  }

  // -------------------------------------------------------------------------
  // Undo / Redo
  // -------------------------------------------------------------------------

  undo(): void {
    const prev = this._undoStack.pop()
    if (prev === undefined) return
    this._redoStack.push(this._current)
    this._current = prev
  }

  redo(): void {
    const next = this._redoStack.pop()
    if (next === undefined) return
    this._undoStack.push(this._current)
    this._current = next
  }

  // -------------------------------------------------------------------------
  // Mutation operations
  // -------------------------------------------------------------------------

  /**
   * Add a predicate to the query body.
   *
   * Variable names are auto-generated from the schema param names.  If a
   * variable with the same base letter already exists in the current query,
   * a numeric suffix is appended to avoid accidental joins.
   *
   * Returns the id of the newly created PredicateInstance.
   */
  addPredicate(
    name: string,
    schemaParams: string[],
    forceJoin?: Record<string, string>,
  ): string {
    const next = this.cloneState()

    // Build the set of variable names already in use in the current query
    const existingVars = new Set<string>()
    for (const pred of next.predicates) {
      for (const param of pred.params) {
        existingVars.add(param.varName)
      }
    }

    const id = nextId()
    const params: PredicateParam[] = schemaParams.map((schemaName, position) => {
      if (forceJoin && forceJoin[schemaName]) {
        return { position, schemaName, varName: forceJoin[schemaName] }
      }

      // Try suffix 0, 1, 2, … until we find a name not in use
      let suffix = 0
      let candidate = genVarName(schemaName, suffix)
      while (existingVars.has(candidate)) {
        suffix++
        candidate = genVarName(schemaName, suffix)
      }
      existingVars.add(candidate)
      next.varCounter = Math.max(next.varCounter, suffix)
      return { position, schemaName, varName: candidate }
    })

    next.predicates.push({ id, name, params })
    this.pushState(next)
    return id
  }

  /**
   * Remove a predicate instance by id.
   * No-op if the id is not found.
   */
  removePredicate(id: string): void {
    if (!this._current.predicates.some((p) => p.id === id)) return
    const next = this.cloneState()
    next.predicates = next.predicates.filter((p) => p.id !== id)
    this.pushState(next)
  }

  /**
   * Connect (join) a variable slot in one predicate to an existing variable
   * name. This sets the `varName` for the specified (predicateId, position)
   * slot to `targetVarName`.
   *
   * Useful for explicitly wiring together two predicates via a shared
   * variable.
   */
  connectVariable(
    predicateId: string,
    position: number,
    targetVarName: string,
  ): void {
    const pred = this._current.predicates.find((p) => p.id === predicateId)
    if (!pred) return
    const param = pred.params.find((p) => p.position === position)
    if (!param) return

    const next = this.cloneState()
    const nextPred = next.predicates.find((p) => p.id === predicateId)!
    const nextParam = nextPred.params.find((p) => p.position === position)!
    nextParam.varName = targetVarName
    this.pushState(next)
  }

  /**
   * Rename a variable globally across all predicate instances.
   * Useful when the user explicitly renames a variable in the UI.
   */
  renameVariable(oldName: string, newName: string): void {
    if (oldName === newName) return
    const next = this.cloneState()
    for (const pred of next.predicates) {
      for (const param of pred.params) {
        if (param.varName === oldName) {
          param.varName = newName
        }
      }
    }
    this.pushState(next)
  }

  /**
   * Replace the entire state at once (useful for loading a saved query or
   * resetting).  This also clears the undo / redo history.
   */
  reset(state?: QueryState): void {
    this._current = state ?? { predicates: [], varCounter: 0 }
    this._undoStack = []
    this._redoStack = []
  }

  // -------------------------------------------------------------------------
  // Serialisation (convenience delegates)
  // -------------------------------------------------------------------------

  toDatalog(): string {
    return serializeToDatalog(this._current)
  }

  toReadable(): string {
    return serializeToReadable(this._current)
  }

  sharedVariables(): Map<string, string> {
    return computeSharedVariables(this._current.predicates)
  }
}
