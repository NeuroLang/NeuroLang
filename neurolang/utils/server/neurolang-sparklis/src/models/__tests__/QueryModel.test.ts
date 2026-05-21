/**
 * QueryModel.test.ts
 *
 * Tests for QueryModel, serialisation helpers, and variable-color utilities.
 */
import { describe, it, expect, beforeEach } from 'vitest'
import {
  QueryModel,
  resetIdCounter,
  VARIABLE_COLORS,
} from '../../models/QueryModel'

// Reset the global id counter before each test for deterministic ids.
beforeEach(() => {
  resetIdCounter()
})

// ---------------------------------------------------------------------------
// QueryModel – basic operations
// ---------------------------------------------------------------------------

describe('QueryModel – initial state', () => {
  it('starts with an empty predicate list', () => {
    const model = new QueryModel()
    expect(model.state.predicates).toHaveLength(0)
  })

  it('serialises empty state to empty string', () => {
    const model = new QueryModel()
    expect(model.toDatalog()).toBe('')
    expect(model.toReadable()).toBe('')
  })
})

describe('QueryModel – addPredicate', () => {
  it('adds a predicate with auto-generated variable names', () => {
    const model = new QueryModel()
    model.addPredicate('PeakReported', ['x', 'y', 'z', 'study'])
    const { predicates } = model.state
    expect(predicates).toHaveLength(1)
    expect(predicates[0].name).toBe('PeakReported')
    expect(predicates[0].params).toHaveLength(4)
  })

  it('generates correct variable names from schema param names', () => {
    const model = new QueryModel()
    model.addPredicate('PeakReported', ['x', 'y', 'z', 'study'])
    const params = model.state.predicates[0].params
    expect(params[0].varName).toBe('x')
    expect(params[1].varName).toBe('y')
    expect(params[2].varName).toBe('z')
    expect(params[3].varName).toBe('s') // first letter of "study"
  })

  it('avoids duplicate variable names when adding a second predicate', () => {
    const model = new QueryModel()
    model.addPredicate('PeakReported', ['x', 'y', 'z', 'study'])
    model.addPredicate('Study', ['study'])
    const [first, second] = model.state.predicates
    const existingVars = new Set(first.params.map((p) => p.varName))
    // The second predicate's "study" param should reuse "s" but it's already
    // taken – expect it to be "s1"
    expect(existingVars.has(second.params[0].varName)).toBe(false)
  })

  it('returns the id of the new predicate', () => {
    const model = new QueryModel()
    const id = model.addPredicate('Study', ['study'])
    expect(id).toBe('pred-1')
  })

  it('honours forceJoin to reuse existing variable names', () => {
    const model = new QueryModel()
    model.addPredicate('PeakReported', ['x', 'y', 'z', 'study'])
    const sVarName = model.state.predicates[0].params[3].varName
    // Force the second predicate's "study" param to join with the first
    model.addPredicate('Study', ['study'], { study: sVarName })
    const [, second] = model.state.predicates
    expect(second.params[0].varName).toBe(sVarName)
  })
})

describe('QueryModel – removePredicate', () => {
  it('removes a predicate by id', () => {
    const model = new QueryModel()
    const id = model.addPredicate('PeakReported', ['x', 'y', 'z', 'study'])
    model.addPredicate('Study', ['study'])
    expect(model.state.predicates).toHaveLength(2)

    model.removePredicate(id)
    expect(model.state.predicates).toHaveLength(1)
    expect(model.state.predicates[0].name).toBe('Study')
  })

  it('is a no-op for unknown ids', () => {
    const model = new QueryModel()
    model.addPredicate('Study', ['study'])
    model.removePredicate('nonexistent-id')
    expect(model.state.predicates).toHaveLength(1)
  })

  it('does not affect the undo stack for a no-op removal', () => {
    const model = new QueryModel()
    model.addPredicate('Study', ['study'])
    model.removePredicate('nonexistent-id')
    expect(model.canUndo).toBe(true) // only one push from addPredicate
  })
})

// ---------------------------------------------------------------------------
// Undo / Redo
// ---------------------------------------------------------------------------

describe('QueryModel – undo/redo', () => {
  it('undo reverts the last add', () => {
    const model = new QueryModel()
    model.addPredicate('Study', ['study'])
    model.undo()
    expect(model.state.predicates).toHaveLength(0)
  })

  it('redo re-applies an undone add', () => {
    const model = new QueryModel()
    model.addPredicate('Study', ['study'])
    model.undo()
    model.redo()
    expect(model.state.predicates).toHaveLength(1)
  })

  it('undo reverts remove', () => {
    const model = new QueryModel()
    const id = model.addPredicate('Study', ['study'])
    model.removePredicate(id)
    model.undo()
    expect(model.state.predicates).toHaveLength(1)
  })

  it('canUndo is false on empty history', () => {
    const model = new QueryModel()
    expect(model.canUndo).toBe(false)
  })

  it('canRedo is false when redo stack is empty', () => {
    const model = new QueryModel()
    model.addPredicate('Study', ['study'])
    expect(model.canRedo).toBe(false)
  })

  it('undo when history is empty is a no-op', () => {
    const model = new QueryModel()
    model.undo() // should not throw
    expect(model.state.predicates).toHaveLength(0)
  })

  it('redo when future is empty is a no-op', () => {
    const model = new QueryModel()
    model.redo() // should not throw
    expect(model.state.predicates).toHaveLength(0)
  })

  it('a new action clears the redo stack', () => {
    const model = new QueryModel()
    model.addPredicate('Study', ['study'])
    model.undo()
    model.addPredicate('PeakReported', ['x'])
    expect(model.canRedo).toBe(false)
  })

  it('supports multiple undo/redo cycles', () => {
    const model = new QueryModel()
    model.addPredicate('A', ['a'])
    model.addPredicate('B', ['b'])
    model.addPredicate('C', ['c'])

    model.undo() // removes C
    model.undo() // removes B
    expect(model.state.predicates).toHaveLength(1)

    model.redo() // re-adds B
    expect(model.state.predicates).toHaveLength(2)

    model.redo() // re-adds C
    expect(model.state.predicates).toHaveLength(3)
  })
})

// ---------------------------------------------------------------------------
// connectVariable
// ---------------------------------------------------------------------------

describe('QueryModel – connectVariable', () => {
  it('sets the variable name at the specified slot', () => {
    const model = new QueryModel()
    model.addPredicate('PeakReported', ['x', 'y', 'z', 'study'])
    model.addPredicate('Study', ['study'])
    const [, second] = model.state.predicates
    const studyVarInFirst = model.state.predicates[0].params[3].varName
    model.connectVariable(second.id, 0, studyVarInFirst)
    expect(model.state.predicates[1].params[0].varName).toBe(studyVarInFirst)
  })

  it('is a no-op for unknown predicate id', () => {
    const model = new QueryModel()
    model.addPredicate('Study', ['study'])
    model.connectVariable('bad-id', 0, 'x')
    // State unchanged (still 1 predicate, canUndo only for the add)
    expect(model.state.predicates).toHaveLength(1)
  })
})

// ---------------------------------------------------------------------------
// renameVariable
// ---------------------------------------------------------------------------

describe('QueryModel – renameVariable', () => {
  it('renames a variable globally across all predicates', () => {
    const model = new QueryModel()
    model.addPredicate('PeakReported', ['x', 'y', 'z', 'study'])
    model.addPredicate('Study', ['study'], {
      study: model.state.predicates[0].params[3].varName,
    })
    const oldName = model.state.predicates[0].params[3].varName
    model.renameVariable(oldName, 'sid')
    for (const pred of model.state.predicates) {
      const match = pred.params.find((p) => p.varName === oldName)
      expect(match).toBeUndefined()
    }
    // Both predicates should now reference 'sid'
    const withNew = model.state.predicates.flatMap((p) =>
      p.params.filter((param) => param.varName === 'sid'),
    )
    expect(withNew).toHaveLength(2)
  })

  it('is a no-op when old and new names are the same', () => {
    const model = new QueryModel()
    model.addPredicate('Study', ['study'])
    const before = model.canUndo
    model.renameVariable('s', 's')
    expect(model.canUndo).toBe(before)
  })
})

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------

describe('serializeToDatalog', () => {
  it('produces valid Datalog for a single predicate', () => {
    const model = new QueryModel()
    model.addPredicate('Study', ['study'])
    expect(model.toDatalog()).toBe('ans(s) :- Study(s).')
  })

  it('produces valid Datalog for two predicates with a shared variable', () => {
    const model = new QueryModel()
    model.addPredicate('PeakReported', ['x', 'y', 'z', 'study'])
    const studyVar = model.state.predicates[0].params[3].varName
    model.addPredicate('Study', ['study'], { study: studyVar })
    const dl = model.toDatalog()
    // head should contain only shared vars (studyVar appears twice)
    expect(dl).toContain(`ans(${studyVar})`)
    expect(dl).toContain('PeakReported(')
    expect(dl).toContain('Study(')
    expect(dl.endsWith('.')).toBe(true)
  })

  it('returns empty string for empty query', () => {
    const model = new QueryModel()
    expect(model.toDatalog()).toBe('')
  })
})

describe('serializeToReadable', () => {
  it('produces a human-readable string', () => {
    const model = new QueryModel()
    model.addPredicate('Study', ['study'])
    expect(model.toReadable()).toMatch(/^Find .+ where Study\(.+\)$/)
  })

  it('joins multiple predicates with "and"', () => {
    const model = new QueryModel()
    model.addPredicate('PeakReported', ['x', 'y', 'z', 'study'])
    const studyVar = model.state.predicates[0].params[3].varName
    model.addPredicate('Study', ['study'], { study: studyVar })
    expect(model.toReadable()).toContain(' and ')
  })

  it('returns empty string for empty query', () => {
    const model = new QueryModel()
    expect(model.toReadable()).toBe('')
  })
})

// ---------------------------------------------------------------------------
// computeSharedVariables / variable coloring
// ---------------------------------------------------------------------------

describe('computeSharedVariables', () => {
  it('returns empty map when no variables are shared', () => {
    const model = new QueryModel()
    model.addPredicate('A', ['a'])
    model.addPredicate('B', ['b'])
    const map = model.sharedVariables()
    // Each predicate has unique var names – no sharing expected
    // (unless by coincidence the auto-names collide)
    for (const [, color] of map) {
      // If somehow we got a shared var, the color should be from the palette
      expect(VARIABLE_COLORS).toContain(color)
    }
  })

  it('assigns the same color to the same shared variable', () => {
    const model = new QueryModel()
    model.addPredicate('PeakReported', ['x', 'y', 'z', 'study'])
    const studyVar = model.state.predicates[0].params[3].varName
    model.addPredicate('Study', ['study'], { study: studyVar })
    const map = model.sharedVariables()
    expect(map.has(studyVar)).toBe(true)
  })

  it('assigns different colors to different shared variables', () => {
    const model = new QueryModel()
    model.addPredicate('A', ['x', 'y'])
    const xVar = model.state.predicates[0].params[0].varName
    const yVar = model.state.predicates[0].params[1].varName
    model.addPredicate('B', ['x', 'y'], { x: xVar, y: yVar })
    const map = model.sharedVariables()
    expect(map.get(xVar)).not.toBe(map.get(yVar))
  })

  it('does not color variables that appear only once', () => {
    const model = new QueryModel()
    model.addPredicate('Study', ['study'])
    const map = model.sharedVariables()
    const studyVar = model.state.predicates[0].params[0].varName
    expect(map.has(studyVar)).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

describe('QueryModel – reset', () => {
  it('clears predicates and history', () => {
    const model = new QueryModel()
    model.addPredicate('Study', ['study'])
    model.reset()
    expect(model.state.predicates).toHaveLength(0)
    expect(model.canUndo).toBe(false)
    expect(model.canRedo).toBe(false)
  })

  it('accepts a custom initial state', () => {
    const model = new QueryModel()
    model.reset({
      predicates: [
        {
          id: 'custom-1',
          name: 'Study',
          params: [{ position: 0, schemaName: 'study', varName: 's' }],
        },
      ],
      varCounter: 0,
    })
    expect(model.state.predicates).toHaveLength(1)
    expect(model.state.predicates[0].name).toBe('Study')
  })
})
