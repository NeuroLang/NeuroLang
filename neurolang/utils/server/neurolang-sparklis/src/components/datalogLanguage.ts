/**
 * datalogLanguage.ts
 *
 * CodeMirror 6 language extension for Datalog syntax highlighting.
 *
 * Uses the StreamLanguage API (CM6 wrapper around the classic streaming
 * tokenizer pattern) together with @lezer/highlight tags to produce
 * properly highlighted Datalog programs.
 *
 * Token categories:
 *   - Keywords: :- (neck), exists, PROB
 *   - Operators: &, ~, =, !=, <, <=, >, >=
 *   - Punctuation: ( ) , .
 *   - Strings: single- and double-quoted literals
 *   - Numbers: integer and floating-point
 *   - Comments: % line comments
 *   - Predicates/identifiers: upper-case-starting names are treated as
 *     "type names" (relation names) and lower-case as variable names
 */

import {
  StreamLanguage,
  HighlightStyle,
  syntaxHighlighting,
  StringStream,
} from '@codemirror/language'
import { tags as t } from '@lezer/highlight'
import { Extension } from '@codemirror/state'

// ---------------------------------------------------------------------------
// Datalog keywords
// ---------------------------------------------------------------------------

const KEYWORDS = new Set(['exists', 'PROB'])

// Regex patterns (must start with ^ for StreamLanguage.match)
const IDENTIFIER_RE = /^[a-zA-Z_][a-zA-Z0-9_]*/
const NUMBER_RE = /^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?/
const OPERATOR_CHARS = new Set(['&', '~', '=', '!', '<', '>'])

// ---------------------------------------------------------------------------
// Stream parser state
// ---------------------------------------------------------------------------

interface DatalogState {
  inString: false | '"' | "'"
}

function startState(): DatalogState {
  return { inString: false }
}

function copyState(state: DatalogState): DatalogState {
  return { ...state }
}

function token(stream: StringStream, state: DatalogState): string | null {
  // -- Handle string continuation --
  if (state.inString) {
    const quote = state.inString
    while (!stream.eol()) {
      const ch = stream.next()
      if (ch === '\\') {
        stream.next() // consume escape
      } else if (ch === quote) {
        state.inString = false
        break
      }
    }
    return 'string'
  }

  // Skip whitespace
  if (stream.eatSpace()) return null

  // -- Line comment: % ... --
  if (stream.match(/^%[^\n]*/)) return 'comment'

  // -- Neck operator (:-)  --
  if (stream.match(/^:-/)) return 'keyword'

  // -- String literals --
  const ch = stream.peek()
  if (ch === '"' || ch === "'") {
    stream.next()
    state.inString = ch
    // tokenize the first chunk inside the same call
    while (!stream.eol()) {
      const c = stream.next()
      if (c === '\\') {
        stream.next()
      } else if (c === ch) {
        state.inString = false
        break
      }
    }
    return 'string'
  }

  // -- Numbers --
  if (stream.match(NUMBER_RE)) return 'number'

  // -- Operators: &, ~, !=, =, <, <=, >, >= --
  if (ch !== undefined && OPERATOR_CHARS.has(ch)) {
    stream.next()
    // consume second char for two-char ops
    const peek2 = stream.peek()
    if (
      (ch === '!' && peek2 === '=') ||
      (ch === '<' && peek2 === '=') ||
      (ch === '>' && peek2 === '=')
    ) {
      stream.next()
    }
    return 'operator'
  }

  // -- Identifiers and keywords --
  if (stream.match(IDENTIFIER_RE)) {
    const word = stream.current()
    if (KEYWORDS.has(word)) return 'keyword'
    // Upper-case first letter → predicate/relation name (typeName)
    if (/^[A-Z_]/.test(word)) return 'typeName'
    // Lower-case → variable or helper identifier
    return 'variableName'
  }

  // -- Punctuation: ( ) , . --
  const c = stream.next()
  if (c !== undefined && c !== null && '(),.'.includes(c)) return 'punctuation'

  return null
}

// ---------------------------------------------------------------------------
// StreamLanguage definition
// ---------------------------------------------------------------------------

export const datalogStreamLanguage = StreamLanguage.define<DatalogState>({
  name: 'datalog',
  startState,
  copyState,
  token,
  languageData: {
    commentTokens: { line: '%' },
  },
})

// ---------------------------------------------------------------------------
// Highlight style
// ---------------------------------------------------------------------------

export const datalogHighlightStyle = HighlightStyle.define([
  // keywords (:-,  exists, PROB) – blue
  { tag: t.keyword, color: '#1d4ed8', fontWeight: 'bold' },
  // relation / predicate names (upper-case) – purple
  { tag: t.typeName, color: '#7c3aed', fontWeight: '600' },
  // variable names (lower-case) – teal
  { tag: t.variableName, color: '#0f766e' },
  // operators (&, ~, =, …) – dark red
  { tag: t.operator, color: '#b91c1c' },
  // strings – amber
  { tag: t.string, color: '#b45309' },
  // numbers – green
  { tag: t.number, color: '#15803d' },
  // punctuation – grey
  { tag: t.punctuation, color: '#6b7280' },
  // comments – grey italic
  { tag: t.comment, color: '#9ca3af', fontStyle: 'italic' },
])

// ---------------------------------------------------------------------------
// Convenience export: combined extension array
// ---------------------------------------------------------------------------

/**
 * Returns a CodeMirror Extension array that enables Datalog syntax
 * highlighting.  Pass this as part of the `extensions` array when creating
 * an EditorView.
 */
export function datalogLanguage(): Extension[] {
  return [datalogStreamLanguage, syntaxHighlighting(datalogHighlightStyle)]
}
