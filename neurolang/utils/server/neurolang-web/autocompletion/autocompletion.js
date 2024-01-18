import { FacetsController } from '../facets/facets'
import { API_ROUTE } from '../constants'

/**
 * Class to manage query autocompletion.
 */
export class AutocompletionController {

  constructor (editor, sc, engine) {
    this.editor = editor
    this.sc = sc
    this.engine = engine
    this.fc = new FacetsController(this.editor, this.sc)

    this.editor.on('keydown', (cm, event) => {
      if (event.shiftKey && event.key === 'Tab') {
        // Prevent the default behaviour of the tab key
        event.preventDefault()
        this.fc._cleanAllFacets()
        this._requestAutocomplete()
      }
    })
  }

  updateEngine (engine) {
    this.engine = engine
  }

   /**
  * Test if, in line line, pos is in a pattern
  */
  _isCursorInPattern (line, pos) {
    const pattern = /<[a-zA-Z_]+>/g
    const matches = line.match(pattern)

    if (matches) {
      for (let match of matches) {
        const startIndex = line.indexOf(match)
        const endIndex = startIndex + match.length - 1
        let l = endIndex - startIndex + 1

        if (pos >= startIndex && pos <= endIndex) {
          return {
            content: match,
            start: startIndex,
            end:endIndex
          }
        }

        line = line.substring(0, startIndex) + ' '.repeat(l) + line.substring(startIndex + l)
      }
    }
    return false
  }

  /**
 * Send the input string to the autocompletion endpoint through the
 * autocompletion route and gets the result back.
 */
  _requestAutocomplete () {
    // get the entire text from the CodeMirror instance
    const allText = this.editor.getValue()

    // get the cursor's current position
    const cursorPos = this.editor.getCursor()

    // get the line number where the cursor is
    const cursorLineNumber = cursorPos.line

    const cursorLineContent = this.editor.getLine(cursorLineNumber)
    const cursorLinePosition = cursorPos.ch

    // get the position in the whole text of the first character of that line
    const lineStartPos = this.editor.indexFromPos({ line: cursorLineNumber, ch: 0 })

    // get the position in the whole text of the cursor
    const cursorIndex = this.editor.indexFromPos(cursorPos)

    // Split the text into lines
    let lines = allText.split('\n')

    if (cursorLineNumber >= 0 && cursorLineNumber < lines.length) {
      // in the cursor line, get the substring from the line start to the cursor
      const subline = lines[cursorLineNumber].substring(0, cursorLinePosition)
      // in the cursor line, replace the original line by the substring

      // The substring contains a pattern or the cursor position is in a pattern
      if (  /<[A-Za-z_]+>/.test(subline) || this._isCursorInPattern (lines[cursorLineNumber], cursorLinePosition)) {
        lines[cursorLineNumber] = ''
      }
      // The substring doesn't contain a pattern -> for the cursor line, keep only the substring for autocompletion
      else if (lines[cursorLineNumber].trim()) {
        lines[cursorLineNumber] = subline
      }
    }

    // separate the text to get the symbols and the text for autocompletion
    let cursorline = lines.splice(cursorLineNumber, 1)

    $.post(API_ROUTE.autocompletion, { text: allText, engine: this.engine, line: cursorLineNumber, startpos: lineStartPos, endpos: cursorIndex, notCursorLines: lines.join('\n'), cursorLine: cursorline[0] }, data => {

      // get the entire text from the CodeMirror instance
      const allText = this.editor.getValue()
      let facets = JSON.parse(data.tokens)
      let rules = facets.rules

      // Empty line
      if (!cursorLineContent.trim()) {
        const rule = rules.rule
        const expression = rules.expression
        this.fc.createPatternsContainer()
        this.fc.addPatternsFacetEventListeners()
        this.fc.displayPatternsFacet(expression.values)

      // Not empty line
      } else {
        const pattern = this._isCursorInPattern(cursorLineContent, cursorLinePosition)

        if (pattern) {
          this.editor.setSelection({line: cursorLineNumber, ch: pattern.start}, {line: cursorLineNumber, ch: pattern.end+1})
          const patternContent = pattern.content.slice(1, -1)

          // Pattern is in rule properties
          if (rules.hasOwnProperty(patternContent)) {

            // Pattern has a value
            if (rules[patternContent].hasOwnProperty("values")) {

              // Only one possible value/pattern
              if (rules[patternContent]["values"].length == 1) {
                this._writeValueInTextEditor(rules[patternContent].values[0])

              // Several possible patterns lines
              } else {
                this.fc.createPatternsContainer()
                this.fc.addPatternsFacetEventListeners(true)
                this.fc.displayPatternsFacet(rules[patternContent].values)
              }

            // Pattern does not have value
            } else if (rules[patternContent].hasOwnProperty("params") && rules[patternContent].hasOwnProperty("unit")) {
              if (rules[patternContent].params == "number") {
                this.fc.createInputContainer('Number of ' + rules[patternContent].unit.slice(1, -1) + 's :')
                this.fc.addInputFacetEventListeners(rules[patternContent].unit, rules[patternContent].sep, true)
                this.fc.displayInputFacet()
              }
            }
          }

        } else {
          delete facets.rules
          const k = Object.keys(facets)[0]
          // Only one accepted next token
          if ((Object.keys(facets).length == 1) && (facets[k].length == 1)) {
            this._writeValueInTextEditor(facets[k][0])
          // Several accepted tokens
          } else {
            this.fc.createFacets (facets)
            // Display the facets based on the tokens
            this.fc.displayFacets(facets)
          }
        }
      }
    })
  }

  _writeValueInTextEditor (val) {

    if (this.editor.getSelection().length) {
      var selectedRange = this.editor.getSelection()
      this.editor.replaceSelection(val)
    } else {

      // get the cursor position in the CodeMirror editor
      const cursorPos = this.editor.getCursor()

      // insert the selected value at the current cursor position
      this.editor.replaceRange(val, cursorPos)

      // calculate the end position based on the length of the inserted value
      const endPos = { line: cursorPos.line, ch: cursorPos.ch + val.length }

      // Move cursor to end of inserted value
      this.editor.setCursor(endPos)
    }
  }
}
