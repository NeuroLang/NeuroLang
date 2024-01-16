import './autocompletion.css'
import { SymbolsController } from '../symbols/symbols'
import { API_ROUTE } from '../constants'

/**
 * Class to manage query submission.
 */
export class AutocompletionController {

  /// HTML related properties
  facetsContainerElement = document.getElementById('facetsContainer')
  leftFacetElement = null
  leftFacetLabelElement = null
  leftFacetContainerElement = null
  _currentLeftFacetHandler = null
  rightFacetElement = null
  rightFacetLabelElement = null
  rightFacetContainerElement = null
  _currentRightFacetHandler = null
  patternFacetElement = null
  patternFacetLabelElement = null
  patternFacetContainerElement = null
  _currentPatternFacetHandler = null

  inputFacetElement = null
  inputFacetLabelElement = null
  inputFacetButtonElement = null
  inputFacetContainerElement = null

  constructor (editor, sc, engine) {

    this.editor = editor
    this.sc = sc
    this.engine = engine

    this.editor.on('keydown', (cm, event) => {
      if (event.shiftKey && event.key === 'Tab') {
        // Prevent the default behaviour of the tab key
        event.preventDefault()
        this._cleanAllFacets()
//        this._createFacets()
//        this._createFacetsContainers()

        // this._requestAutocomplete(contentToCursor)
        this._requestAutocomplete()
      }
    })
  }

  updateEngine (engine) {
    this.engine = engine
  }

  _cleanAllFacets () {
    this._clearFacetsContent()
    this._removeFacetsEventListeners()
    this._removeFacetsContainers()
  }

  // Clear previous facets contents
  _clearFacetsContent () {
    if (this.leftFacetLabelElement) {
      this.leftFacetLabelElement.textContent = ''
    }
    if (this.leftFacetElement) {
      this.leftFacetElement.innerHTML = ''
    }

    if (this.rightFacetLabelElement) {
      this.rightFacetLabelElement.textContent = ''
    }
    if (this.rightFacetElement) {
      this.rightFacetElement.innerHTML = ''
    }

    if (this.patternFacetLabelElement) {
      this.patternFacetLabelElement.textContent = ''
    }
    if (this.patternFacetElement) {
      this.patternFacetElement.innerHTML = ''
    }

    if (this.inputFacetLabelElement) {
      this.inputFacetLabelElement.textContent = ''
    }
    if (this.inputFacetElement) {
      this.inputFacetElement.value = ''
    }
    if (this.inputFacetButtonElement) {
      this.inputFacetButtonElement.innerHTML = ''
    }
  }

  // Make sure to remove any previous event listener to avoid duplicates
  _removeFacetsEventListeners () {
    if (this.leftFacetElement) {
      if (this._currentLeftFacetHandler) {
        this._currentLeftFacetHandler = this.leftFacetElement.removeEventListener('change', this._currentLeftFacetHandler)
      }
    }
    if (this.rightFacetElement) {
      if (this._currentRightFacetHandler) {
        this._currentRightFacetHandler = this.rightFacetElement.removeEventListener('change', this._currentRightFacetHandler)
      }
    }
    if (this.patternFacetElement) {
      if (this._currentPatternFacetHandler) {
        this._currentPatternFacetHandler = this.patternFacetElement.removeEventListener('change', this._currentPatternFacetHandler)
      }
    }
    if (this.inputFacetButtonElement) {
      if (this._currentInputFacetHandler) {
        this._currentInputFacetHandler = this.inputFacetButtonElement.removeEventListener('click', this._currentInputFacetHandler)
      }
    }
  }

  _removeFacetsContainers () {

    // Left facet container

    // Remove the left facet container if it already exists
    if (this.leftFacetLabelElement) {
      this.leftFacetLabelElement = this.leftFacetLabelElement.remove()
    }
    if (this.leftFacetElement) {
      this.leftFacetElement = this.leftFacetElement.remove()
    }
    if (this.leftFacetContainerElement) {
      this.leftFacetContainerElement = this.leftFacetContainerElement.remove()
    }

    // Right facet container

    // Remove the right facet container if it already exists
    if (this.rightFacetLabelElement) {
      this.rightFacetLabelElement = this.rightFacetLabelElement.remove()
    }
    if (this.rightFacetElement) {
      this.rightFacetElement = this.rightFacetElement.remove()
    }
    if (this.rightFacetContainerElement) {
      this.rightFacetContainerElement = this.rightFacetContainerElement.remove()
    }

    // Pattern facet container

    // Remove the right facet container if it already exists
    if (this.patternFacetLabelElement) {
      this.patternFacetLabelElement = this.patternFacetLabelElement.remove()
    }
    if (this.patternFacetElement) {
      this.patternFacetElement = this.patternFacetElement.remove()
    }
    if (this.patternFacetContainerElement) {
      this.patternFacetContainerElement = this.patternFacetContainerElement.remove()
    }

    // Input facet container

    if (this.inputFacetLabelElement) {
      this.inputFacetLabelElement = this.inputFacetLabelElement.remove()
    }
    if (this.inputFacetElement) {
      this.inputFacetElement = this.inputFacetElement.remove()
    }
    if (this.inputFacetButtonElement) {
      this.inputFacetButtonElement = this.inputFacetButtonElement.remove()
    }
    if (this.inputFacetContainerElement) {
      this.inputFacetContainerElement = this.inputFacetContainerElement.remove()
    }
  }

  _createFacets (facetsObject) {
    this._createFacetsContainers()
    this._addFacetsEventListeners(facetsObject)
  }

  _createFacetsContainers () {
    // Retrieve the div element by class
    var containerDiv = document.querySelector('.ui.segment.code-mirror-container.facetsInnerContainer');

    // Create the first 'facet' div
    var leftFacetDiv = document.createElement('div');
    leftFacetDiv.className = 'facet';
    leftFacetDiv.id = 'leftFacetContainer';

    // Create label for the first select
    var leftLabel = document.createElement('label');
    leftLabel.id = 'leftLabel';
    leftLabel.htmlFor = 'leftFacet';
    leftLabel.textContent = 'Categories';

    // Create the first select
    var leftSelect = document.createElement('select');
    leftSelect.id = 'leftFacet';
    leftSelect.size = '5';

    // Append the label and select to the first 'facet' div
    leftFacetDiv.appendChild(leftLabel);
    leftFacetDiv.appendChild(leftSelect);

    // Append the first 'facet' div to the container
    containerDiv.appendChild(leftFacetDiv);

    // Define class element
    this.leftFacetElement = document.getElementById('leftFacet')
    this.leftFacetLabelElement = document.getElementById('leftLabel')
    this.leftFacetContainerElement = document.getElementById('leftFacetContainer')

    // Repeat the process for the second 'facet'
    var rightFacetDiv = document.createElement('div');
    rightFacetDiv.className = 'facet';
    rightFacetDiv.id = 'rightFacetContainer';

    var rightLabel = document.createElement('label');
    rightLabel.id = 'rightLabel';
    rightLabel.htmlFor = 'rightFacet';
    rightLabel.textContent = 'Values';

    var rightSelect = document.createElement('select');
    rightSelect.id = 'rightFacet';
    rightSelect.size = '5';

    rightFacetDiv.appendChild(rightLabel);
    rightFacetDiv.appendChild(rightSelect);

    containerDiv.appendChild(rightFacetDiv);

    // Define class element
    this.rightFacetElement = document.getElementById('rightFacet')
    this.rightFacetLabelElement = document.getElementById('rightLabel')
    this.rightFacetContainerElement = document.getElementById('rightFacetContainer')
  }

  _addFacetsEventListeners (facetsObject) {

    // Left facet

    // Create a new handler function that has access to facetsObject
    this._currentLeftFacetHandler = (event) => this._handleLeftFacetClick(event, facetsObject)

    // Attach the new event listener
    this.leftFacetElement.addEventListener('change', this._currentLeftFacetHandler)

    // Right facet

    // Create a new handler function that has access to facetsObject
    this._currentRightFacetHandler = (event) => this._handleRightFacetClick(event)

    // Attach the new event listener
    this.rightFacetElement.addEventListener('change', this._currentRightFacetHandler)
  }

  _getWordAtPosition (line, position) {
    // Regular expression to match a word (sequence of non-space, non-parenthesis characters)
    const wordRegex = /[^\s]+/g;

    let match;
    while ((match = wordRegex.exec(line)) !== null) {
        // Check if the position is within the bounds of the current word
        if (match.index <= position && wordRegex.lastIndex > position) {
            return match[0];
        }
    }

    // Return null if no word is found at the position
    return null;
  }

  _isCursorInPattern (line, pos) {
    const pattern = /<[a-zA-Z_]+>/g
    const matches = line.match(pattern)

    if (matches) {
      for (let match of matches) {
        const startIndex = line.indexOf(match)
        const endIndex = startIndex + match.length - 1

        if (pos >= startIndex && pos <= endIndex) {
          return {
           content: match,
           start: startIndex,
           end: endIndex
          }
        }
        line = line.replace(match, ' '.repeat(match.length))
      }
    }
    return false
  }

  _checkWordFormat (word) {
    if (word) {
      return word.startsWith('<') && word.endsWith('>');
    }
    return false
  }

  _lineContainsPattern (line) {
    // Regular expression to match a word (sequence of non-space, non-parenthesis characters)
    const wordRegex = /<[A-Za-z_]+>/;
    // Line contains a pattern
    if (wordRegex.test(line)) {
      return true
    }
    // Return false if no pattern is found in line
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

    const postData = {
      curCursor: cursorPos,
      cursorLineNumber: cursorPos.line,
      pattern: false
    }

    console.log("cursorLineNumber :", cursorLineNumber)
    console.log("this.editor.getLine(cursorLineNumber) :", this.editor.getLine(cursorLineNumber))
    console.log("cursorLineContent :", cursorLineContent)
    console.log("lines[cursorLineNumber] :", lines[cursorLineNumber])

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
        this._createPatternsContainer()
        this._addPatternsFacetEventListeners()
        this._displayPatternsFacet(rule.values)

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
                this._createPatternsContainer()
                this._addPatternsFacetEventListeners(true)
                this._displayPatternsFacet(rules[patternContent].values)
              }

            // Pattern does not have value
            } else if (rules[patternContent].hasOwnProperty("params") && rules[patternContent].hasOwnProperty("unit")) {
              if (rules[patternContent].params == "number") {
                this._createInputContainer('Number of ' + rules[patternContent].unit.slice(1, -1) + 's :')
                this._addInputFacetEventListeners(rules[patternContent].unit, rules[patternContent].sep, true)
                this._displayInputFacet()
              }
            }
          }

        //
        } else {
          delete facets.rules
          const k = Object.keys(facets)[0]

          // Only one accepted next token
          if ((Object.keys(facets).length == 1) && (facets[k].length == 1)) {
            this._writeValueInTextEditor(facets[k][0])

          // Several accepted tokens
          } else {
            this._createFacets (facets)

            // Display the facets based on the tokens
            this._displayFacets(facets)
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

  _createInputContainer (labelText) {
    // Retrieve the div element by class
    var containerDiv = document.querySelector('.ui.segment.code-mirror-container.facetsInnerContainer');

    // Create the input 'facet' div
    var inputFacetDiv = document.createElement('div');
    inputFacetDiv.className = 'facet';
    inputFacetDiv.id = 'inputFacetContainer';

    // Create label for the input
    var inputLabel = document.createElement('label');
    inputLabel.id = 'inputLabel';
    inputLabel.htmlFor = 'inputFacet';
    inputLabel.textContent = labelText;

    // Create the input
    var inputInput = document.createElement('input');
    inputInput.id = 'inputFacet';
    inputInput.type = 'text';

    // Create the Ok button
    var okButton = document.createElement('button')
    okButton.id = 'inputOkButton'
    okButton.textContent = 'OK'

    // Append the label and select to the first 'facet' div
    inputFacetDiv.appendChild(inputLabel);
    inputFacetDiv.appendChild(inputInput);
    inputFacetDiv.appendChild(okButton);

    // Append the first 'facet' div to the container
    containerDiv.appendChild(inputFacetDiv);

    // Define class element
    this.inputFacetElement = document.getElementById('inputFacet')
    this.inputFacetLabelElement = document.getElementById('inputLabel')
    this.inputFacetButtonElement = document.getElementById('inputOkButton')
    this.inputFacetContainerElement = document.getElementById('inputFacetContainer')
  }

  _displayInputFacet () {
    // Display the facets container
    // overrides the 'display: none;' from the CSS.
    this.facetsContainerElement.style.display = 'flex'

    // This will always run, whether there are tokens or not
    this.editor.on('cursorActivity', () => {
      // set the facets container back to be hidden
      this.facetsContainerElement.style.display = 'none'
    })
  }

  _displayPatternsFacet (ruleObject) {
//    console.log(" ")
//    console.log("____in _displayPatternsFacet (ruleObject)____")
    // Add 'All' option to the left facet
//    const deselectOption = document.createElement('option')
//    deselectOption.textContent = 'All'
//    this.patternFacetElement.appendChild(deselectOption)

//    console.log("ruleObject :", JSON.stringify(ruleObject, null, 2))

    for (let item of ruleObject) {
//      console.log("  ", item, " : ", typeof item)
      if (typeof item === "string") {
        const option = document.createElement('option')
        option.textContent = item
        this.patternFacetElement.appendChild(option)
      } else {
        const key = Object.keys(item)[0]
//        console.log("optgroup key :", key)
        let optgroup = document.createElement('optgroup')
        optgroup.label = key
//        console.log("item :", item)
        for (let i of item[key]) {
          const option = document.createElement('option')
          option.textContent = i
          optgroup.appendChild(option)
        }
        this.patternFacetElement.appendChild(optgroup)
      }
    }

    // Display the facets container
    // overrides the 'display: none;' from the CSS.
    this.facetsContainerElement.style.display = 'flex'

    // This will always run, whether there are tokens or not
    this.editor.on('cursorActivity', () => {
      // set the facets container back to be hidden
      this.facetsContainerElement.style.display = 'none'
    })
  }

  _createPatternsContainer () {
    // Retrieve the div element by class
    var containerDiv = document.querySelector('.ui.segment.code-mirror-container.facetsInnerContainer');

    // Create the first 'facet' div
    var patternFacetDiv = document.createElement('div');
    patternFacetDiv.className = 'facet';
    patternFacetDiv.id = 'patternFacetContainer';

    // Create label for the first select
    var patternLabel = document.createElement('label');
    patternLabel.id = 'patternLabel';
    patternLabel.htmlFor = 'patternFacet';
    patternLabel.textContent = 'Patterns';

    // Create the first select
    var patternSelect = document.createElement('select');
    patternSelect.id = 'patternFacet';
    patternSelect.size = '5';

    // Append the label and select to the first 'facet' div
    patternFacetDiv.appendChild(patternLabel);
    patternFacetDiv.appendChild(patternSelect);

    // Append the first 'facet' div to the container
    containerDiv.appendChild(patternFacetDiv);

    // Define class element
    this.patternFacetElement = document.getElementById('patternFacet')
    this.patternFacetLabelElement = document.getElementById('patternLabel')
    this.patternFacetContainerElement = document.getElementById('patternFacetContainer')
  }

  _addInputFacetEventListeners(patternUnit, patternSep, inPattern = false) {
    // Create a new handler function that has access to facetsObject
    this._currentInputFacetHandler = (event) => this._handleInputFacetClick(event, patternUnit, patternSep, inPattern)

    // Attach the new event listener
    this.inputFacetButtonElement.addEventListener('click', this._currentInputFacetHandler)
  }

  _handleInputFacetClick (event, patternUnit, patternSep, inPattern = false) {
    // Get the value from the input
    const inputValue = parseInt(this.inputFacetElement.value, 10)
    const valuesToWrite = (new Array(inputValue).fill(patternUnit)).join(' '+patternSep+' ')

    // Check if inputValue is not empty
    if (inputValue) {
      if (this.editor.getSelection().length) {
//        var selectedRange = this.editor.getSelection()
        this.editor.replaceSelection(valuesToWrite)
      } else {
        // get the cursor position in the CodeMirror editor
        const cursorPos = this.editor.getCursor()

        // insert the selected value at the current cursor position
        this.editor.replaceRange(valuesToWrite, cursorPos)

        // calculate the end position based on the length of the inserted value
        const endPos = { line: cursorPos.line, ch: cursorPos.ch + valuesToWrite.length }

        // Move cursor to end of inserted value
        this.editor.setCursor(endPos)
      }

      // check if the selected value starts with '<' and ends with '>'
//      if (selectedValue == "<identifier_regexp>") {
//        cursorPos = this.editor.getCursor()
//
//        endPos = { line: cursorPos.line, ch: cursorPos.ch }
//        cursorPos.ch = cursorPos.ch - selectedValue.length
//        // select the text that was just inserted
//        this.editor.setSelection(cursorPos, endPos)
//      }
    }
  }

  _addPatternsFacetEventListeners(inPattern = false) {
    // Right facet

    // Create a new handler function that has access to facetsObject
    this._currentPatternFacetHandler = (event, inPattern) => this._handlePatternFacetClick(event, inPattern)

    // Attach the new event listener
    this.patternFacetElement.addEventListener('change', this._currentPatternFacetHandler)
  }

  _handlePatternFacetClick (inPattern = false) {
    const selectedValue = this.patternFacetElement.value

    // get the cursor position in the CodeMirror editor
    var cursorPos = this.editor.getCursor()

    // calculate the end position based on the length of the inserted value
    var endPos = { line: cursorPos.line, ch: cursorPos.ch + selectedValue.length }

    if (selectedValue && selectedValue !== 'All') {

      if (inPattern) {
        var selectedRange = this.editor.getSelection()
        this.editor.replaceSelection(selectedValue)
      } else {
        // get the cursor position in the CodeMirror editor
//        const cursorPos = this.editor.getCursor()

        // insert the selected value at the current cursor position
        this.editor.replaceRange(selectedValue, cursorPos)

        // calculate the end position based on the length of the inserted value
//        const endPos = { line: cursorPos.line, ch: cursorPos.ch + selectedValue.length }

        // Move cursor to end of inserted value
        this.editor.setCursor(endPos)
      }

      // check if the selected value starts with '<' and ends with '>'
      const valuesToSelect = ["<identifier_regexp>", "<cmd_identifier>"]
//      if (selectedValue == "<identifier_regexp>") {
      if (valuesToSelect.includes(selectedValue)) {
        cursorPos = this.editor.getCursor()

        endPos = { line: cursorPos.line, ch: cursorPos.ch }
        cursorPos.ch = cursorPos.ch - selectedValue.length
        // select the text that was just inserted
        this.editor.setSelection(cursorPos, endPos)
      }
//      else {
        // Move cursor to end of inserted value
//        this.editor.setCursor(endPos)
//      }

    }

    // Hide the facets and 'ok' button as they are no longer needed
    // set the facets container back to be hidden
    this.facetsContainerElement.style.display = 'none'

    // Refocus the CodeMirror editor to keep the cursor visible in the textarea
    this.editor.focus()
  }

  /**
 * Displays the facets.
 * @param {Object} facetsObject the categories to be displayed in the facets and their values
 */
  _displayFacets (facetsObject) {

    // Add 'All' option to the left facet
    const deselectOption = document.createElement('option')
    deselectOption.textContent = 'All'
    this.leftFacetElement.appendChild(deselectOption)

    // Add facetsObject keys to left facet
    for (const key of Object.keys(facetsObject)) {
      const option = document.createElement('option')
      option.textContent = key
      this.leftFacetElement.appendChild(option)
    }

    // Display the facets container
    // overrides the 'display: none;' from the CSS.
    this.facetsContainerElement.style.display = 'flex'

    // This will always run, whether there are tokens or not
    this.editor.on('cursorActivity', () => {
      // set the facets container back to be hidden
      this.facetsContainerElement.style.display = 'none'
    })
  }

  /**
 * Handles the click event for the left facet.
 * @param {Event} event the event object associated with the click in the left facet
 * @param {Object} facetsObject the categories to be displayed in the facets and their values
 */
  _handleLeftFacetClick (event, facetsObject) {

    // Retrieve the selected option
    const selectedKey = event.target.value

    // Clear previous options
    while (this.rightFacetElement.firstChild) {
      this.rightFacetElement.removeChild(this.rightFacetElement.firstChild)
    }

    if (selectedKey === 'All') {
      for (const key of Object.keys(facetsObject)) {
        for (const value of facetsObject[key]) {
          const option = document.createElement('option')
          option.value = value
          option.textContent = value
          this.rightFacetElement.appendChild(option)
        }
      }
    } else if (facetsObject[selectedKey]) {
      for (const value of facetsObject[selectedKey]) {
        const option = document.createElement('option')
        option.value = value
        option.textContent = value
        this.rightFacetElement.appendChild(option)
      }
    }

    this.editor.focus()
  }

  /**
 * Handles the click event for the right facet.
 * @param {Event} event the event object associated with the click in the right facet
 */
  _handleRightFacetClick (event) {
    const selectedValue = this.rightFacetElement.value

    const dropdownItems = []
    this.sc.dropdown.find('.menu .item').each(function () {
      dropdownItems.push($(this).data('value'))
    })
    if (dropdownItems.includes(selectedValue)) {
      // this.sc.dropdown is a jQuery object wrapping the dropdown element
      this.sc.dropdown.dropdown('set selected', selectedValue)
    }

    if (selectedValue && selectedValue !== 'All') {
      // get the cursor position in the CodeMirror editor
      const cursorPos = this.editor.getCursor()

      // insert the selected value at the current cursor position
      this.editor.replaceRange(selectedValue, cursorPos)

      // calculate the end position based on the length of the inserted value
      const endPos = { line: cursorPos.line, ch: cursorLinePosition + selectedValue.length }

      // check if the selected value starts with '<' and ends with '>'
      if (selectedValue.startsWith('<') && selectedValue.endsWith('>')) {
        // select the text that was just inserted
        this.editor.setSelection(cursorPos, endPos)
      } else {
        // Move cursor to end of inserted value
        this.editor.setCursor(endPos)
      }
    }

    // Hide the facets and 'ok' button as they are no longer needed
    // set the facets container back to be hidden
    this.facetsContainerElement.style.display = 'none'

    // Refocus the CodeMirror editor to keep the cursor visible in the textarea
    this.editor.focus()
  }
}
