import { SymbolsController } from '../symbols/symbols'
import { API_ROUTE } from '../constants'

/**
 * Class to manage query submission.
 */
export class AutocompletionController {
  constructor (editor, sc, engine) {
    this.editor = editor
    this.sc = sc
    this.engine = engine

    this.editor.on('keydown', (cm, event) => {
      if (event.shiftKey && event.key === 'Tab') {
        // Prevent the default behaviour of the tab key
        event.preventDefault()
        // this._requestAutocomplete(contentToCursor)
        this._requestAutocomplete()
      }
    })

    /// Attach the change event listener to the right facet here to make sure we avoid duplicates.
    /// Otherwise the event lister would have to be removed before its creation.
    document.getElementById('rightFacet').addEventListener('change', (event) => {
      this._handleRightFacetClick(event)
    })
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
    const lineNumber = cursorPos.line

    // get the position in the whole text of the first character of that line
    const lineStartPos = this.editor.indexFromPos({ line: lineNumber, ch: 0 })

    // get the position in the whole text of the cursor
    const cursorIndex = this.editor.indexFromPos(cursorPos)

    $.post(API_ROUTE.autocompletion, { text: allText, engine: this.engine, line: lineNumber, startpos: lineStartPos, endpos: cursorIndex }, data => {
      const facets = JSON.parse(data.tokens)
      // Display the facets based on the tokens
      this._displayFacets(facets)
    })
  }

  /**
 * Displays the facets.
 * @param {Object} facetsObject the categories to be displayed in the facets and their values
 */
  _displayFacets (facetsObject) {
    const leftFacet = document.getElementById('leftFacet')
    const rightFacet = document.getElementById('rightFacet')
    const facetsContainer = document.getElementById('facetsContainer')

    // Clear previous facet items
    leftFacet.innerHTML = ''
    rightFacet.innerHTML = ''

    // Add 'All' option to the left facet
    const deselectOption = document.createElement('option')
    deselectOption.textContent = 'All'
    leftFacet.appendChild(deselectOption)

    // Add facetsObject keys to left facet
    for (const key of Object.keys(facetsObject)) {
      const option = document.createElement('option')
      option.textContent = key
      leftFacet.appendChild(option)
    }

    // Make sure to remove any previous event listener to avoid duplicates
    if (this._currentLeftFacetHandler) {
      leftFacet.removeEventListener('change', this._currentLeftFacetHandler)
    }

    // Create a new handler function that has access to facetsObject
    this._currentLeftFacetHandler = (event) => this._handleLeftFacetClick(event, facetsObject)

    // Attach the new event listener
    leftFacet.addEventListener('change', this._currentLeftFacetHandler)

    // Display the facets container
    // overrides the 'display: none;' from the CSS.
    facetsContainer.style.display = 'flex'

    // This will always run, whether there are tokens or not
    this.editor.on('cursorActivity', () => {
      // set the facets container back to be hidden
      facetsContainer.style.display = 'none'
    })
  }

  /**
 * Handles the click event for the left facet.
 * @param {Event} event the event object associated with the click in the left facet
 * @param {Object} facetsObject the categories to be displayed in the facets and their values
 */
  _handleLeftFacetClick (event, facetsObject) {
    const rightFacet = document.getElementById('rightFacet')

    // Retrieve the selected option
    const selectedKey = event.target.value

    // Clear previous options
    while (rightFacet.firstChild) {
      rightFacet.removeChild(rightFacet.firstChild)
    }

    if (selectedKey === 'All') {
      for (const key of Object.keys(facetsObject)) {
        for (const value of facetsObject[key]) {
          const option = document.createElement('option')
          option.value = value
          option.textContent = value
          rightFacet.appendChild(option)
        }
      }
    } else if (facetsObject[selectedKey]) {
      for (const value of facetsObject[selectedKey]) {
        const option = document.createElement('option')
        option.value = value
        option.textContent = value
        rightFacet.appendChild(option)
      }
    }

    this.editor.focus()
  }

  /**
 * Handles the click event for the right facet.
 * @param {Event} event the event object associated with the click in the right facet
 */
  _handleRightFacetClick (event) {
    const rightFacet = document.getElementById('rightFacet')
    const selectedValue = rightFacet.value

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
      const endPos = { line: cursorPos.line, ch: cursorPos.ch + selectedValue.length }

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
    const facetsContainer = document.getElementById('facetsContainer')
    // set the facets container back to be hidden
    facetsContainer.style.display = 'none'

    // Refocus the CodeMirror editor to keep the cursor visible in the textarea
    this.editor.focus()
  }
}
