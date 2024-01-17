import './autocompletion.css'
//import { SymbolsController } from '../symbols/symbols'
import { API_ROUTE } from '../constants'

/**
 * Class to manage query autocompletion.
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

  constructor (editor, sc, engine) {

    this.editor = editor
    this.sc = sc
    this.engine = engine

    this.editor.on('keydown', (cm, event) => {
      if (event.shiftKey && event.key === 'Tab') {
        // Prevent the default behaviour of the tab key
        event.preventDefault()
        this._cleanFacets()
//        this._createFacets()
//        this._createFacetsContainers()

        // this._requestAutocomplete(contentToCursor)
        this._requestAutocomplete()
      }
    })

    /// Attach the change event listener to the right facet here to make sure we avoid duplicates.
    /// Otherwise the event lister would have to be removed before its creation.
//    this.rightFacetElement = document.getElementById('rightFacet')
//    this.rightFacetElement.addEventListener('change', (event) => {
//      this._handleRightFacetClick(event)
//    })
  }

  updateEngine (engine) {
    this.engine = engine
  }

  _cleanFacets () {
    this._clearFacetsContent()
    this._removeFacetsEventListeners()
    this._removeFacetsContainers()

//    if (this.rightFacetElement) {
      // Clear previous facet items
//      this.rightFacetElement.innerHTML = ''
//    }

//    if (this.leftFacetElement) {
      // Clear previous left facet items
//      this.leftFacetElement.innerHTML = ''
      // Make sure to remove any previous event listener to avoid duplicates
//      if (this._currentLeftFacetHandler) {
//        this._currentLeftFacetHandler = this.leftFacetElement.removeEventListener('change', this._currentLeftFacetHandler)
//      }
      // Remove the left facet container if it already exists
//      this.leftFacetElement = this.leftFacetElement.remove()
//    }


  }

  _clearFacetsContent () {
    if (this.leftFacetLabelElement) {
      // Clear previous left facet label content
      this.leftFacetLabelElement.textContent = ''
    }
    if (this.leftFacetElement) {
      // Clear previous left facet items
      this.leftFacetElement.innerHTML = ''
    }

    if (this.rightFacetLabelElement) {
      // Clear previous left facet label content
      this.rightFacetLabelElement.textContent = ''
    }
    if (this.rightFacetElement) {
      // Clear previous left facet items
      this.rightFacetElement.innerHTML = ''
    }
  }

  _removeFacetsEventListeners () {
    if (this.leftFacetElement) {
      // Make sure to remove any previous event listener to avoid duplicates
      if (this._currentLeftFacetHandler) {
        this._currentLeftFacetHandler = this.leftFacetElement.removeEventListener('change', this._currentLeftFacetHandler)
      }
    }

    if (this.rightFacetElement) {
      // Make sure to remove any previous event listener to avoid duplicates
      if (this._currentRightFacetHandler) {
        this._currentRightFacetHandler = this.rightFacetElement.removeEventListener('change', this._currentRightFacetHandler)
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
      this._createFacets (facets)

      // Display the facets based on the tokens
      this._displayFacets(facets)
    })
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
    // set the facets container back to be hidden
    this.facetsContainerElement.style.display = 'none'

    // Refocus the CodeMirror editor to keep the cursor visible in the textarea
    this.editor.focus()
  }
}
