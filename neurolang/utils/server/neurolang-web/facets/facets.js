import './facets.css'
//import { SymbolsController } from '../symbols/symbols'
//import { API_ROUTE } from '../constants'

/**
 * Class to manage facets.
 */
export class FacetsController {

  constructor (editor, sc) {
    this.editor = editor
    this.sc = sc

    /// HTML related properties
    this.facetsContainerElement = document.getElementById('facetsContainer')
    this.leftFacetElement = null
    this.leftFacetLabelElement = null
    this.leftFacetContainerElement = null
    this._currentLeftFacetHandler = null

    this.rightFacetElement = null
    this.rightFacetLabelElement = null
    this.rightFacetContainerElement = null
    this._currentRightFacetHandler = null

    this.patternFacetElement = null
    this.patternFacetLabelElement = null
    this.patternFacetContainerElement = null
    this._currentPatternFacetHandler = null

    this.inputFacetElement = null
    this.inputFacetLabelElement = null
    this.inputFacetButtonElement = null
    this.inputFacetContainerElement = null
  }

  _cleanAllFacets () {
    this._clearFacetsContent()
    this._removeFacetsEventListeners()
    this._removeFacetsContainers()
  }

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

  createFacets (facetsObject) {
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

  createPatternsContainer () {
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

  addPatternsFacetEventListeners(inPattern = false) {
    // Right facet

    // Create a new handler function that has access to facetsObject
    this._currentPatternFacetHandler = (event, inPattern) => this._handlePatternFacetClick(event, inPattern)

    // Attach the new event listener
    this.patternFacetElement.addEventListener('change', this._currentPatternFacetHandler)
  }

  _handlePatternFacetClick (inPattern = false) {
    const selectedValue = this.patternFacetElement.value

    const dropdownItems = []
    this.sc.dropdown.find('.menu .item').each(function () {
      dropdownItems.push($(this).data('value'))
    })
    if (dropdownItems.includes(selectedValue)) {
      // this.sc.dropdown is a jQuery object wrapping the dropdown element
      this.sc.dropdown.dropdown('set selected', selectedValue)
    }

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

  displayPatternsFacet (ruleObject) {

    for (let item of ruleObject) {
      if (typeof item === "string") {
        const option = document.createElement('option')
        option.textContent = item
        this.patternFacetElement.appendChild(option)
      } else {
        const key = Object.keys(item)[0]
        let optgroup = document.createElement('optgroup')
        optgroup.label = key
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

   /**
  * Displays the facets.
  * @param {Object} facetsObject the categories to be displayed in the facets and their values
  */
  displayFacets (facetsObject) {

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

  addInputFacetEventListeners(patternUnit, patternSep, inPattern = false) {
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

  createInputContainer (labelText) {
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

    // Apply ok button styles
    okButton = document.getElementById('inputOkButton')
//    okButton.style.display = 'none'
    okButton.style.padding = '5px 15px'
    okButton.style.alignSelf = 'center'
    okButton.style.marginLeft = '10px'

    // Define class element
    this.inputFacetElement = document.getElementById('inputFacet')
    this.inputFacetLabelElement = document.getElementById('inputLabel')
    this.inputFacetButtonElement = document.getElementById('inputOkButton')
    this.inputFacetContainerElement = document.getElementById('inputFacetContainer')
  }

  displayInputFacet () {
    // Display the facets container
    // overrides the 'display: none;' from the CSS.
    this.facetsContainerElement.style.display = 'flex'

    // This will always run, whether there are tokens or not
    this.editor.on('cursorActivity', () => {
      // set the facets container back to be hidden
      this.facetsContainerElement.style.display = 'none'
    })
  }
}