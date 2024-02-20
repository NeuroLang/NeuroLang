import $ from '../jquery-bundler'

/**
 * Class to manage a facet.
 */
export class Facet {
  /**
  * Constructor of a facet.
  * @param {editor} the editor linked to the textarea
  * @param {facetsContainer} the container in which are all the facets
  * @param {parentContainer} the inner container in which are all the facets, parent of the facets containers
  * @param {containerId} the id of this container
  */
  constructor (editor, facetsContainerElement, parentContainerElement, containerId) {
    this.editor = editor
    this.facetsContainerElement = facetsContainerElement
    this.parentContainerElement = parentContainerElement
    this.containerId = containerId
    this.container = new Container(this.parentContainerElement, this.containerId)
    this.label = null
    this.element = null
  }

  /**
  * Adds a label in this facet.
  * @param {labelId} the id of the label is this facet
  * @param {labelText} the text of the label
  * @param {facetId} the id of this facet
  */
  addLabel (labelId, labelText, facetId) {
    this.label = null
    if (labelId) {
      this.label = new Label(this.container.element, labelId, labelText, facetId)
    }
  }

  /**
  * Adds an element in this facet.
  * @param {elementType} the type of the element between single quotes. Possible values: categories, regexpvalues, patterns, number, regexp, aggregateButton and valueButton.
  * @param {elementId} the id of the element
  * @param {data} the data associated with the element
  * @param {key} the key name of the data to display. Optional.
  */
  addElement (elementType, elementId, data, key = false) {
    this.element = null

    if (elementType === 'categories') {
      this.element = new CategoriesSelect(this.editor, this.facetsContainerElement, this.parentContainerElement, this.container.element, elementId, data, key)
    } else if (elementType === 'regexpvalues') {
      this.element = new RegexpSelect(this.editor, this.facetsContainerElement, this.parentContainerElement, this.container.element, elementId, data, key)
    } else if (elementType === 'patterns') {
      this.element = new PatternsSelect(this.editor, this.facetsContainerElement, this.parentContainerElement, this.container.element, elementId, data, key)
    } else if (elementType === 'number') {
      this.element = new NumberInput(this.editor, this.facetsContainerElement, this.parentContainerElement, this.container.element, elementId, data, key)
    } else if (elementType === 'regexp') {
      this.element = new RegexpInput(this.editor, this.facetsContainerElement, this.parentContainerElement, this.container.element, elementId, data, key)
    } else if (elementType === 'aggregateButton') {
      this.element = new AggregateButton(this.editor, this.facetsContainerElement, this.parentContainerElement, this.container.element, elementId, data, key)
    } else if (elementType === 'valueButton') {
      this.element = new ValueButton(this.editor, this.facetsContainerElement, this.parentContainerElement, this.container.element, elementId, data, key = false)
    }
  }

  /**
  * Removes this facet from the parent container.
  */
  remove () {
    if (this.label) {
      this.label = this.label.remove()
    }
    if (this.element) {
      this.element = this.element.remove()
    }
    if (this.container) {
      this.container = this.container.remove()
    }
    return null
  }
}

/**
 * Class to manage a label.
 */
class Label {
  /**
  * Constructor of a facet.
  * @param {labelContainerElement} the parent container of this label
  * @param {labelId} the id of this label
  * @param {labelText} the text of this label
  * @param {labelForElementId} the id of element to which this label is associated to
  */
  constructor (labelContainerElement, labelId, labelText, labelForElementId) {
    // Create label for the facet
    this.element = document.createElement('label')
    this.element.id = labelId
    this.element.htmlFor = labelForElementId
    this.element.textContent = labelText
    // Append the label and facet to the first 'facet' div
    labelContainerElement.appendChild(this.element)
    // Apply select styles
    this.element.style.marginLeft = '5px'
    this.element.style.marginRight = '5px'
  }

  /**
  * Removes this label from the parent container
  */
  remove () {
    // Clear label
    this.element.textContent = ''
    this.element.htmlFor = ''
    this.element.id = ''
    // Remove label
    this.element = this.element.remove()
    return null
  }
}

/**
 * Class to manage a container.
 */
class Container {
  /**
  * Constructor of a facet.
  * @param {parentContainerElement} the parent container of this container
  * @param {containerId} the id of this container
  */
  constructor (parentContainerElement, containerId) {
    // Create the facet container div
    this.element = document.createElement('div')
    this.element.className = 'facet'
    this.element.id = containerId
    // Append the first 'facet' div to the container
    parentContainerElement.appendChild(this.element)
  }

  /**
  * Removes this container from the parent container
  */
  remove () {
    // Clear container
    this.element.className = ''
    this.element.id = ''
    // Remove container
    this.element = this.element.remove()
    return null
  }
}

/**
 * Class to manage an element.
 */
class Element {
  /**
  * Constructor of an element.
  */
  constructor () {
    this.queryAlert = $('#queryAlert')
    this.qMsg = this.queryAlert.find('.nl-query-message')
    this.resultsContainer = $('#symbolsContainer')
    this.scdropdown = this.resultsContainer.find('.nl-symbols-dropdown')
  }

  /**
  * Set a message in the message box below the query box.
  * @param {string} style the style class for the message (info, error, success, warning)
  * @param {*} content the content for the message
  * @param {*} header the header for the message
  */
  _setAlert (content, header, help) {
    const qHeader = this.queryAlert.find('.nl-query-header')
    if (typeof header !== 'undefined') {
      qHeader.text(header)
    } else {
      qHeader.empty()
    }
    const qMsg = this.queryAlert.find('.nl-query-message')
    if (typeof content !== 'undefined') {
      qMsg.text(content)
    } else {
      qMsg.empty()
    }
    const qHelp = this.queryAlert.find('.nl-query-help')
    if (typeof help !== 'undefined') {
      qHelp.text(help)
      qHelp.show()
    } else {
      qHelp.empty()
      qHelp.hide()
    }
    this.queryAlert.removeClass('info error warning success')
    this.queryAlert.addClass('error')
    this.queryAlert.show()
  }

  _clearAlert () {
    this.queryAlert.hide()
    this.editor.clearGutter('marks')
    this.editor.getAllMarks().forEach((elt) => elt.clear())
  }

  show (editor, facetsContainerElement, mess, alert) {
    // Display the facets container
    // overrides the 'display: none;' from the CSS.
    facetsContainerElement.style.display = 'flex'

    // This will always run, whether there are tokens or not
    editor.on('cursorActivity', () => {
      if (mess) {
        mess.empty()
      }
      if (alert) {
        alert.hide()
      }
      this.editor.clearGutter('marks')
      this.editor.getAllMarks().forEach((elt) => elt.clear())

      // set the facets container back to be hidden
      facetsContainerElement.style.display = 'none'
    })
  }
}

class Button extends Element {
  constructor (editor, facetsContainerElement, parentContainerElement, containerDiv, elementId, data, key) {
    super()
    this.editor = editor
    this.facetsContainerElement = facetsContainerElement
    this.parentContainerElement = parentContainerElement
    this.element = this._createElement(containerDiv, elementId)
    this.allData = data
    this.key = key
    this.clickHandler = null
  }

  hide () {
    this._clearAlert()

    // Hide the facets and 'ok' button as they are no longer needed
    // set the facets container back to be hidden
    this.facetsContainerElement.style.display = 'none'

    // Refocus the CodeMirror editor to keep the cursor visible in the textarea
    this.editor.focus()
  }

  _createElement (containerDiv, elementId) {
    // Create the button
    let element = document.createElement('button')
    element.id = elementId
    element.textContent = 'OK'

    // Append button to parent container
    containerDiv.appendChild(element)

    // Apply ok button styles
    element = document.getElementById(elementId)
    element.style.padding = '5px 15px'
    element.style.alignSelf = 'center'
    // element.style.padding = '5px 15px'
    // element.style.alignSelf = 'center'
    element.style.marginLeft = '5px'
    element.style.marginRight = '5px'
    element.disabled = true
    return element
  }

  _removeElement () {
    // Clear Button
    this.element.innerHTML = ''
    this.element.textContent = ''
    this.element.id = ''
    // Remove button
    this.element = this.element.remove()
  }
}

class AggregateButton extends Button {
  constructor (editor, facetsContainerElement, parentContainerElement, containerDiv, elementId, data, key) {
    super(editor, facetsContainerElement, parentContainerElement, containerDiv, elementId, data, key)
    this.selectFacets = []
  }

  addClickEventListeners (editor, facetsContainerElement, patternUnit, patternSep = false, inPattern = false) {
    // Create a new handler function that has access to facetsObject
    this.clickHandler = (event) => this._handleClick(editor, facetsContainerElement, patternUnit, patternSep, inPattern)

    // Attach the new event listener
    this.element.addEventListener('click', this.clickHandler)
  }

  _handleClick (editor, facetsContainerElement, patternUnit, patternSep, inPattern = false) {
    // Get the value from the input
    const valueToWrite = patternUnit.join(' ' + patternSep + ' ')

    // Check if valuesToWrite is not empty
    if (valueToWrite) {
      if (editor.getSelection().length) {
        editor.replaceSelection(valueToWrite)
      } else {
        // get the cursor position in the CodeMirror editor
        const cursorPos = editor.getCursor()

        // insert the selected value at the current cursor position
        editor.replaceRange(valueToWrite, cursorPos)

        // calculate the end position based on the length of the inserted value
        const endPos = { line: cursorPos.line, ch: cursorPos.ch + valueToWrite.length }

        // Move cursor to end of inserted value
        editor.setCursor(endPos)
      }
    }
    this.hide()
  }

  remove () {
    this._removeClickEventListener()
    this._removeElement()
    return null
  }

  _removeClickEventListener () {
    if (this.element) {
      if (this.clickHandler) {
        this.clickHandler = this.element.removeEventListener('click', this.clickHandler)
      }
    }
  }
}

class ValueButton extends Button {
  addClickEventListeners (editor, facetsContainerElement, valueToWrite, inPattern = false) {
    // Create a new handler function that has access to facetsObject
    this.clickHandler = (event) => this._handleClick(editor, facetsContainerElement, valueToWrite, inPattern)

    // Attach the new event listener
    this.element.addEventListener('click', this.clickHandler)
  }

  _handleClick (editor, facetsContainerElement, valueToWrite, inPattern = false) {
    // Check if valuesToWrite is not empty
    if (valueToWrite.val) {
      if (editor.getSelection().length) {
        editor.replaceSelection(valueToWrite.val)
      } else {
        // get the cursor position in the CodeMirror editor
        const cursorPos = editor.getCursor()

        // insert the selected value at the current cursor position
        editor.replaceRange(valueToWrite.val, cursorPos)

        // calculate the end position based on the length of the inserted value
        const endPos = { line: cursorPos.line, ch: cursorPos.ch + valueToWrite.val.length }

        // Move cursor to end of inserted value
        editor.setCursor(endPos)
      }
    }
    this.hide()
  }

  remove () {
    this._removeClickEventListener()
    this._removeElement()
    return null
  }

  _removeClickEventListener () {
    if (this.element) {
      if (this.clickHandler) {
        this.clickHandler = this.element.removeEventListener('click', this.clickHandler)
      }
    }
  }
}

class Input extends Element {
  constructor (editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key) {
    super()
    this.editor = editor
    this.facetsContainerElement = facetsContainerElement
    this.parentContainerElement = parentContainerElement
    this.elementContainer = elementContainer
    this.allData = data
    this.key = key
    this.element = this._createElement(this.elementContainer, elementId)
    this.facets = []
    this.buttonId = null
  }

  _createElement (containerDiv, elementId) {
    // Create the facet
    let element = document.createElement('input')
    element.id = elementId
    element.type = 'text'
    containerDiv.appendChild(element)

    // Apply select styles
    element = document.getElementById(element.id)
    // element.style.padding = '5px 15px'
    // element.style.alignSelf = 'center'
    element.style.marginLeft = '5px'
    element.style.marginRight = '5px'
    return element
  }

  updateButtonId (buttonId) {
    this.buttonId = buttonId
  }

  getValue () {
    return this.element.value
  }

  _removeElement () {
    // Clear element
    this.element.id = ''
    this.element.innerHTML = ''
    // Append the label and facet to the first 'facet' div
    this.element = this.element.remove()
  }

  _removeFacets () {
    if (this.facets.length) {
      for (let f of this.facets) {
        if (f) {
          f = f.remove()
        }
      }
      this.facets = []
    }
  }
}

class NumberInput extends Input {
  constructor (editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key) {
    super(editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key)
    this.changeHandler = null
  }

  addChangeEventListeners (editor, inPattern = false) {
    // Create a new handler function that has access to facetsObject
    this.changeHandler = (event) => this._handleChange(event, editor, inPattern)

    // Attach the new event listener
    this.element.addEventListener('input', this.changeHandler)
  }

  _handleChange (event, editor, inPattern) {
    // Clear the previous facets created if another regexp option was selected before
    this._removeFacets()

    const intValue = parseInt(this.element.value, 10)
    if (!isNaN(intValue)) {
      this._clearAlert()
      const selectedValues = new Array(intValue).fill(null)
      const buttonId = this.element.id + '_button_element'
      for (let step = 0; step < intValue; step++) {
        const unitKey = this.allData[this.key].unit.slice(1, -1)

        // Create
        const patternFacetNew = new Facet(
          this.editor,
          this.facetsContainerElement,
          this.parentContainerElement,
          this.element.id + '_select_container_' + (step + 1))

        patternFacetNew.addLabel(
          this.element.id + '_select_label_' + (step + 1),
          unitKey + ' ' + (step + 1) + ' options :',
          this.element.id + '_select_element_' + (step + 1))

        patternFacetNew.addElement(
          'regexpvalues',
          this.element.id + '_select_element_' + (step + 1),
          this.allData,
          this.key)
        patternFacetNew.element.updatePosition(step)
        patternFacetNew.element.updateButtonId(buttonId)

        patternFacetNew.element.addChangeEventListeners(this.editor, this.facetsContainerElement, selectedValues)

        // Display
        patternFacetNew.element.fill(unitKey)
        patternFacetNew.element.show(this.editor, this.facetsContainerElement, this.qMsg, this.queryAlert)

        // Add to facets
        this.facets.push(patternFacetNew)
      }

      const buttonNew = new Facet(
        this.editor,
        this.facetsContainerElement,
        this.parentContainerElement,
        this.element.id + '_button_container')

      buttonNew.addElement(
        'aggregateButton',
        buttonId,
        this.allData,
        this.key)
      buttonNew.element.addClickEventListeners(this.editor, this.facetsContainerElement, selectedValues, this.allData[this.key].sep, inPattern)

      // Display
      buttonNew.element.show(this.editor, this.facetsContainerElement, this.qMsg, this.queryAlert)
      this.facets.push(buttonNew)
    } else {
      if (!(this.element.value).length) {
        this._clearAlert()
      } else {
        this._setAlert('The current value is of incorrect type.', 'Type of value error', 'The entered value must be an integer !!!')
      }
    }
  }

  remove () {
    this._clearAlert()
    this._removeFacets()
    this._removeChangeEventListener()
    this._removeElement()
    return null
  }

  _removeChangeEventListener () {
    if (this.element) {
      if (this.changeHandler) {
        this.changeHandler = this.element.removeEventListener('input', this.changeHandler)
      }
    }
  }
}

class RegexpInput extends Input {
  constructor (editor, facetsContainerElement, parentContainerElement, containerDiv, elementId, data, key) {
    super(editor, facetsContainerElement, parentContainerElement, containerDiv, elementId, data, key)
    this.changeHandler = null
    this.facets = []
    this.buttonId = null
    this.queryAlert = $('#queryAlert')
    this.qMsg = this.queryAlert.find('.nl-query-message')
  }

  addChangeEventListeners (editor, regexpObj, regexpVal) {
    // Create a new handler function that has access to facetsObject
    this.changeHandler = (event) => this._handleChange(event, editor, regexpObj, regexpVal)

    // Attach the new event listener
    this.element.addEventListener('input', this.changeHandler)
  }

  _handleChange (event, editor, regexpObj, regexpVal) {
    const regexpStr = regexpObj.regexp
    const regexp = new RegExp(regexpStr)
    const button = document.getElementById(this.buttonId)

    if (regexp.test('/' + this.element.value + '/')) {
      this._clearAlert()
      regexpVal.val = this.element.value
      if ('quotes' in regexpObj) {
        const regexpQuotes = regexpObj.quotes
        regexpVal.val = regexpQuotes + this.element.value + regexpQuotes
      }
      button.disabled = false
    } else {
      if (this.element.value === '') {
        this._clearAlert()
      } else {
        let helpMess = 'The entered value must match the following regular expression :' + regexpStr
        if (regexpVal.key === 'float') {
          helpMess = 'The entered value must be a float number.'
        } else if (regexpVal.key === 'integer') {
          helpMess = 'The entered value must be an integer.'
        }
        this._setAlert('The current value has an incorrect format.', 'Matching error', helpMess)
      }
      button.disabled = true
    }
  }

  remove () {
    this._removeFacets()
    this._removeChangeEventListener()
    this._removeElement()
    return null
  }

  _removeChangeEventListener () {
    if (this.element) {
      if (this.changeHandler) {
        this.changeHandler = this.element.removeEventListener('input', this.changeHandler)
      }
    }
  }
}

/**
 * Class to manage a Select object.
 */
class Select extends Element {
  constructor (editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key, sc = false) {
    super()
    this.editor = editor
    this.facetsContainerElement = facetsContainerElement
    this.parentContainerElement = parentContainerElement
    this.elementContainer = elementContainer
    this.element = this._createElement(this.elementContainer, elementId)
    this.allData = data
    this.data = this.allData.key
    this.key = key
    this.changeHandler = null
    this.facets = []
  }

  _selectInSymbolsTable (value) {
    const dropdownItems = []
    this.scdropdown.find('.menu .item').each(function () {
      dropdownItems.push($(this).data('value'))
    })
    if (dropdownItems.includes(value)) {
      // this.sc.dropdown is a jQuery object wrapping the dropdown element
      this.scdropdown.dropdown('set selected', value)
    }
  }

  _createElement (elementContainer, elementId) {
    // Create the facet
    let element = document.createElement('select')
    element.id = elementId
    element.size = '5'
    elementContainer.appendChild(element)
    // Apply select styles
    element = document.getElementById(element.id)
    element.style.marginLeft = '5px'
    element.style.marginRight = '5px'
    return element
  }

  hide () {
    this._setAlert()
    // Hide the facets and 'ok' button as they are no longer needed
    // set the facets container back to be hidden
    this.facetsContainerElement.style.display = 'none'

    // Refocus the CodeMirror editor to keep the cursor visible in the textarea
    this.editor.focus()
  }

  _removeChangeEventListener () {
    if (this.element) {
      if (this.changeHandler) {
        this.changeHandler = this.element.removeEventListener('change', this.changeHandler)
      }
    }
  }

  _removeElement () {
    this.element.innerHTML = ''
    this.element.id = ''
    this.element = this.element.remove()
  }

  _removeFacets () {
    if (this.facets.length) {
      for (let f of this.facets) {
        if (f) {
          f = f.remove()
        }
      }
      this.facets = []
    }
  }
}

/**
 * Class to manage categories select element.
 */
export class CategoriesSelect extends Select {
  constructor (editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key) {
    super(editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key)
    this.valuesFacet = null
  }

  //  addChangeEventListeners (editor, valuesSelect) {
  addChangeEventListeners (editor, refData) {
    this.changeHandler = (event) => this._handleClick(event, editor, refData)

    // Attach the new event listener
    this.element.addEventListener('change', this.changeHandler)
  }

  _setValuesFacet (key = false) {
    const newFacet = new Facet(
      this.editor,
      this.facetsContainerElement,
      this.parentContainerElement,
      'rightFacetContainer')
    newFacet.addLabel(this.element.id + 'rightFacetLabel', 'Values', 'rightFacet')
    newFacet.addElement('patterns', 'rightFacet', this.allData)
    newFacet.element.addChangeEventListeners(this.editor, this.facetsContainerElement)
    newFacet.element.fill(key)
    newFacet.element.show(this.editor, this.facetsContainerElement, this.qMsg, this.queryAlert)
    return newFacet
  }

  /**
  * Handles the click event for the left facet.
  * @param {Event} event the event object associated with the click in the left facet
  */
  _handleClick (event, editor, refData) {
    // Retrieve the selected option
    const selectedKey = event.target.value
    this._removeFacets()

    if (selectedKey) {
      if (selectedKey === 'All') {
        //        const dataSelect = this.allData
        this.facets.push(this._setValuesFacet())
      } else if (this.allData[selectedKey]) {
        this.facets.push(this._setValuesFacet(selectedKey))
      }
    }
    editor.focus()
  }

  /**
  * Displays the facets.
  * @param {Object} facetsObject the categories to be displayed in the facets and their values
  */
  fill (facetsObject) {
    // Add 'All' option to the left facet
    const deselectOption = document.createElement('option')
    deselectOption.textContent = 'All'
    this.element.appendChild(deselectOption)

    // Add facetsObject keys to left facet
    for (const key of Object.keys(facetsObject)) {
      const option = document.createElement('option')
      option.textContent = key
      this.element.appendChild(option)
    }
  }

  remove () {
    this._removeFacets()
    this._removeChangeEventListener()
    this._removeElement()
    return null
  }
}

/**
 * Class to manage values select element.
 */
export class RegexpSelect extends Select {
  constructor (editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key, sc) {
    super(editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key, sc)
    this.position = -1
    this.buttonId = null
  }

  updateButtonId (buttonId) {
    this.buttonId = buttonId
  }

  updatePosition (pos) {
    this.position = pos
  }

  addChangeEventListeners (editor, facetsContainerElement, tab) {
    // Create a new handler function that has access to facetsObject
    this.changeHandler = (event) => this._handleClick(event, editor, facetsContainerElement, tab)

    // Attach the new event listener
    this.element.addEventListener('change', this.changeHandler)
  }

  /**
  * Handles the click event for the right facet.
  * @param {Event} event the event object associated with the click in the right facet
  */
  _handleClick (event, editor, facetsContainerElement, tab) {
    const selectedValue = this.element.value

    // put the value as selected in the symbols table as well
    this._selectInSymbolsTable(selectedValue)

    tab[this.position] = selectedValue

    const button = document.getElementById(this.buttonId)
    if (tab.every(value => Boolean(value))) {
      button.disabled = false
    } else {
      button.disabled = true
    }
    editor.focus()
  }

  /**
  * Displays the facets.
  * @param {Object} ruleObject the categories to be displayed in the facets and their values
  */
  fill (keyToFillSelect) {
    const keyData = this.allData[keyToFillSelect]

    // Add ruleObject keys to left facet
    for (const item of keyData.values) {
      const curKey = item.slice(1, -1)
      if ((curKey in this.allData) && ('params' in this.allData[curKey]) && (this.allData[curKey].params === 'expandable')) {
        const optgroup = document.createElement('optgroup')
        optgroup.label = curKey
        for (const i of this.allData[curKey].values) {
          const option = document.createElement('option')
          option.textContent = i
          optgroup.appendChild(option)
        }
        this.element.appendChild(optgroup)
      } else {
        const option = document.createElement('option')
        option.textContent = item
        this.element.appendChild(option)
      }
    }
  }

  remove () {
    this._removeFacets()
    this._removeChangeEventListener()
    this._removeElement()
    return null
  }
}

/**
 * Class to manage patterns select element.
 */
export class PatternsSelect extends Select {
  addChangeEventListeners (editor, facetsContainerElement, inPattern = false) {
    // Create a new handler function that has access to facetsObject
    this.changeHandler = (event) => this._handleClick(editor, facetsContainerElement, inPattern)

    // Attach the new event listener
    this.element.addEventListener('change', this.changeHandler)
  }

  /**
  * Handles the click event for the right facet.
  * @param {Event} event the event object associated with the click in the right facet
  */
  _handleClick (editor, facetsContainerElement, inPattern) {
    this._removeFacets()

    // get selected value
    const selectedValue = this.element.value

    // put the value as selected in the symbols table as well
    this._selectInSymbolsTable(selectedValue)

    // get the cursor position in the CodeMirror editor
    const cursorPos = editor.getCursor()

    // calculate the end position based on the length of the inserted value
    const endPos = { line: cursorPos.line, ch: cursorPos.ch + selectedValue.length }

    if (selectedValue) {
      const selectedValueToKey = selectedValue.slice(1, -1)

      // check if the selected value is a key in data and has a key "regexp"
      if (Object.hasOwn(this.allData, selectedValueToKey) && Object.hasOwn(this.allData[selectedValueToKey], 'regexp')) {
        const regexpObj = this.allData[selectedValueToKey]
        const regexpVal = { key: selectedValueToKey, val: '' }
        const newInput = new Facet(
          this.editor,
          this.facetsContainerElement,
          this.parentContainerElement,
          this.element.id + '_input_container')
        newInput.addLabel(this.element.id + '_input_label', 'Enter the value :', this.element.id + '_input')
        newInput.addElement('regexp', this.element.id + '_input', this.allData, selectedValueToKey)
        newInput.element.addChangeEventListeners(this.editor, regexpObj, regexpVal)
        newInput.element.show(this.editor, this.facetsContainerElement, this.qMsg, this.queryAlert)

        const buttonNew = new Facet(
          this.editor,
          this.facetsContainerElement,
          this.parentContainerElement,
          this.element.id + '_button_container')
        //        buttonNew.addLabel(null, '', this.element.id + '_buttonFacet')
        buttonNew.addElement('valueButton', this.element.id + '_button', this.allData)
        buttonNew.element.addClickEventListeners(this.editor, this.facetsContainerElement, regexpVal, inPattern)
        buttonNew.element.show(this.editor, this.facetsContainerElement, this.qMsg, this.queryAlert)

        newInput.element.updateButtonId(buttonNew.element.element.id)

        this.facets.push(newInput, buttonNew)

        editor.focus()
      } else {
        if (inPattern) {
          //          var selectedRange = editor.getSelection()
          editor.replaceSelection(selectedValue)
        } else {
          // insert the selected value at the current cursor position
          editor.replaceRange(selectedValue, cursorPos)

          // Move cursor to end of inserted value
          editor.setCursor(endPos)
        }
      }
    }
  }

  /**
  * Displays the facets.
  * @param {Object} ruleObject the categories to be displayed in the facets and their values
  */
  fill (keyToFillSelect = false) {
    const keysToFillSelect = []
    if (keyToFillSelect) {
      keysToFillSelect.push(keyToFillSelect)
    } else {
      for (const k in this.allData) {
        keysToFillSelect.push(k)
      }
    }

    for (const selectKey of keysToFillSelect) {
      const keyData = this.allData[selectKey]

      // Add ruleObject keys to left facet
      for (const item of keyData.values) {
        const curKey = item.slice(1, -1)
        if ((curKey in this.allData) && ('params' in this.allData[curKey]) && (this.allData[curKey].params === 'expandable')) {
          const optgroup = document.createElement('optgroup')
          optgroup.label = curKey
          for (const i of this.allData[curKey].values) {
            const option = document.createElement('option')
            option.textContent = i
            optgroup.appendChild(option)
          }
          this.element.appendChild(optgroup)
        } else {
          const option = document.createElement('option')
          option.textContent = item
          this.element.appendChild(option)
        }
      }
    }
  }

  remove () {
    this._removeFacets()
    this._removeChangeEventListener()
    this._removeElement()
    return null
  }
}
