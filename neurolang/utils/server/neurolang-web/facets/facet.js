import $ from '../jquery-bundler'

/**
 * Class to manage facets.
 */
export class Facet {
  /**
  * Handles the click event for the right facet.
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
    //    console.log(" ")
    //    console.log("________________________________")
    //    console.log("___ Facet.constructor()___")
    //    console.log("this.editor :", this.editor)
    //    console.log("this.facetsContainerElement :", this.facetsContainerElement)
    //    console.log("this.parentContainerElement :", this.parentContainerElement)
    //    console.log("this.containerId :", this.containerId)
  }

  addLabel (labelId, labelText, facetId) {
    this.label = null
    if (labelId) {
      this.label = new Label(this.container.element, labelId, labelText, facetId)
      //      console.log(" ")
      //      console.log("________________________________")
      //      console.log("___ Facet.addLabel()___")
      //      console.log("labelId :", labelId)
      //      console.log("labelText :", labelText)
      //      console.log("facetId :", facetId)
    }
  }

  addElement (elementType, elementId, data, key = false) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ Facet.addElement()___')
    console.log('elementType :', elementType)
    console.log('elementId :', elementId)
    console.log('data :', data)
    console.log('key :', key)
    this.element = null

    if (elementType === 'categories') {
      this.element = new CategoriesSelect(this.editor, this.facetsContainerElement, this.parentContainerElement, this.container.element, elementId, data, key)
    } else if (elementType === 'values') {
      this.element = new ValuesSelect(this.editor, this.facetsContainerElement, this.parentContainerElement, this.container.element, elementId, data, key)
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

class Label {
  constructor (labelContainerElement, labelId, labelText, labelForElementId) {
    // Create label for the facet
    this.element = document.createElement('label')
    this.element.id = labelId
    this.element.htmlFor = labelForElementId
    this.element.textContent = labelText
    // Append the label and facet to the first 'facet' div
    labelContainerElement.appendChild(this.element)
    // Apply select styles
    // let element = document.getElementById(this.element.id)
    // element.style.padding = '5px 15px'
    // element.style.alignSelf = 'center'
    this.element.style.marginLeft = '5px'
    this.element.style.marginRight = '5px'
  }

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

class Container {
  constructor (parentContainerElement, containerId) {
    // Create the facet container div
    this.element = document.createElement('div')
    this.element.className = 'facet'
    this.element.id = containerId
    // Append the first 'facet' div to the container
    parentContainerElement.appendChild(this.element)
  }

  remove () {
    // Clear container
    this.element.className = ''
    this.element.id = ''
    // Remove container
    this.element = this.element.remove()
    return null
  }
}

class Element {
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
  //  _setAlert (style, content, header, help) {
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

    console.log('mess :', mess)
    console.log('mess type :', typeof mess)
    console.log('alert :', alert)
    console.log('alert type :', typeof alert)

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
    //    this.queryAlert = $('#queryAlert')
    //    this.qMsg = this.queryAlert.find('.nl-query-message')
    //    this.generatorFacetId = generatorFacetId
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
    console.log(' ')
    console.log('________________________________')
    console.log('___ AggregateButton._handleClick()___')
    console.log('patternUnit :', patternUnit)
    console.log('patternSep :', patternSep)
    console.log('inPattern :', inPattern)
    console.log('this.allData :', this.allData)
    console.log('this.key :', this.key)

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

      // check if the selected value starts with '<' and ends with '>'
      //      if (selectedValue == "<identifier_regexp>") {
      //        cursorPos = editor.getCursor()
      //
      //        endPos = { line: cursorPos.line, ch: cursorPos.ch }
      //        cursorPos.ch = cursorPos.ch - selectedValue.length
      //        // select the text that was just inserted
      //        editor.setSelection(cursorPos, endPos)
      //      }
    }

    this.hide()

    // Hide the facets and 'ok' button as they are no longer needed
    // set the facets container back to be hidden
    //    facetsContainerElement.style.display = 'none'

    // Refocus the CodeMirror editor to keep the cursor visible in the textarea
    //    editor.focus()
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
//  constructor (editor, facetsContainerElement, parentContainerElement, containerDiv, facetId, data, key) {
//    super(editor, facetsContainerElement, parentContainerElement, containerDiv, facetId, data, key)
//  }

  addClickEventListeners (editor, facetsContainerElement, valueToWrite, inPattern = false) {
    // Create a new handler function that has access to facetsObject
    this.clickHandler = (event) => this._handleClick(editor, facetsContainerElement, valueToWrite, inPattern)

    // Attach the new event listener
    this.element.addEventListener('click', this.clickHandler)
  }

  _handleClick (editor, facetsContainerElement, valueToWrite, inPattern = false) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ ValueButton._handleClick()___')
    // Get the value from the input
    //    console.log("valuesToWrite :", valuesToWrite)
    //    console.log("valueToWrite :", valueToWrite.val)

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

        // check if the selected value starts with '<' and ends with '>'
        //      if (selectedValue == "<identifier_regexp>") {
        //        cursorPos = editor.getCursor()

        //        endPos = { line: cursorPos.line, ch: cursorPos.ch }
        //        cursorPos.ch = cursorPos.ch - selectedValue.length
        //        // select the text that was just inserted
        //        editor.setSelection(cursorPos, endPos)
      }
    }

    this.hide()

    // Hide the facets and 'ok' button as they are no longer needed
    // set the facets container back to be hidden
    //    facetsContainerElement.style.display = 'none'

    // Refocus the CodeMirror editor to keep the cursor visible in the textarea
    //    editor.focus()
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
    console.log('________________________________')
    console.log('___ Input.constructor()___')
    console.log('this.editor :', this.editor)
    console.log('this.facetsContainerElement :', this.facetsContainerElement)
    console.log('this.parentContainerElement :', this.parentContainerElement)
    console.log('this.elementContainer :', this.elementContainer)
    console.log('elementId :', elementId)
    this.allData = data
    this.key = key
    console.log('this.allData :', this.allData)
    console.log('this.key :', this.key)
    this.element = this._createElement(this.elementContainer, elementId)
    this.facets = []
    this.buttonId = null
    console.log(' ')
    console.log('this.element :', this.element)
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

export class NumberInput extends Input {
  constructor (editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key) {
    //    super()
    super(editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key)

    //    this.editor = editor
    //    this.facetsContainerElement = facetsContainerElement
    //    this.parentContainerElement = parentContainerElement
    //    this.elementContainer = elementContainer
    //    this.element = this._createElement(this.elementContainer, elementId)
    //    this.allData = data
    //    this.key = key
    //    console.log("$ PatternsSelectInput.constructor() $")
    //    console.log("editor :", editor)
    //    console.log("editor :", editor)
    //    console.log("editor :", editor)
    //    console.log("editor :", editor)

    //    console.log("this.data :", this.data)
    //    console.log("this.key :", this.key)
    //    this.facets = []
    //    console.log("this.element :", this.element)

    this.changeHandler = null
    //    this.queryAlert = $('#queryAlert')
    //    this.qMsg = this.queryAlert.find('.nl-query-message')

    //    this.buttonId = null
  }

  addChangeEventListeners (editor, inPattern = false) {
    // Create a new handler function that has access to facetsObject
    //    this.changeHandler = (event) => this._handleClick(event, editor, valuesSelect)
    this.changeHandler = (event) => this._handleChange(event, editor, inPattern)

    // Attach the new event listener
    this.element.addEventListener('input', this.changeHandler)
  }

  _handleChange (event, editor, inPattern) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ PatternsSelectInput._handleChange()___')
    console.log('this.allData :', this.allData)
    console.log('this.data :', this.data)
    console.log('this.key :', this.key)

    // Clear the previous facets created if another regexp option was selected before
    this._removeFacets()

    const intValue = parseInt(this.element.value, 10)
    console.log(' ')
    console.log('intValue :', intValue)

    //    const int_regexp = new RegExp("^-?\\d+$")

    if (!isNaN(intValue)) {
      this._clearAlert()
      const selectedValues = new Array(intValue).fill(null)
      console.log('selectedValues :', selectedValues)
      const buttonId = this.element.id + '_button_element'
      for (let step = 0; step < intValue; step++) {
        console.log(' ')
        console.log('  current step :', step)
        const unitKey = this.allData[this.key].unit.slice(1, -1)
        console.log('  unitKey :', unitKey)

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
        console.log(' ')
        patternFacetNew.element.fill(unitKey)
        patternFacetNew.element.show(this.editor, this.facetsContainerElement, this.qMsg, this.queryAlert)

        // Add to facets
        this.facets.push(patternFacetNew)
      }

      console.log('Button :')
      console.log('  Container :')
      console.log('    id :', (this.element.id + '_button_container'))
      const buttonNew = new Facet(
        this.editor,
        this.facetsContainerElement,
        this.parentContainerElement,
        this.element.id + '_button_container')

      console.log('  Element :')
      console.log('    type : \'aggregateButton\'')
      console.log('    id :', buttonId)
      console.log('    this.allData :', this.allData)
      console.log('    this.key :', this.key)
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

    //    editor.focus()
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

export class RegexpInput extends Input {
  constructor (editor, facetsContainerElement, parentContainerElement, containerDiv, elementId, data, key) {
    super(editor, facetsContainerElement, parentContainerElement, containerDiv, elementId, data, key)
    this.changeHandler = null
    this.facets = []
    this.buttonId = null
    this.queryAlert = $('#queryAlert')
    this.qMsg = this.queryAlert.find('.nl-query-message')
  }

  addChangeEventListeners (editor, regexpObj, regexpVal) {
    //    console.log(" ")
    //    console.log("___in InputRegExp.addChangeEventListeners()___")
    // Create a new handler function that has access to facetsObject
    //    this.changeHandler = (event) => this._handleClick(event, editor, valuesSelect)
    this.changeHandler = (event) => this._handleChange(event, editor, regexpObj, regexpVal)

    // Attach the new event listener
    this.element.addEventListener('input', this.changeHandler)
  }

  _handleChange (event, editor, regexpObj, regexpVal) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ RegexpInput._handleChange()___')

    console.log(' ')
    //    const regexpStr = regexpObj['regexp']
    const regexpStr = regexpObj.regexp
    console.log('regexpStr :', regexpStr)
    console.log('regexpVal :', regexpVal)
    console.log('this.buttonId :', this.buttonId)
    console.log('input value : *' + this.element.value + '*')
    //    console.log("this.element.value :", this.element.value)
    const regexp = new RegExp(regexpStr)
    const button = document.getElementById(this.buttonId)

    let valueMatches = false

    if (regexpStr === 'float') {
      valueMatches = /^-?\d+(\.\d+)?$/.test(this.element.value) && (this.element.value).includes('.')
    } else if (regexpStr === 'float') {
      const num = parseInt(this.element.value, 10)
      valueMatches = !isNaN(num)
    } else {
      console.log('entered value :', this.element.value)
      valueMatches = regexp.test('/' + this.element.value + '/')
      console.log('test result :', valueMatches)
    }
    //    if (int_regexp.test(this.element.value) || !(this.element.value).length) {
    //      this._clearAlert()
    if (regexp.test('/' + this.element.value + '/')) {
      //      console.log("match ok")
      this._clearAlert()
      regexpVal.val = this.element.value
      if ('quotes' in regexpObj) {
      //        const regexpQuotes = regexpObj['quotes']
        const regexpQuotes = regexpObj.quotes
        regexpVal.val = regexpQuotes + this.element.value + regexpQuotes
      }
      button.disabled = false
    } else {
      console.log('this.element.value :', this.element.value)
      console.log('(this.element.value).length :', (this.element.value).length)
      //      console.log('this.element.value == \'\' :', this.element.value == '')
      console.log('this.element.value === \'\' :', this.element.value === '')
      if (this.element.value === '') {
        console.log('clearing alert...')
        this._clearAlert()
        console.log('clearing alert...')
      } else {
        //      console.log("match not ok")
        //      this.qMsg.text("Unvalid string.")
        //      this.queryAlert.show()
        let helpMess = 'The entered value must match the following regular expression :' + regexpStr
        console.log('regexpVal.key :', regexpVal.key)
        //        console.log('regexpVal.key == \'float\' :', regexpVal.key == 'float')
        console.log('regexpVal.key === \'float\' :', regexpVal.key === 'float')
        //        console.log('regexpVal.key == \'integer\' :', regexpVal.key == 'integer')
        console.log('regexpVal.key === \'integer\' :', regexpVal.key === 'integer')
        if (regexpVal.key === 'float') {
          console.log('float ok')
          helpMess = 'The entered value must be a float number.'
          console.log(helpMess)
        } else if (regexpVal.key === 'integer') {
          console.log('integer ok')
          helpMess = 'The entered value must be an integer.'
          console.log(helpMess)
        }
        console.log('out of if')
        console.log(helpMess)
        this._setAlert('The current value has an incorrect format.', 'Matching error', helpMess)
      }
      button.disabled = true
    }
    //    console.log("button.disabled apres :", button.disabled)

    //    editor.focus()
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
        //        console.log("this.leftFacetNew :")
        this.changeHandler = this.element.removeEventListener('input', this.changeHandler)
        //        console.log("this.changeHandler :", this.changeHandler)
        //        console.log("this.element :", this.element)
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
    //    console.log(" ")
    //    console.log("___in Select constructor()___")
    //    console.log("in Select constructor")
    //    this.sc = sc
    this.editor = editor
    this.facetsContainerElement = facetsContainerElement
    this.parentContainerElement = parentContainerElement
    this.elementContainer = elementContainer
    this.element = this._createElement(this.elementContainer, elementId)
    this.allData = data
    this.data = this.allData.key
    console.log(' ')
    console.log('this.data 1 :', this.allData.key)
    console.log('this.data 2 :', this.allData[key])
    this.key = key
    this.changeHandler = null
    //    console.log(" ")
    //    console.log("this.data :", this.data)
    this.facets = []
    //    console.log("Select constructor ok :")
  }

  _selectInSymbolsTable (value) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ Select._selectInSymbolsTable()___')
    //    console.log("this.sc :", this.sc)
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
    //    console.log("Select element :", element)
    element.id = elementId
    //    console.log("Select id :", element.id)
    element.size = '5'
    //    console.log("Select size :", element.size)
    elementContainer.appendChild(element)

    // Apply select styles
    element = document.getElementById(element.id)
    //    element.style.padding = '5px 15px'
    //    element.style.alignSelf = 'center'
    element.style.marginLeft = '5px'
    element.style.marginRight = '5px'

    return element
  }

  //  show (editor, facetsContainerElement, mess, alert) {
  //    // Display the facets container
  //    // overrides the 'display: none;' from the CSS.
  //    facetsContainerElement.style.display = 'flex'
  //
  //    console.log("mess :", mess)
  //    console.log("mess type :",typeof mess)
  //    console.log("alert :", alert)
  //    console.log("alert type :",typeof alert)
  //
  //    // This will always run, whether there are tokens or not
  //    editor.on('cursorActivity', () => {
  //      if (mess) {
  //        mess.empty()
  //      }
  //      if (alert) {
  //        alert.hide()
  //      }
  //      // set the facets container back to be hidden
  //      facetsContainerElement.style.display = 'none'
  //    })
  //  }

  hide () {
    this._setAlert()
    // Hide the facets and 'ok' button as they are no longer needed
    // set the facets container back to be hidden
    this.facetsContainerElement.style.display = 'none'

    // Refocus the CodeMirror editor to keep the cursor visible in the textarea
    this.editor.focus()
  }

  //  remove () {
  //    this._removeFacets()
  //    this._removeChangeEventListener ()
  //    this._removeElement()
  //    return null
  //  }

  _removeChangeEventListener () {
    if (this.element) {
      if (this.changeHandler) {
        this.changeHandler = this.element.removeEventListener('change', this.changeHandler)
      }
    }
  }

  _removeElement () {
    // Clear element
    this.element.innerHTML = ''
    this.element.id = ''
    //    this.element = document.createElement(this.elementType)
    //    console.log("this.element before :", this.element)
    // Append the label and facet to the first 'facet' div
    this.element = this.element.remove()
  }

  //  _setValuesFacet (dataObject) {
  //    const newFacet = new Facet(
  //      this.editor,
  //      this.facetsContainerElement,
  //      this.parentContainerElement,
  //      'rightFacetContainer')
  //    //      console.log(" ")
  //    //      console.log("newFacet :", newFacet)
  //    //      console.log("this.element.id :", this.element.id)
  //    //      console.log("newFacet.element.element.id :", newFacet.element.element.id)
  //    newFacet.addLabel(this.element.id + 'rightFacetLabel', 'Values', 'rightFacet')
  //    newFacet.addElement('values', 'rightFacet', dataObject)
  //    newFacet.element.addChangeEventListeners(this.editor, this.facetsContainerElement)
  //    newFacet.element.fill(dataObject, 'values')
  //    newFacet.element.show(this.editor, this.facetsContainerElement, this.qMsg, this.queryAlert)
  //    return newFacet
  //  }

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
    console.log(' ')
    console.log('________________________________')
    console.log('___ CategoriesSelect.addChangeEventListeners()___')
    console.log('refData :', refData)
    this.changeHandler = (event) => this._handleClick(event, editor, refData)

    // Attach the new event listener
    this.element.addEventListener('change', this.changeHandler)
  }

  _setValuesFacet (dataObject, refData) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ CategoriesSelect._setValuesFacet()___')
    console.log('this.allData :', this.allData)
    console.log('dataObject :', dataObject)
    console.log('refData :', refData)
    const newFacet = new Facet(
      this.editor,
      this.facetsContainerElement,
      this.parentContainerElement,
      'rightFacetContainer')
    //      console.log(" ")
    //      console.log("newFacet :", newFacet)
    //      console.log("this.element.id :", this.element.id)
    //      console.log("newFacet.element.element.id :", newFacet.element.element.id)
    newFacet.addLabel(this.element.id + 'rightFacetLabel', 'Values', 'rightFacet')
    //      newFacet.addElement('values', 'rightFacet', this.allData[selectedKey])
    newFacet.addElement('patterns', 'rightFacet', this.allData)
    //      newFacet.element.addChangeEventListeners(this.editor, this.facetsContainerElement)
    newFacet.element.addChangeEventListeners(this.editor, this.facetsContainerElement, refData)
    //    newFacet.element.fill(dataObject, 'values')
    newFacet.element.fill(dataObject, refData)
    newFacet.element.show(this.editor, this.facetsContainerElement, this.qMsg, this.queryAlert)
    return newFacet
  }

  /**
  * Handles the click event for the left facet.
  * @param {Event} event the event object associated with the click in the left facet
  */
  //  _handleClick (event, editor, valuesSelect) {
  _handleClick (event, editor, refData) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ CategoriesSelect._handleClick()___')
    console.log('this.allData :', this.allData)
    console.log('refData :', refData)

    // Retrieve the selected option
    const selectedKey = event.target.value
    console.log('selectedKey :', selectedKey)

    // Clear previous options
    //    while (valuesSelect.element.firstChild) {
    //      valuesSelect.element.removeChild(valuesSelect.element.firstChild)
    //    }
    //    while (this.rightFacet.element.firstChild) {
    //      this.rightFacet.element.removeChild(this.rightFacet.element.firstChild)
    //    }

    if (this.valuesFacet) {
      this.valuesFacet = this.valuesFacet.remove()
    }

    if (selectedKey) {
      //      let valuesToDisplay = null

      if (selectedKey === 'All') {
        console.log('selectedKey === \'All\'')
        const values = []
        const dataSelect = this.allData
        for (const key of Object.keys(dataSelect)) {
          for (const value of dataSelect[key]) {
            values.push(value)
          }
        }
        console.log('*** Before call this._setValuesFacet()')
        console.log('values :', values)
        console.log('refData :', refData)
        this.valuesFacet = this._setValuesFacet(values, refData)
      } else if (this.allData[selectedKey]) {
        console.log('this.allData[selectedKey] true')
        //        console.log(" ")
        //        console.log("-- not All --")
        //        console.log("selectedKey :", selectedKey)
        //        console.log("this.data[selectedKey] :", this.data[selectedKey])
        //    for (const value of this.data[selectedKey]) {
        //      const option = document.createElement('option')
        //      option.value = value
        //      option.textContent = value
        //      // this.rightFacet.element.appendChild(option)
        //      // valuesSelect.element.appendChild(option)
        //      // this.valuesFacet.element.element.appendChild(option)
        //    }
        console.log('*** Before call this._setValuesFacet()')
        console.log('this.allData[selectedKey] :', this.allData[selectedKey])
        console.log('refData :', refData)
        //        this.valuesFacet = this._setValuesFacet(this.allData[selectedKey], refData)
        this.valuesFacet = this._setValuesFacet(selectedKey, refData)
      }
    }
    editor.focus()
  }

  /**
  * Displays the facets.
  * @param {Object} facetsObject the categories to be displayed in the facets and their values
  */
  fill (facetsObject) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ CategoriesSelect.fill()___')
    console.log('facetsObject :', facetsObject)

    // Add 'All' option to the left facet
    const deselectOption = document.createElement('option')
    deselectOption.textContent = 'All'
    //    console.log("this.leftFacetNew :", this.leftFacetNew)
    this.element.appendChild(deselectOption)

    // Add facetsObject keys to left facet
    for (const key of Object.keys(facetsObject)) {
      const option = document.createElement('option')
      option.textContent = key
      this.element.appendChild(option)
    }
  }

  remove () {
    if (this.valuesFacet) {
      this.valuesFacet.remove()
    }
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
    //    this.generatorFacetId = generatorFacetId
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
    console.log(' ')
    console.log('________________________________')
    console.log('___ RegexpSelect._handleClick()___')
    console.log('tab avant :', tab)
    //    console.log("patternSep :", patternSep)
    //    console.log("inPattern :", inPattern)
    console.log('this.allData :', this.allData)
    console.log('this.key :', this.key)
    //    console.log("this.sc :", this.sc)
    const selectedValue = this.element.value

    // put the value as selected in the symbols table as well
    this._selectInSymbolsTable(selectedValue)

    tab[this.position] = selectedValue
    console.log('tab apres :', tab)

    const button = document.getElementById(this.buttonId)
    if (tab.every(value => Boolean(value))) {
      button.disabled = false
    } else {
      button.disabled = true
    }

    //    const dropdownItems = []
    //    this.sc.dropdown.find('.menu .item').each(function () {
    //      dropdownItems.push($(this).data('value'))
    //    })
    //    if (dropdownItems.includes(selectedValue)) {
    // this.sc.dropdown is a jQuery object wrapping the dropdown element
    //      this.sc.dropdown.dropdown('set selected', selectedValue)
    //    }

    //    if (selectedValue && selectedValue !== 'All') {
    // get the cursor position in the CodeMirror editor
    //      const cursorPos = editor.getCursor()

    // insert the selected value at the current cursor position
    //      editor.replaceRange(selectedValue, cursorPos)

    // calculate the end position based on the length of the inserted value
    //      const endPos = { line: cursorPos.line, ch: cursorPos.ch + selectedValue.length }

    // check if the selected value starts with '<' and ends with '>'
    //      if (selectedValue.startsWith('<') && selectedValue.endsWith('>')) {
    //    // select the text that was just inserted
    //        editor.setSelection(cursorPos, endPos)
    //      } else {
    //    // Move cursor to end of inserted value
    //        editor.setCursor(endPos)
    //      }
    //    }

    // Hide the facets and 'ok' button as they are no longer needed
    // set the facets container back to be hidden
    //    facetsContainerElement.style.display = 'none'

    // Refocus the CodeMirror editor to keep the cursor visible in the textarea
    editor.focus()
  }

  /**
  * Displays the facets.
  * @param {Object} ruleObject the categories to be displayed in the facets and their values
  */
  fill (keyToFillSelect) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ PatternsSelect.fill()___')
    console.log('keyToFillSelect :', keyToFillSelect)
    const keyData = this.allData[keyToFillSelect]
    console.log('keyData :', keyData)

    // Add ruleObject keys to left facet
    for (const item of keyData.values) {
      console.log(' ')
      console.log('item :', item)
      const curKey = item.slice(1, -1)
      console.log('curKey :', curKey)
      //      if ((curKey in this.allData) && ('params' in this.allData[curKey]) && (this.allData[curKey]['params'] == 'expandable')) {
      //      if ((curKey in this.allData) && ('params' in this.allData[curKey]) && (this.allData[curKey].params == 'expandable')) {
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
    if (this.valuesFacet) {
      this.valuesFacet.remove()
    }
    this._removeFacets()
    //    console.log(" ")
    //    console.log("this.buttonFacet :", this.buttonFacet)
    //    if (this.buttonFacet) {
    //      this.buttonFacet.element.remove()
    //    }
    this._removeChangeEventListener()
    this._removeElement()
    return null
  }
}

/**
 * Class to manage values select element.
 */
export class ValuesSelect extends Select {
//  constructor (editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key, sc) {
//    super(editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key, sc)
//  }

  addChangeEventListeners (editor, facetsContainerElement, refData, inPattern = false) {
    // Create a new handler function that has access to facetsObject
    this.changeHandler = (event) => this._handleClick(editor, facetsContainerElement, refData, inPattern)

    // Attach the new event listener
    this.element.addEventListener('change', this.changeHandler)
  }

  /**
  * Handles the click event for the right facet.
  * @param {Event} event the event object associated with the click in the right facet
  */
  _handleClick (editor, facetsContainerElement, refData, inPattern) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ ValuesSelect._handleClick()___')
    console.log('this.allData :', this.allData)
    console.log('refData :', refData)
    console.log('inPattern :', inPattern)

    //    this._removeFacets()

    // get selected value
    const selectedValue = this.element.value
    console.log('selectedValue :', selectedValue)

    // put the value as selected in the symbols table as well
    this._selectInSymbolsTable(selectedValue)

    // get the cursor position in the CodeMirror editor
    const cursorPos = editor.getCursor()

    // calculate the end position based on the length of the inserted value
    const endPos = { line: cursorPos.line, ch: cursorPos.ch + selectedValue.length }

    if (selectedValue && selectedValue !== 'All') {
      const selectedValueToKey = selectedValue.slice(1, -1)

      // check if the selected value is a key in data and has a key "regexp"
      //      if (refData.hasOwnProperty(selectedValueToKey) && refData[selectedValueToKey].hasOwnProperty('regexp')) {
      if (Object.hasOwn(refData, selectedValueToKey) && Object.hasOwn(refData[selectedValueToKey], 'regexp')) {
        //        Object.hasOwn(refData[selectedValueToKey], 'regexp')
        console.log(' ')
        console.log('selectedValueToKey in refdata')
        const regexpObj = refData[selectedValueToKey]
        console.log('regexpObj :', regexpObj)
        const regexpVal = { key: selectedValueToKey, val: '' }
        const newInput = new Facet(
          this.editor,
          this.facetsContainerElement,
          this.parentContainerElement,
          this.element.id + '_input_container')
        newInput.addLabel(this.element.id + '_input_label', 'Enter the value :', this.element.id + '_input')
        newInput.addElement('regexp', this.element.id + '_input', this.allData, selectedValueToKey)
        //        newInput.element.addChangeEventListeners(this.editor, regexpObj, regexpVal)
        newInput.element.show(this.editor, this.facetsContainerElement, this.qMsg, this.queryAlert)

        const buttonNew = new Facet(
          this.editor,
          this.facetsContainerElement,
          this.parentContainerElement,
          this.element.id + '_button_container')
        buttonNew.addElement('valueButton', this.element.id + '_button', this.allData)
        buttonNew.element.addClickEventListeners(this.editor, this.facetsContainerElement, regexpVal, inPattern)
        buttonNew.element.show(this.editor, this.facetsContainerElement, this.qMsg, this.queryAlert)

        //        console.log("button id :", buttonNew.element.element.id)
        newInput.element.updateButtonId(buttonNew.element.element.id)

        this.facets.push(newInput, buttonNew)
        //        this.facets.push(newInput)

        for (const item of this.facets) {
          console.log('elt : ', item.element)
          console.log('elt type : ', item.element.element.nodeName)
        }

        editor.focus()
      } else {
        // Previous version
        //      // get the cursor position in the CodeMirror editor
        // //      const cursorPos = editor.getCursor()
        //
        //      // insert the selected value at the current cursor position
        //      editor.replaceRange(selectedValue, cursorPos)
        //
        //      // calculate the end position based on the length of the inserted value
        // //      const endPos = { line: cursorPos.line, ch: cursorPos.ch + selectedValue.length }
        //
        //      // check if the selected value starts with '<' and ends with '>'
        //      if (selectedValue.startsWith('<') && selectedValue.endsWith('>')) {
        //        // select the text that was just inserted
        //        editor.setSelection(cursorPos, endPos)
        //      } else {
        //        // Move cursor to end of inserted value
        //        editor.setCursor(endPos)
        //      }

        if (inPattern) {
          //          var selectedRange = editor.getSelection()
          editor.replaceSelection(selectedValue)
        } else {
          // get the cursor position in the CodeMirror editor
          // const cursorPos = editor.getCursor()

          // insert the selected value at the current cursor position
          editor.replaceRange(selectedValue, cursorPos)

          // calculate the end position based on the length of the inserted value
          // const endPos = { line: cursorPos.line, ch: cursorPos.ch + selectedValue.length }

          // Move cursor to end of inserted value
          editor.setCursor(endPos)
        }
      }
    }

    // Hide the facets and 'ok' button as they are no longer needed
    // set the facets container back to be hidden
    //    facetsContainerElement.style.display = 'none'

    // Refocus the CodeMirror editor to keep the cursor visible in the textarea
    //    editor.focus()
  }

  /**
  * Displays the facets.
  * @param {Object} facetsObject the categories to be displayed in the facets and their values
  */
  //  fill (facetsObject) {
  //    // Add facetsObject keys to left facet
  //    for (let item of facetsObject) {
  //      const option = document.createElement('option')
  //      option.textContent = item
  //      this.element.appendChild(option)
  //    }
  //  }

  fill (keyToFillSelect) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ PatternsSelect.fill()___')
    console.log('keyToFillSelect :', keyToFillSelect)
    const keyData = this.allData[keyToFillSelect]
    console.log('keyData :', keyData)

    // Add ruleObject keys to left facet
    for (const item of keyData.values) {
      console.log(' ')
      console.log('item :', item)
      const curKey = item.slice(1, -1)
      console.log('curKey :', curKey)
      //      if ((curKey in this.allData) && ('params' in this.allData[curKey]) && (this.allData[curKey]['params'] == 'expandable')) {
      //      if ((curKey in this.allData) && ('params' in this.allData[curKey]) && (this.allData[curKey].params == 'expandable')) {
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
//  constructor (editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key, sc) {
//    super(editor, facetsContainerElement, parentContainerElement, elementContainer, elementId, data, key, sc)
//  }

  addChangeEventListeners (editor, facetsContainerElement, refData, inPattern = false) {
    // Create a new handler function that has access to facetsObject
    this.changeHandler = (event) => this._handleClick(editor, facetsContainerElement, refData, inPattern)

    // Attach the new event listener
    this.element.addEventListener('change', this.changeHandler)
  }

  /**
  * Handles the click event for the right facet.
  * @param {Event} event the event object associated with the click in the right facet
  */
  _handleClick (editor, facetsContainerElement, allDataObject, inPattern) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ PatternsSelect._handleClick()___')
    console.log('allDataObject :', allDataObject)
    console.log('this.allData :', this.allData)

    this._removeFacets()

    // get selected value
    const selectedValue = this.element.value
    //    console.log("selectedValue :", selectedValue)

    // put the value as selected in the symbols table as well
    this._selectInSymbolsTable(selectedValue)

    // get the cursor position in the CodeMirror editor
    const cursorPos = editor.getCursor()

    // calculate the end position based on the length of the inserted value
    const endPos = { line: cursorPos.line, ch: cursorPos.ch + selectedValue.length }

    if (selectedValue) {
      const selectedValueToKey = selectedValue.slice(1, -1)

      // check if the selected value is a key in data and has a key "regexp"
      //      if (allDataObject.hasOwnProperty(selectedValueToKey) && allDataObject[selectedValueToKey].hasOwnProperty('regexp')) {
      if (Object.hasOwn(allDataObject, selectedValueToKey) && Object.hasOwn(allDataObject[selectedValueToKey], 'regexp')) {
      //        Object.hasOwn(allDataObject[selectedValueToKey], 'regexp')
        const regexpObj = allDataObject[selectedValueToKey]
        console.log('regexpObj :', regexpObj)
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

        console.log('button id :', buttonNew.element.element.id)
        newInput.element.updateButtonId(buttonNew.element.element.id)

        this.facets.push(newInput, buttonNew)

        for (const item of this.facets) {
          console.log('elt : ', item.element)
          console.log('elt type : ', item.element.element.nodeName)
        }

        editor.focus()
      } else {
        if (inPattern) {
          //          var selectedRange = editor.getSelection()
          editor.replaceSelection(selectedValue)
        } else {
          // get the cursor position in the CodeMirror editor
          // const cursorPos = editor.getCursor()

          // insert the selected value at the current cursor position
          editor.replaceRange(selectedValue, cursorPos)

          // calculate the end position based on the length of the inserted value
          // const endPos = { line: cursorPos.line, ch: cursorPos.ch + selectedValue.length }

          // Move cursor to end of inserted value
          editor.setCursor(endPos)
        }
        //    this.hide()
      }

      //      cursorPos = editor.getCursor()
      //      endPos = { line: cursorPos.line, ch: cursorPos.ch }
      //      cursorPos.ch = cursorPos.ch - selectedValue.length
      // select the text that was just inserted
      //      editor.setSelection(cursorPos, endPos)

      //      else {
      //  // Move cursor to end of inserted value
      //        this.editor.setCursor(endPos)

      //      }
    }
  }

  /**
  * Displays the facets.
  * @param {Object} ruleObject the categories to be displayed in the facets and their values
  */
  fill (keyToFillSelect, refData = false) {
    console.log(' ')
    console.log('________________________________')
    console.log('___ PatternsSelect.fill()___')
    console.log('keyToFillSelect :', keyToFillSelect)
    console.log('this.allData :', this.allData)
    const keyData = this.allData[keyToFillSelect]
    console.log('keyData :', keyData)

    if (!refData) {
      refData = this.allData
    }

    // Add ruleObject keys to left facet
    for (const item of keyData.values) {
      console.log(' ')
      console.log('item :', item)
      const curKey = item.slice(1, -1)
      console.log('curKey :', curKey)
      //      if ((curKey in this.allData) && ('params' in this.allData[curKey]) && (this.allData[curKey]['params'] == 'expandable')) {
      //      if ((curKey in this.allData) && ('params' in this.allData[curKey]) && (this.allData[curKey].params == 'expandable')) {
      //      if ((curKey in this.allData) && ('params' in this.allData[curKey]) && (this.allData[curKey].params === 'expandable')) {
      if ((curKey in refData) && ('params' in refData[curKey]) && (refData[curKey].params === 'expandable')) {
        const optgroup = document.createElement('optgroup')
        optgroup.label = curKey
        for (const i of refData[curKey].values) {
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
