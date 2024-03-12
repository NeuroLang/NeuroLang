import './facets.css'
import Facet from './facet'

/**
 * Class to manage the facets.
 */
export default class FacetsController {
  constructor (editor, sc) {
    this.editor = editor
    this.sc = sc
    this.patterns = null

    // Retrieve the facets container div element by class
    this.containerDiv = document.querySelector('.ui.segment.code-mirror-container.facetsInnerContainer')

    /// HTML related properties
    this.facetsContainerElement = document.getElementById('facetsContainer')

    this.facets = []

    this.inputFacetNew = {
      editor: null,
      facetsContainerElement: null,
      parentContainerElement: null,
      element: null,
      data: null,
      key: null,
      facets: []
    }
  }

  _cleanAllFacets () {
    if (this.facets.length) {
      for (let f of this.facets) {
        if (f) {
          f = f.remove()
        }
      }
      this.facets = []
    }
  }

  updatePatterns (patterns) {
    this.patterns = patterns
  }

  createFacets (dataObject, key = false, type = 'categories', inPattern = false, value = false) {
    // Categories facet
    if (type === 'categories') {
      // Create
      const leftFacetNew = new Facet(
        this.editor,
        this.facetsContainerElement,
        this.containerDiv,
        'leftFacetContainer'
      )
      leftFacetNew.addLabel('leftLabel', 'Categories', 'leftFacet')
      leftFacetNew.addElement('categories', 'leftFacet', dataObject[key])
      leftFacetNew.element.addChangeEventListeners(this.editor, dataObject.rules)
      // Display
      delete dataObject.rules
      leftFacetNew.element.fill()
      leftFacetNew.element.show(this.editor, this.facetsContainerElement)
      // Add to facets
      this.facets.push(leftFacetNew)
    } else if (type === 'patterns') { // Patterns facet
      // Create
      const patternFacetNew = new Facet(
        this.editor,
        this.facetsContainerElement,
        this.containerDiv,
        'pattern_container')
      patternFacetNew.addLabel('pattern_label', 'Patterns', 'pattern_select')
      patternFacetNew.addElement('patterns', 'pattern_select', dataObject, key)
      patternFacetNew.element.addChangeEventListeners(this.editor, this.facetsContainerElement, false, inPattern)
      // Display
      patternFacetNew.element.fill(key)
      patternFacetNew.element.show(this.editor, this.facetsContainerElement)
      // Add to facets
      this.facets.push(patternFacetNew)
    } else if (type === 'number') { // Input facet
      // Create
      const inputFacetNew = new Facet(
        this.editor,
        this.facetsContainerElement,
        this.containerDiv,
        'numberinput_container')
      inputFacetNew.addLabel('numberinput_label', 'Number of ' + key, 'numberinput_input')
      inputFacetNew.addElement('number', 'numberinput_input', dataObject, key, value)
      inputFacetNew.element.addChangeEventListeners(this.editor, inPattern)
      // Display
      inputFacetNew.element.show(this.editor, this.facetsContainerElement)
      // Add to facets
      this.facets.push(inputFacetNew)
    }
  }
}
