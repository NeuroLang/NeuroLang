import './facets.css'
//import { CategoriesSelect } from './facet'
import { Facet } from './facet'
//import { PatternsSelect } from './facet'
//import { ValuesSelect } from './facet'

/**
 * Class to manage the facets.
 */
export class FacetsController {

  constructor (editor, sc) {
    this.editor = editor
    this.sc = sc
    this.patterns = null

    // Retrieve the facets container div element by class
    this.containerDiv = document.querySelector('.ui.segment.code-mirror-container.facetsInnerContainer');

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
    console.log(' ')
    console.log('________________________________')
    console.log('___ FacetsController.createFacets()___')
    console.log("dataObject :", dataObject)
    console.log("this.patterns :", this.patterns)
    console.log("dataObject.rules :", dataObject.rules)
    console.log("dataObject['rules'] :", dataObject['rules'])

    // Categories facet
    if (type === 'categories') {
      // Create
      let leftFacetNew = new Facet(
        this.editor,
        this.facetsContainerElement,
        this.containerDiv,
        'leftFacetContainer'
      )
      leftFacetNew.addLabel('leftLabel', 'Categories', 'leftFacet')
      leftFacetNew.addElement('categories', 'leftFacet', dataObject)
      leftFacetNew.element.addChangeEventListeners(this.editor, dataObject.rules)
      // Display
      delete dataObject.rules
      console.log("After delete :")
      console.log("dataObject :", dataObject)
      console.log("this.patterns :", this.patterns)
      console.log("dataObject.rules :", dataObject.rules)
      console.log("dataObject['rules'] :", dataObject['rules'])
      leftFacetNew.element.fill(dataObject, true)
      leftFacetNew.element.show(this.editor, this.facetsContainerElement)
      // Add to facets
      this.facets.push(leftFacetNew)
    }

    // Patterns facet
    else if (type === 'patterns') {
      // Create
      let patternFacetNew = new Facet(
        this.editor,
        this.facetsContainerElement,
        this.containerDiv,
        'pattern_container')
      patternFacetNew.addLabel('pattern_label', 'Patterns', 'pattern_select')
      patternFacetNew.addElement('patterns', 'pattern_select', dataObject, key)
      patternFacetNew.element.addChangeEventListeners(this.editor, this.facetsContainerElement, this.patterns, inPattern)
      // Display
      patternFacetNew.element.fill(key)
      patternFacetNew.element.show(this.editor, this.facetsContainerElement)
      // Add to facets
      this.facets.push(patternFacetNew)
    }

    // Input facet
    else if (type === 'number') {
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
