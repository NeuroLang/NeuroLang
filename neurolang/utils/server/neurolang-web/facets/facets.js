import './facets.css'
import { CategoriesSelect } from './facet'
import { Facet } from './facet'
import { PatternsSelect } from './facet'
import { ValuesSelect } from './facet'

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
//          console.log(" ")
//          console.log("f :", f)
          f = f.remove()
        }
      }
      this.facets = []
    }
  }

  updatePatterns (patterns) {
    this.patterns = patterns
  }

  createFacets (dataObject, key, type = 'categories', inPattern = false, value = false) {
    console.log(" ")
    console.log("________________________________")
    console.log("___facets.js - createFacets()___")

    console.log(" ")
    console.log("dataObject :", dataObject)
    console.log("key :", key)
    console.log("type :", type)
    console.log("inPattern :", inPattern)
    console.log("value :", value)

//    , labelId, labelText, facetId, elementType, data, generatorFacetId = false
    // Categories facet
    if (type == 'categories') {
      // Create
      let leftFacetNew = new Facet(
        this.editor,
        this.facetsContainerElement,
        this.containerDiv,
        'leftFacetContainer'
      )
      leftFacetNew.addLabel('leftLabel', 'Categories', 'leftFacet')
      leftFacetNew.addElement('categories', 'leftFacet', dataObject, key)
      leftFacetNew.element.addChangeEventListeners(this.editor)
      // Display
      leftFacetNew.element.fill(dataObject[key], true)
      leftFacetNew.element.show(this.editor, this.facetsContainerElement)
      // Add to facets
      this.facets.push(leftFacetNew)
    }

    // Patterns facet
    else if (type == 'patterns') {

      // Create

      console.log("")
      console.log("*** Call new Facet(...)")
      console.log("before call :")
      console.log("var1 :", this.editor)
      console.log("var2 :", this.facetsContainerElement)
      console.log("var3 :", this.containerDiv)
      console.log("var4 :", 'pattern_container')
      let patternFacetNew = new Facet(
        this.editor,
        this.facetsContainerElement,
        this.containerDiv,
        'pattern_container')

      console.log("")
      console.log("*** Call addLabel(...)")
      console.log("before call :")
      console.log("var1 : 'pattern_label'")
      console.log("var2 : 'Patterns'")
      console.log("var3 : 'pattern_select'")
      patternFacetNew.addLabel('pattern_label', 'Patterns', 'pattern_select')

      console.log("")
      console.log("*** Call addElement(...)")
      console.log("before call :")
      console.log("var1 : 'patterns'")
      console.log("var2 : 'pattern_select'")
      console.log("var3 = dataObject :", dataObject)
      console.log("var4 = key :", key)
      patternFacetNew.addElement('patterns', 'pattern_select', dataObject, key)
//      console.log("patternFacetNew :", patternFacetNew)
//      console.log("patternFacetNew.element :", patternFacetNew.element)
      patternFacetNew.element.addChangeEventListeners(this.editor, this.facetsContainerElement, this.patterns, inPattern)

      console.log("pattern facet :", patternFacetNew)

      // Display
      patternFacetNew.element.fill(key)
      patternFacetNew.element.show(this.editor, this.facetsContainerElement)
      // Add to facets
      this.facets.push(patternFacetNew)
    }

    // Input facet
    else if (type == 'number') {

      // Create

      console.log("")
      console.log("*** Call new Facet(...)")
      console.log("before call :")
      console.log("var1 :", this.editor)
      console.log("var2 :", this.facetsContainerElement)
      console.log("var3 :", this.containerDiv)
      console.log("var4 :", 'numberinput_container')
      let inputFacetNew = new Facet(
        this.editor,
        this.facetsContainerElement,
        this.containerDiv,
        'numberinput_container')

      console.log("")
      console.log("*** Call addLabel(...)")
      console.log("before call :")
      console.log("var1 : 'numberinput_label'")
//      console.log("var2 :", Number of, "key")
      console.log("var3 : 'numberinput_input'")
      inputFacetNew.addLabel('numberinput_label', 'Number of ' + key, 'numberinput_input')

      console.log("")
      console.log("*** Call addElement(...)")
      console.log("before call :")
      console.log("var1 : 'number'")
      console.log("var2 : 'numberinput_input'")
      console.log("var3 = dataObject :", dataObject)
      console.log("var4 = key :", key)
      console.log("var4 = value :", value)
      inputFacetNew.addElement('number', 'numberinput_input', dataObject, key, value)

      console.log("input facet :", inputFacetNew)

      inputFacetNew.element.addChangeEventListeners(this.editor, inPattern)
      // Display
//      inputFacetNew.element.fill(dataObject['expression'].values)
      inputFacetNew.element.show(this.editor, this.facetsContainerElement)
      // Add to facets
      this.facets.push(inputFacetNew)
    }
  }
}