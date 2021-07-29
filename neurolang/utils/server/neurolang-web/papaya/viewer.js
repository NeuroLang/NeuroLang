import './viewer.css'
import $ from 'jquery'
import { API_ROUTE } from '../constants'

const LUTS = [
  { name: 'red', data: [[0, 0.96, 0.26, 0.21], [1, 0.96, 0.26, 0.21]], gradation: false }, // #f44336
  { name: 'pink', data: [[0, 0.91, 0.12, 0.39], [1, 0.91, 0.12, 0.39]], gradation: false }, // #e91e63
  { name: 'deep purple', data: [[0, 0.4, 0.23, 0.72], [1, 0.4, 0.23, 0.72]], gradation: false }, // #673ab7
  { name: 'indigo', data: [[0, 0.25, 0.32, 0.71], [1, 0.25, 0.32, 0.71]], gradation: false }, // #3f51b5
  { name: 'blue', data: [[0, 0.13, 0.59, 0.95], [1, 0.13, 0.59, 0.95]], gradation: false }, // #2196f3
  { name: 'cyan', data: [[0, 0, 0.74, 0.83], [1, 0, 0.74, 0.83]], gradation: false }, // #00bcd4
  { name: 'teal', data: [[0, 0, 0.59, 0.53], [1, 0, 0.59, 0.53]], gradation: false }, // #009688
  { name: 'green', data: [[0, 0.30, 0.69, 0.31], [1, 0.30, 0.69, 0.31]], gradation: false }, // #4caf50
  { name: 'lime', data: [[0, 0.80, 0.86, 0.22], [1, 0.80, 0.86, 0.22]], gradation: false }, // #cddc39
  { name: 'yellow', data: [[0, 1, 0.92, 0.23], [1, 1, 0.92, 0.23]], gradation: false }, // #ffeb3b
  { name: 'orange', data: [[0, 1, 0.60, 0], [1, 1, 0.60, 0]], gradation: false }, // #ff9800
  { name: 'blue gray', data: [[0, 0.38, 0.49, 0.55], [1, 0.38, 0.49, 0.55]], gradation: false } // #607d8b
]

class AtlasCache {
  constructor () {
    this.cache = {}
  }

  remove (key) {
    delete this.cache[key]
  }

  exists (key) {
    return !!this.cache[key]
  }

  get (key) {
    return this.cache[key]
  }

  set (key, atlas, callback) {
    delete this.cache[key]
    this.cache[key] = atlas
    if (typeof callback !== 'undefined') {
      callback(atlas)
    }
  }
}

const ATLAS_CACHE = new AtlasCache()
const resultsContainer = $('#resultsContainer')
const papayaContainer = $('#nlPapayaContainer')

/**
 * Hide the papaya viewer.
 */
export function hideViewer () {
  $('#nlPapayaContainer').hide(500)
  $('#resultsContainer').width('100%')
}

export class PapayaViewer {
  constructor (atlasKey) {
    this.atlasKey = atlasKey
    this.imageIds = []
  }

  showViewer () {
    if (ATLAS_CACHE.exists(this.atlasKey)) {
      resultsContainer.width('50%')
      papayaContainer.show(500, () => this._initViewer())
    } else {
      $.get(API_ROUTE.atlas)
        .done((data) => {
          ATLAS_CACHE.set(this.atlasKey, data.data.image)
          resultsContainer.width('50%')
          papayaContainer.show(500, () => this._initViewer())
        })
    }
  }

  hideViewer () {
    papayaContainer.hide(500)
    resultsContainer.width('100%')
  }

  _initParams () {
    const params = []
    params.worldSpace = true
    params.kioskMode = true
    params.showControlBar = true
    params.showImageButtons = true
    params.luts = LUTS
    this.params = params
  }

  /**
   * Initialize or reset the viewer.
   * Sets the atlas image as the first image
   */
  _initViewer () {
    this.atlasImage = ATLAS_CACHE.get(this.atlasKey)
    this._initParams()
    window.atlas = this.atlasImage
    this.params.encodedImages = ['atlas']
    if (papayaContainers.length === 0) {
    // add a new viewer
      papaya.Container.addViewer('nlPapayaContainer', this.params)
    } else {
    // reset the first (only) viewer
      papaya.Container.resetViewer(0, this.params)
    }
    this.imageIds = ['atlas']
  }

  /**
   * Add an image to the papaya viewer.
   * @param {*} name the unique name for this image
   * @param {*} image the image data (base64 encoded)
   * @param {*} min the min value for the image
   * @param {*} max the max value for the image
   */
  addImage (name, image, min, max) {
    window[name] = image
    papaya.Container.addImage(0, name, this._getImageParams(name, image, min, max))
    this.imageIds.push(name)
  }

  removeImage (name) {
    const idx = this.imageIds.indexOf(name)
    papaya.Container.removeImage(0, idx)
    this.imageIds.splice(idx, 1)
  }

  imageIndex (name) {
    return this.imagesIds.indexOf(name)
  }

  _getImageParams (name, image, min, max) {
    console.log(name)
    console.log(min)
    console.log(max)
    const imageParams = {}
    if (typeof min !== 'undefined' && typeof max !== 'undefined' && min === max) {
      // showing a segmented region
      imageParams.lut = 'lime'
      imageParams.alpha = 0.8
    } else {
      // showing an overlay
      if (typeof min !== 'undefined') {
        imageParams.min = Number(min.toFixed(2))
      }
      if (typeof max !== 'undefined') {
        imageParams.max = Number(max.toFixed(2))
      }
    }
    const params = []
    params[name] = imageParams
    return params
  }
}
