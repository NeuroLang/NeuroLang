import './viewer.css'
import $ from '../jquery-bundler'
import { API_ROUTE } from '../constants'
import Plotly from 'plotly.js-dist-min'

const LUTS = [
  { name: 'red', data: [[0, 0.96, 0.26, 0.21], [1, 0.96, 0.26, 0.21]], gradation: false, hex: '#f44336' }, // #f44336
  { name: 'pink', data: [[0, 0.91, 0.12, 0.39], [1, 0.91, 0.12, 0.39]], gradation: false, hex: '#e91e63' }, // #e91e63
  { name: 'deep purple', data: [[0, 0.4, 0.23, 0.72], [1, 0.4, 0.23, 0.72]], gradation: false, hex: '#673ab7' }, // #673ab7
  { name: 'indigo', data: [[0, 0.25, 0.32, 0.71], [1, 0.25, 0.32, 0.71]], gradation: false, hex: '#3f51b5' }, // #3f51b5
  { name: 'blue', data: [[0, 0.13, 0.59, 0.95], [1, 0.13, 0.59, 0.95]], gradation: false, hex: '#2196f3' }, // #2196f3
  { name: 'cyan', data: [[0, 0, 0.74, 0.83], [1, 0, 0.74, 0.83]], gradation: false, hex: '#00bcd4' }, // #00bcd4
  { name: 'teal', data: [[0, 0, 0.59, 0.53], [1, 0, 0.59, 0.53]], gradation: false, hex: '#009688' }, // #009688
  { name: 'green', data: [[0, 0.30, 0.69, 0.31], [1, 0.30, 0.69, 0.31]], gradation: false, hex: '#4caf50' }, // #4caf50
  { name: 'lime', data: [[0, 0.80, 0.86, 0.22], [1, 0.80, 0.86, 0.22]], gradation: false, hex: '#cddc39' }, // #cddc39
  { name: 'yellow', data: [[0, 1, 0.92, 0.23], [1, 1, 0.92, 0.23]], gradation: false, hex: '#ffeb3b' }, // #ffeb3b
  { name: 'orange', data: [[0, 1, 0.60, 0], [1, 1, 0.60, 0]], gradation: false, hex: '#ff9800' }, // #ff9800
  { name: 'blue gray', data: [[0, 0.38, 0.49, 0.55], [1, 0.38, 0.49, 0.55]], gradation: false, hex: '#607d8b' } // #607d8b
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

export class PapayaViewer {
  constructor (atlasKey) {
    this.atlasKey = atlasKey
    this.imageIds = []
    this.lutIndex = 0
    this.resultsContainer = $('#symbolsContainer')
    this.papayaContainer = $('#nlPapayaContainer')
  }

  showViewer () {
    $('.nl-papaya-alert').hide()
    if (ATLAS_CACHE.exists(this.atlasKey)) {
      this.resultsContainer.width('50%')
      if (this.papayaContainer.is(':hidden')) {
        this.papayaContainer.show(500, () => this._initViewer())
      } else {
        this._initViewer()
      }
    } else {
      $.get(API_ROUTE.atlas)
        .done((data) => {
          ATLAS_CACHE.set(this.atlasKey, data.data.image)
          this.resultsContainer.width('50%')
          if (this.papayaContainer.is(':hidden')) {
            this.papayaContainer.show(500, () => this._initViewer())
          } else {
            this._initViewer()
          }
        })
    }
  }

  hideViewer (duration = 500) {
    this.papayaContainer.hide(duration)
    this.resultsContainer.width('100%')
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
      papaya.Container.addViewer('nlPapayaParent', this.params)
    } else {
    // reset the first (only) viewer
      papaya.Container.resetViewer(0, this.params)
    }
    this.imageIds = ['atlas']
  }

  /**
   * Add an image to the papaya viewer.
   * Maximum 8 images can be added to the papaya viewer
   * @param {*} name the unique name for this image
   * @param {*} image the image data (base64 encoded)
   * @param {*} min the min value for the image
   * @param {*} max the max value for the image
   */
  addImage (name, image, min, max) {
    let res = null
    if (this.canAdd()) {
      window[name] = image
      const imageParams = this._getImageParams(name, image, min, max)
      papaya.Container.addImage(0, name, imageParams)
      this.imageIds.push(name)
      if ('hex' in imageParams[name]) {
        res = imageParams[name].hex
      }
    }
    return res
  }

  removeImage (name) {
    const idx = this.imageIds.indexOf(name)
    if (idx > -1) {
      papaya.Container.removeImage(0, idx)
      this.imageIds.splice(idx, 1)
    }
  }

  canAdd () {
    return this.imageIds.length - 1 < 8
  }

  imageIndex (name) {
    return this.imageIds.indexOf(name)
  }

  setCoordinates (coords) {
    papayaContainers[0].viewer.gotoWorldCoordinate(
      new papaya.core.Coordinate(
        coords[0],
        coords[1],
        coords[2]
      ),
      false
    )
  }

  onImageLoaded () {
    // this.showImageHistogram(this.imageIds.length - 1)
  }

  showImageHistogram (imageId) {
    const index = this.imageIndex(imageId)
    const trace = {
      x: papayaContainers[0].viewer.screenVolumes[index].volume.imageData.data.filter((elt) => elt !== 0),
      type: 'histogram'
    }
    const data = [trace]
    const layout = {
      title: 'Histogram of non-zero image data',
      showlegend: false,
      width: 380,
      height: 380
    }
    Plotly.newPlot('nlHistogramContainer', data, layout)
  }

  _getImageParams (name, image, min, max) {
    const imageParams = {}
    if (typeof min !== 'undefined' && typeof max !== 'undefined' && min === max) {
      // showing a segmented region
      const lut = LUTS[this.lutIndex]
      imageParams.lut = lut.name
      imageParams.hex = lut.hex
      this.lutIndex = (this.lutIndex + 1) % LUTS.length
      imageParams.alpha = 0.8
    } else {
      // showing an overlay
      if (typeof min !== 'undefined') {
        imageParams.min = Number(min.toFixed(2))
      }
      if (typeof max !== 'undefined') {
        imageParams.max = Number(max.toFixed(2))
      }
      imageParams.loadingComplete = () => this.onImageLoaded()
    }
    const params = []
    params[name] = imageParams
    return params
  }
}
