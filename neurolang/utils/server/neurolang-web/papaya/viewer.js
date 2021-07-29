import './viewer.css'
import $ from 'jquery'
import { API_ROUTE } from '../constants'

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

  addImage (name, image, params) {
    window[name] = image
    papaya.Container.addImage(0, name, params)
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
}

function defaultImageParams(name) {

}