import './viewer.css'
import $ from 'jquery'
import { API_ROUTE } from '../constants'

/**
 * Reset the papaya viewer and show it.
 */
export function showViewer () {
  // send a request for the atlas image
  $.get(API_ROUTE.atlas)
    .done(function (data) {
      $('#resultsContainer').width('50%')
      $('#nlPapayaContainer').show(500, function () { initViewer(data.data.image) })
    })
}

/**
 * Hide the papaya viewer.
 */
export function hideViewer () {
  $('#nlPapayaContainer').hide(500)
  $('#resultsContainer').width('100%')
}

function initViewer (atlasImage) {
  const params = initParams()
  window.atlas = atlasImage
  params.encodedImages = ['atlas']
  if (papayaContainers.length === 0) {
    // add a new viewer
    papaya.Container.addViewer('nlPapayaContainer', params)
  } else {
    // reset the first (only) viewer
    papaya.Container.resetViewer(0, params)
  }
}

function initParams () {
  const params = []
  params.worldSpace = true
  params.kioskMode = true
  params.showControlBar = true
  params.showImageButtons = false
  return params
}

function addImage (name, image, params) {
  window[name] = image
  papaya.Container.addImage(0, name, params)
}
