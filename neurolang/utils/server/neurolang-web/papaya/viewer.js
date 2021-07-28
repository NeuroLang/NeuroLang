import $ from 'jquery'

export function showViewer () {
  // send a request for the atlas image
  $.get('http://localhost:8888/v1/atlas')
    .done(function (data) {
      initViewer(data.data.image)
    })
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
