import './results.css'
import $ from '../jquery-bundler'
import { PapayaViewer } from '../papaya/viewer'
import { API_ROUTE, DATA_TYPES, PUBMED_BASE_URL } from '../constants'

export class ResultsController {
  constructor () {
    this.results = undefined
    this.activeSymbol = undefined
    this.tableData = undefined

    this.resultsContainer = $('#symbolsContainer')
    this.dropdown = this.resultsContainer.find('.nl-symbols-dropdown')
    this.tabTable = this.resultsContainer.find('.nl-symbols-table')

    this.tabTable.off('draw.dt').on('draw.dt', () => this.onTableDraw())
    this.dropdown.dropdown()
    this.dropdown.find('input').on('change', (evt) => this.onSymbolChange(evt))
    this.viewer = new PapayaViewer()
  }

  /**
   * Set the results of a query execution
   * @param {*} data
   */
  setQueryResults (data) {
    this.resultsContainer.show()
    this.results = data.data
    this._updateSymbolsList()
  }

  /**
   * Hide the symbols container
   */
  hide () {
    this.resultsContainer.hide()
    this.viewer.hideViewer(0)
  }

  /**
   * Set the list of symbols in the dropdown
   */
  _updateSymbolsList () {
    const keys = Object.keys(this.results.results)
    keys.sort()
    this.dropdown.dropdown('setup menu',
      { values: keys.map((elt) => ({ value: elt, name: elt })) }
    )
    this.dropdown.dropdown('set selected', keys[0])
  }

  /**
   * Callback for symbol change
   * @param {*} evt
   */
  onSymbolChange (evt) {
    this.activeSymbol = evt.target.value
    // clear previous table data
    if ($.fn.DataTable.isDataTable(this.tabTable)) {
      this.tabTable.DataTable().destroy()
      this.tabTable.empty()
    }

    // prepare table by defining col types
    // and initialize table
    const tab = this.results.results[this.activeSymbol]
    const cols = tab.columns.map((col, idx) => {
      const ret = {
        title: col
      }
      if (tab.row_type[idx] === DATA_TYPES.studyID) {
        ret.render = renderPMID
      } else if (tab.row_type[idx] === DATA_TYPES.VBROverlay || tab.row_type[idx] === DATA_TYPES.VBR) {
        ret.render = renderVBROverlay
      }
      return ret
    })
    this.tabTable.DataTable({
      processing: true,
      serverSide: true,
      pageLength: 25,
      order: [],
      searching: false,
      columns: cols,
      ajax: (data, callback, settings) => this.fetchNewTableData(data, callback, settings)
    })

    // hide or show papaya viewer.
    if (tab.row_type.some((elt) => elt === DATA_TYPES.VBROverlay || elt === DATA_TYPES.VBR)) {
      this.viewer.showViewer()
    } else {
      this.viewer.hideViewer()
    }
  }

  /**
   * This function is called by the DataTables object to fetch new data
   * to be displayed in the table (either upon initialization, or when the user
   * changes page, or sorts a column).
   * It needs to fetch the new data from the server, based on the criteria in
   * the data parameter, and then call the callback with the new data.
   * @param {*} data
   * @param {*} callback
   * @param {*} settings
   */
  fetchNewTableData (data, callback, settings) {
    // get the symbol, page start, length
    const queryData = {
      symbol: this.activeSymbol,
      start: data.start,
      length: data.length
    }
    // add sorting info
    if ('order' in data && data.order.length > 0) {
      queryData.sort = data.order[0].column
      queryData.asc = +(data.order[0].dir === 'asc')
    }
    // send get request to the server
    const queryId = this.results.uuid
    $.ajax({
      url: `${API_ROUTE.status}/${queryId}`,
      type: 'get',
      data: queryData
    }).done((result) => {
      this.tableData = result.data
      const dataToDraw = {
        draw: data.draw,
        recordsTotal: result.data.results[this.activeSymbol].size,
        recordsFiltered: result.data.results[this.activeSymbol].size,
        data: result.data.results[this.activeSymbol].values
      }
      callback(dataToDraw)
    })
  }

  /**
   * Whenever the datatable gets redrawn, we need to attach event
   * listeners to controls which might be in the table (show image buttons, etc.)
   */
  onTableDraw () {
    $('.nl-vbr-overlay-switch input').on('change', (evt) => this.onImageSwitchChanged(evt))
    $('.nl-vbr-overlay-center').on('click', (evt) => this.onImageCenterClicked(evt))

    $('.nl-vbr-overlay-hist').on('click', (evt) => this.onShowHistogramClicked(evt))
    $('.nl-vbr-overlay-hist').popup({
      on: 'manual',
      html: '<div id="nlHistogramContainer" class="nl-histogram-container"></div>',
      position: 'top center',
      lastResort: 'top center'
    })
  }

  /**
   * Listener for value change on Show Image toggle buttons
   * @param {*} evt
   */
  onImageSwitchChanged (evt) {
    // get the item's image data
    const elmt = $(evt.target)
    const parentDiv = elmt.parent().parent()
    const col = elmt.data('col')
    const row = elmt.data('row')
    const imageID = `image_${row}_${col}`
    const imageData = this.tableData.results[this.activeSymbol].values[row][col]
    $('#nlPapayaContainer .nl-papaya-alert').hide()
    if (evt.target.checked) {
      if (this.viewer.canAdd()) {
        const hex = this.viewer.addImage(imageID, imageData.image, imageData.min, imageData.max)
        if (hex !== null) {
          elmt.siblings('label').attr('style', 'color: ' + hex + ' !important')
          parentDiv.addClass('region-label')
        }
        parentDiv.addClass('displayed')
      } else {
        $('#nlPapayaContainer .nl-papaya-alert').show()
        evt.target.checked = false
      }
    } else {
      this.viewer.removeImage(imageID)
      elmt.siblings('label').attr('style', '')
      parentDiv.removeClass('displayed')
    }
  }

  /**
   * Listener for click events on show histogram buttons
   * @param {*} evt
   */
  onShowHistogramClicked (evt) {
    let elmt = $(evt.target)
    if (elmt.is('i')) {
      elmt = elmt.parent()
    }
    const col = elmt.data('col')
    const row = elmt.data('row')
    const imageID = `image_${row}_${col}`
    const vis = elmt.popup('is visible')
    elmt.popup('toggle')
    if (!vis) {
      this.viewer.showImageHistogram(imageID)
    }
  }

  /**
   * Listener for click events on image center buttons
   * @param {*} evt
   */
  onImageCenterClicked (evt) {
    // get the item's image data
    let elmt = $(evt.target)
    if (elmt.is('i')) {
      elmt = elmt.parent()
    }
    const col = elmt.data('col')
    const row = elmt.data('row')
    const imageData = this.tableData.results[this.activeSymbol].values[row][col]
    this.viewer.setCoordinates(imageData.center)
  }
}

/**
 * Custom renderer for PMID values.
 * Displays PMIDs as links.
 * @param {*} data
 * @param {*} type
 */
function renderPMID (data, type) {
  if (type === 'display') {
    // when datatables is trying to display the value, return a link tag
    return `<a class="nl-pmid-link" href="${PUBMED_BASE_URL}${data}" target="_blank">PubMed:${data}</a>`
  }
  // otherwise return the raw data (for ordering)
  return data
}

/**
 * Custom renderer for VBROverlay values
 * @param {*} data
 * @param {*} type
 * @returns
 */
function renderVBROverlay (data, type, row, meta) {
  if (typeof data === 'object' && 'image' in data) {
    if (type === 'display') {
      // when datatables is trying to display the value, return a switch to display
      const imgDiv = `<div class="nl-vbr-overlay-controls">
      <div class="ui toggle checkbox nl-vbr-overlay-switch">
      <input type="checkbox" data-row=${meta.row} data-col=${meta.col}>
      <label>Show region</label></div>
      <button class="ui tiny icon button nl-vbr-overlay-center nl-overlay-control"
      data-row=${meta.row} data-col=${meta.col} data-tooltip="Center on region">
      <i class="crosshairs icon"></i></button>
      <button class="ui tiny icon button nl-vbr-overlay-hist nl-overlay-control"
      data-row=${meta.row} data-col=${meta.col} data-tooltip="Show image histogram">
      <i class="chart bar outline icon"></i></button></div>
      `
      return imgDiv
    }
    // otherwise return the raw data (for ordering)
    return data.hash
  }
  return data
}
