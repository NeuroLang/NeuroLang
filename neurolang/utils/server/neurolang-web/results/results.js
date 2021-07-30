import './results.css'
import $ from 'jquery'
import { PapayaViewer } from '../papaya/viewer'
import { API_ROUTE, DATA_TYPES, PUBMED_BASE_URL } from '../constants'

/**
 * Show the results of query execution
 * @param {*} data the results from the query execution
 */
export function showQueryResults (data) {
  resultsContainer.show()
  const rm = new ResultsManager(data.data)
  rm.init()
}

/**
 * Hide the results
 */
export function hideQueryResults () {
  resultsContainer.hide()
}

const resultsContainer = $('#resultsContainer')
const resultsTabs = resultsContainer.find('.nl-results-tabs')
const tabTable = resultsContainer.find('.nl-result-table')

class ResultsManager {
  constructor (resultsData) {
    this.results = resultsData
    this.activeSymbol = undefined
    this.tableData = undefined
    tabTable.off('draw.dt').on('draw.dt', () => this.onTableDraw())

    this.viewer = new PapayaViewer()
  }

  /**
   * Initialize the results view with new results.
   * This method will create tabs for all the symbols in the result data
   * and display the data for the default selected tab.
   * @param {*} defaultTab
   */
  init (defaultTab) {
    // clear tabs
    resultsTabs.empty()

    for (const symbol in this.results.results) {
      const tab = $(`<li class='item'>${symbol}</li>`)
      if (typeof defaultTab === 'undefined' || defaultTab === symbol) {
        tab.addClass('active')
        defaultTab = symbol
      }
      tab.on('click', (evt) => this.setActiveResultTab(evt, symbol))
      resultsTabs.append(tab)
    }

    // display selected tab
    this.setActiveResultTab(null, defaultTab)
  }

  /**
   * Callback for when a new tab has been selected
   * @param {*} evt
   * @param {*} symbol
   */
  setActiveResultTab (evt, symbol) {
    // remove active class from previous active tab
    if (evt !== null) {
      resultsContainer.find('li.active').removeClass('active')
      $(evt.target).addClass('active')
    }

    this.activeSymbol = symbol

    // clear previous tab results
    if ($.fn.DataTable.isDataTable(tabTable)) {
      tabTable.DataTable().destroy()
      tabTable.empty()
    }

    // prepare results table by defining col types
    // and initialize table
    const tab = this.results.results[symbol]
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
    tabTable.DataTable({
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
   * Whenever the datatable gets redrawn, we need to add event
   * listeners to the region checkboxes which might be in the table.
   */
  onTableDraw () {
    $('.nl-vbr-overlay-switch input').on('change', (evt) => {
      // get the item's image data
      const elmt = $(evt.target)
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
          }
        } else {
          $('#nlPapayaContainer .nl-papaya-alert').show()
          evt.target.checked = false
        }
      } else {
        this.viewer.removeImage(imageID)
        elmt.siblings('label').attr('style', '')
      }
    })
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
      const imgSwitch = `<div class="ui toggle checkbox nl-vbr-overlay-switch">
      <input type="checkbox" data-row=${meta.row} data-col=${meta.col}>
      <label>Show region</label></div>
      `
      return imgSwitch
    }
    // otherwise return the raw data (for ordering)
    return data.hash
  }
  return data
}
