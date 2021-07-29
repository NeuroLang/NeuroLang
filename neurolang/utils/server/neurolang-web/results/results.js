import './results.css'
import $ from 'jquery'
import { hideViewer, showViewer } from '../papaya/viewer'
import { API_ROUTE, DATA_TYPES, PUBMED_BASE_URL } from '../constants'

export function showQueryResults (queryId, data) {
  resultsContainer.show()
  createResultTabs(data.data)
}

const resultsContainer = $('#resultsContainer')
const resultsTabs = resultsContainer.find('.nl-results-tabs')
const tabTable = resultsContainer.find('.nl-result-table')

function createResultTabs (data, defaultTab) {
  // clear tabs
  resultsTabs.empty()

  // add new tabs
  for (const symbol in data.results) {
    const tab = $(`<li class='nav-item nav-link'>${symbol}</li>`)
    if (typeof defaultTab === 'undefined' || defaultTab === symbol) {
      tab.addClass('active')
      defaultTab = symbol
    }
    tab.on('click', (evt) => setActiveResultTab(evt, data, symbol))
    resultsTabs.append(tab)
  }

  // display selected tab
  setActiveResultTab(null, data, defaultTab)
}

function setActiveResultTab (evt, data, symbol) {
  // remove active class from previous active tab
  if (evt !== null) {
    resultsContainer.find('li.active').removeClass('active')
    $(evt.target).addClass('active')
  }

  // clear previous tab results
  if ($.fn.DataTable.isDataTable(tabTable)) {
    tabTable.DataTable().destroy()
    tabTable.empty()
  }

  // prepare table data and show it
  const queryId = data.uuid
  const tab = data.results[symbol]
  const cols = tab.columns.map((col, idx) => {
    const ret = {
      title: col
    }
    if (tab.row_type[idx] === DATA_TYPES.studyID) {
      ret.render = renderPMID
    } else if (tab.row_type[idx] === DATA_TYPES.VBROverlay) {
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
    ajax: (data, callback, settings) => getAjaxTableData(data, callback, settings, queryId, symbol)
  })

  // hide or show papaya viewer.
  if (tab.row_type.some((elt) => elt === DATA_TYPES.VBROverlay)) {
    showViewer()
  } else {
    hideViewer()
  }
}

/**
 * Custom function to make the ajax call when requesting new data
 * for the datatables.
 * See https://datatables.net/reference/option/ajax
 * @param {*} data
 * @param {*} callback
 * @param {*} settings
 */
function getAjaxTableData (data, callback, settings, queryId, symbol) {
  const queryData = {
    symbol: symbol,
    start: data.start,
    length: data.length
  }
  if ('order' in data && data.order.length > 0) {
    queryData.sort = data.order[0].column
    queryData.asc = +(data.order[0].dir === 'asc')
  }
  $.ajax({
    url: `${API_ROUTE.status}/${queryId}`,
    type: 'get',
    data: queryData
  }).done(function (result) {
    const tableData = {
      draw: data.draw,
      recordsTotal: result.data.results[symbol].size,
      recordsFiltered: result.data.results[symbol].size,
      data: result.data.results[symbol].values
    }
    callback(tableData)
  })
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
function renderVBROverlay (data, type) {
  if (type === 'display') {
    // when datatables is trying to display the value, return a switch to display
    return '<div class="form-check form-switch"><input class="form-check-input" type="checkbox"><label class="form-check-label">Show region</label></div>'
  }
  // otherwise return the raw data (for ordering)
  return data
}
