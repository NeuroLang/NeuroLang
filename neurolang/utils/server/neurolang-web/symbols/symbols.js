import './symbols.css'
import $ from '../jquery-bundler'
import { createMiniColorBar, PapayaViewer } from '../papaya/viewer'
import { API_ROUTE, DATA_TYPES, PUBMED_BASE_URL } from '../constants'

export class SymbolsController {
  constructor () {
    this.results = undefined
    this.activeSymbol = undefined
    this.tableData = undefined

    this.resultsContainer = $('#symbolsContainer')
    this.dropdown = this.resultsContainer.find('.nl-symbols-dropdown')
    this.tabTable = this.resultsContainer.find('.nl-symbols-table')
    this.functionsHelp = this.resultsContainer.find('.nl-functions-msg')
    this.symbolsHelp = this.resultsContainer.find('.nl-symbols-help')
    this.symbolsDownload = this.resultsContainer.find('.nl-symbols-download')

    this.tabTable.off('draw.dt').on('draw.dt', () => this.onTableDraw())
    this.dropdown.dropdown()
    this.dropdown.find('input').on('change', (evt) => this.onSymbolChange(evt))
    this.symbolsHelp.popup()
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
   * Set the active engine. This will trigger a call to the server
   * to fetch the symbols & function available on the engine.
   * @param {*} engine
   */
  setRouteEngine (engine) {
    // reset the results from a previous engine
    this.results = undefined
    // Fetch symbols for this engine type
    this.engine = engine
    $.ajax({
      url: `${API_ROUTE.symbols}/${engine}`,
      type: 'get'
    }).done((data) => {
      if ('status' in data && data.status === 'ok' && data.data.done && !('errorName' in data.data)) {
        // Fetching the symbols done with success
        this.resultsContainer.show()
        this.symbols = data.data
        this._updateSymbolsList()
      } else {
        // An error occurred while fetching the symbols. Fail silently and hide self
        this.symbols = undefined
        console.log('An error occurred while fetching symbols for engine ' + engine)
        console.log(data)
        this.hide()
      }
    })
  }

  /**
   * Set the list of symbols in the dropdown. This list is composed
   * of query result symbols if available, and symbols & functions
   * present on the engine.
   */
  _updateSymbolsList () {
    const menu = this.dropdown.find('.menu')
    menu.empty()
    let selected = null
    // Always add the query symbols at the top of the list
    let resultKeys = []
    if (this.results) {
      resultKeys = Object.keys(this.results.results)
      resultKeys.sort()
      const probClasses = resultKeys.map((val) => this.results.results[val].probabilistic ? 'probabilistic' : '')
      addItemsToDropdownMenu(menu, resultKeys, 'Query symbols', 'symbol query-symbol', probClasses)
      selected = resultKeys.find(val => this.results.results[val].last_parsed_symbol)
      if (!selected) {
        selected = resultKeys[0]
      }
    }
    // Then add the base symbols and functions
    if (this.symbols) {
      const functions = Object.keys(this.symbols.results)
        .filter(elt => this.symbols.results[elt].function)
      const commands = Object.keys(this.symbols.results)
        .filter(elt => this.symbols.results[elt].command)
      const symbols = Object.keys(this.symbols.results)
        .filter(elt => !this.symbols.results[elt].function &&
          !this.symbols.results[elt].command && resultKeys.indexOf(elt) < 0)
      if (symbols.length > 0) {
        addItemsToDropdownMenu(menu, symbols, 'Base symbols', 'symbol base-symbol')
        selected = !selected ? symbols[0] : selected
      }
      if (functions.length > 0) {
        addItemsToDropdownMenu(menu, functions, 'Functions', 'function')
        selected = !selected ? functions[0] : selected
      }
      if (commands.length > 0) {
        addItemsToDropdownMenu(menu, commands, 'Commands', 'command')
        selected = !selected ? commands[0] : selected
      }
    }
    this.dropdown.dropdown('refresh')
    const previousSelected = this.dropdown.dropdown('get value')
    this.dropdown.dropdown('set selected', selected)
    if (previousSelected === selected) {
      // manually trigger a change event to redraw the table
      this.onSymbolChange()
    }
  }

  /**
   * Callback for when the selected symbol changes.
   * @param {*} evt
   */
  onSymbolChange (evt) {
    this.activeSymbol = evt ? evt.target.value : this.activeSymbol

    // get the active symbol metadata
    const isQuerySymbol = this.results && this.activeSymbol in this.results.results
    const tab = isQuerySymbol ? this.results.results[this.activeSymbol] : this.symbols.results[this.activeSymbol]
    if (('function' in tab && tab.function) || ('command' in tab && tab.command)) {
      // Selected symbol is a function, display its doctstring
      this.setFunctionHelp('', `A ${tab.function ? 'function' : 'command'} of type ${tab.type}`, this.activeSymbol, tab.doc)
      this.tabTable.parents('div.dataTables_wrapper').first().hide()
      this.viewer.hideViewer()
    } else {
      // Selected symbol is a RelationalAlgebraSet, display its values
      // clear previous table data
      if ($.fn.DataTable.isDataTable(this.tabTable)) {
        this.tabTable.DataTable().destroy()
        this.tabTable.empty()
      }

      this.tabTable.parents('div.dataTables_wrapper').first().show()
      this.functionsHelp.hide()
      const cols = tab.columns.map((col, idx) => {
        const ret = {
          title: col
        }
        if (tab.row_type[idx] === DATA_TYPES.studyID) {
          ret.render = renderPMID
        } else if (tab.row_type[idx] === DATA_TYPES.VBROverlay || tab.row_type[idx] === DATA_TYPES.VBR) {
          ret.render = renderVBROverlay
        } else if (tab.row_type[idx] === DATA_TYPES.MpltFigure) {
          ret.render = renderMpltFigure
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
      const hasImages = tab.row_type.some((elt) => elt === DATA_TYPES.VBROverlay || elt === DATA_TYPES.VBR)
      if (hasImages) {
        this.viewer.showViewer()
      } else {
        this.viewer.hideViewer()
      }

      // hide or show the download link.
      if (isQuerySymbol && !hasImages) {
        this.symbolsDownload.show()
        this.symbolsDownload.attr('href', `${API_ROUTE.downloads}/${this.results.uuid}?symbol=${this.activeSymbol}`)
      } else {
        this.symbolsDownload.hide()
      }
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
    const url = (this.results && this.activeSymbol in this.results.results)
      ? `${API_ROUTE.status}/${this.results.uuid}`
      : `${API_ROUTE.symbols}/${this.engine}`
    $.ajax({
      url: url,
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
   * Display the help content for the selected function
   * @param {*} style
   * @param {*} content
   * @param {*} header
   * @param {*} help
   */
  setFunctionHelp (style, content, header, help) {
    const fHeader = this.functionsHelp.find('.nl-functions-header')
    if (typeof header !== 'undefined') {
      fHeader.text(header)
    } else {
      fHeader.empty()
    }
    const fMsg = this.functionsHelp.find('.nl-functions-message')
    if (typeof content !== 'undefined') {
      fMsg.text(content)
    } else {
      fMsg.empty()
    }
    const fHelp = this.functionsHelp.find('.nl-functions-help')
    if (typeof help !== 'undefined') {
      fHelp.text(help)
      fHelp.show()
    } else {
      fHelp.empty()
      fHelp.hide()
    }
    this.functionsHelp.removeClass('info error warning success')
    this.functionsHelp.addClass(style)
    this.functionsHelp.show()
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
    $('.nl-image-download').on('click', (evt) => this.onImageDownloadClicked(evt))
    $('.nl-mini-colorbar').on('click', (evt) => this.onMiniColorBarClicked(evt))
    $('.nl-mini-colorbar').popup()
    $('.nl-mplt-figure-toggle').on('click', (evt) => this.onShowFigureClicked(evt))
    $('.nl-mplt-figure-toggle').popup({
      on: 'manual',
      html: '<div id="nlMpltFigureContainer" class="nl-mplt-figure-container">' +
      '<div class="ui active inverted dimmer"><div class="ui text loader">Loading</div></div></div>',
      position: 'right center',
      lastResort: 'right center'
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
        const imageParams = this.viewer.addImage(imageID, imageData.image, imageData.min, imageData.max, imageData.q95)
        if ('hex' in imageParams) {
          elmt.siblings('label').attr('style', 'color: ' + imageParams.hex + ' !important')
          parentDiv.addClass('region-label')
        } else {
          const miniCBDiv = elmt.parents('.nl-vbr-overlay-controls').find('.nl-mini-colorbar')
          const canvas = createMiniColorBar(imageParams.lut)
          canvas.style.height = '100%'
          miniCBDiv.empty()
          miniCBDiv.append(canvas)
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

  onImageDownloadClicked (evt) {
    // get the item's image data
    const elmt = $(evt.target).parents('.nl-image-download')
    const idx = elmt.data('idx')
    const col = elmt.data('col')
    let url = (this.results && this.activeSymbol in this.results.results)
      ? `${API_ROUTE.downloads}/${this.results.uuid}`
      : `${API_ROUTE.downloads}/${this.engine}`
    url += `?symbol=${this.activeSymbol}&col=${col}&idx=${idx}`
    elmt.attr('href', url)
  }

  onMiniColorBarClicked (evt) {
    let elmt = $(evt.target)
    if (elmt.is('canvas')) {
      elmt = elmt.parent()
    }
    const col = elmt.data('col')
    const row = elmt.data('row')
    const imageID = `image_${row}_${col}`
    this.viewer.showColorBar(imageID)
  }

  /**
   * Listener for click events on show figure buttons.
   *
   * Gets the clicked item's row and col indices. Then fetches the data
   * for the figure and adds it to the nlMpltFigureContainer figure container.
   * @param {*} evt
   */
  onShowFigureClicked (evt) {
    let elmt = $(evt.target)
    if (elmt.is('i')) {
      elmt = elmt.parent()
    }
    const row = elmt.data('row')
    const col = elmt.data('col')
    const vis = elmt.popup('is visible')
    elmt.popup('toggle')
    if (!vis) {
      let url = (this.results && this.activeSymbol in this.results.results)
        ? `${API_ROUTE.figure}/${this.results.uuid}`
        : `${API_ROUTE.figure}/${this.engine}`
      url += `?symbol=${this.activeSymbol}&col=${col}&row=${row}`
      $.get(url)
        .done((figureData) => {
          const figContainer = $('#nlMpltFigureContainer')
          figContainer.append(figureData.documentElement)
          figContainer.find('.active.dimmer').removeClass('active')
          url += '&format=png'
          const downloadLink = $(`<a href="${url}" class="nl-figure-download">
          <button class="ui tiny circular icon basic button" data-tooltip="Download figure">
          <i class="download icon"></i></button></a>`)
          figContainer.append(downloadLink)
        })
    }
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
      <div class="nl-mini-colorbar" data-content="Display the colorbar for this image"
      data-row=${meta.row} data-col=${meta.col}></div>
      <button class="ui tiny icon button nl-vbr-overlay-center nl-overlay-control"
      data-row=${meta.row} data-col=${meta.col} data-tooltip="Center on region">
      <i class="crosshairs icon"></i></button>
      <button class="ui tiny icon button nl-vbr-overlay-hist nl-overlay-control"
      data-row=${meta.row} data-col=${meta.col} data-tooltip="Show image histogram">
      <i class="chart bar outline icon"></i></button>
      <a href="#" class="nl-image-download" data-idx=${data.idx} data-col=${meta.col}>
      <button class="ui tiny circular icon button" data-tooltip="Download image file">
      <i class="download icon"></i></button></a></div>
      `
      return imgDiv
    }
    // otherwise return the raw data (for ordering)
    return data.hash
  }
  return data
}

/**
 * Custom renderer for Matplotlib figures
 * @param {*} data
 * @param {*} type
 * @returns
 */
function renderMpltFigure (data, type, row, meta) {
  if (type === 'display') {
    // create a button to display the figure
    const figDiv = `<div class="nl-mplt-figure-controls">
      <img class="ui tiny bordered image nl-mplt-figure-toggle nl-figure-control"
      src="data:image/png;base64, ${data}" data-row=${meta.row} data-col=${meta.col} data-tooltip="Show figure">
      </div>`
    return figDiv
  }
  return data
}

/**
 * Helper function to add items to the dropdown menu.
 * @param {*} menu
 * @param {*} items
 * @param {*} header
 * @param {*} classes
 */
function addItemsToDropdownMenu (menu, items, header, classes, itemClasses) {
  menu.append($('<div class="divider"></div>'))
  menu.append($(`<div class="header">${header}</div>`))
  menu.append($('<div class="divider"></div>'))
  items.forEach((elt, idx) => {
    const div = $(`<div class="item ${classes}" data-value="${elt}">${elt}</div>`)
    if (itemClasses) {
      div.addClass(itemClasses[idx])
    }
    menu.append(div)
  })
}
