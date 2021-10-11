import '../tests/tests.setup'
import '@testing-library/jest-dom/extend-expect'
import $ from '../jquery-bundler'
import 'datatables.net'
import dt from 'datatables.net-se'
import 'fomantic-ui-css/semantic'
import { SymbolsController } from './symbols'
import { API_ROUTE } from '../constants'
import { createMiniColorBar } from '../papaya/viewer'

const mockShowViewer = jest.fn()
const mockHideViewer = jest.fn()
const mockAddImage = jest.fn()
const mockCanAdd = jest.fn()
const mockCenter = jest.fn()
const mockHist = jest.fn()
const mockJqueryGet = jest.fn()

jest.mock('../papaya/viewer', () => ({
  PapayaViewer: jest.fn().mockImplementation(() => {
    return {
      showViewer: mockShowViewer,
      hideViewer: mockHideViewer,
      addImage: mockAddImage,
      canAdd: mockCanAdd,
      setCoordinates: mockCenter,
      showImageHistogram: mockHist
    }
  }),
  createMiniColorBar: jest.fn()
}))

dt(window, $)

const mockResults = {
  status: 'ok',
  data: {
    start: 0,
    length: 10,
    sort: -1,
    asc: true,
    uuid: 'b81b1831-0360-4ae7-a345-9269dabda8f0',
    cancelled: false,
    running: false,
    done: true,
    results: {
      ActivationGivenTerm: {
        row_type: ["<class 'int'>", "<class 'int'>", "<class 'int'>", "<class 'float'>"],
        columns: ['i', 'j', 'k', 'PROB'],
        size: 27697,
        values: [[2.0, 77.0, 46.0, 0.0015321803045817697], [4.0, 63.0, 34.0, 0.00023572711035194132], [5.0, 64.0, 38.0, 0.000419401098292473], [8.0, 54.0, 38.0, 0.001412803281281338], [12.0, 61.0, 40.0, 0.0018112978377986602], [12.0, 69.0, 36.0, 0.00045956587380234065], [14.0, 38.0, 40.0, 0.0018165763699549008]]
      },
      TermInStudyTFIDF: {
        row_type: ["<class 'str'>", "<class 'float'>", "<class 'neurolang.frontend.neurosynth_utils.StudyID'>"],
        columns: ['0', '1', '2'],
        size: 1049299,
        values: [['001', 0.055394216111399996, '9862924'], ['001', 0.09387570522489999, '11595392'], ['001', 0.0689931709903, '12077009'], ['001', 0.0996940021344, '12725761'], ['001', 0.09198243946769999, '12880904'], ['001', 0.0903373983476, '12958082'], ['001', 0.199392155694, '14561452'], ['001', 0.0735499288657, '14741317'], ['001', 0.10293863273399999, '14741643'], ['001', 0.079606876159, '15036060']]
      },
      RegionImage: {
        row_type: ["<class 'neurolang.regions.ExplicitVBROverlay'>"],
        columns: ['agg_create_region'],
        size: 2,
        values: [[{ image: 'someEncodedImageData', hash: '89er8798awre', center: [], max: 0.008, min: 0, q95: 0.001 }], [{ image: 'anotherEncodedImage', hash: '456eras8098ert', center: [], max: 0.009, min: 0.00002, q95: 0.00025 }]]
      }
    }
  }
}

function getTestHTML () {
  const HTML = `
    <div id="queryContainer" class="ui grid container">
    </div>
    <div class="ui container nl-symbols-viewer-container">
      <div id="symbolsContainer" class="nl-symbols-container">
        <div class="ui raised segments">
          <div class="ui label segment secondary">
            <span class="text">Select a symbol to explore</span>
            <div class="ui search selection dropdown nl-symbols-dropdown">
              <input type="hidden" name="symbol">
              <i class="dropdown icon"></i>
              <div class="default text">Select Symbol</div>
              <div class="menu">
              </div>
            </div>
          </div>
          <div class="ui  segment">
            <table class='ui compact striped table nl-symbols-table' id='tabTable' width="100%"></table>
            <div class="ui message nl-functions-msg">
              <div class="header nl-functions-header">
              </div>
              <p class="nl-functions-message"></p>
              <pre class="nl-functions-help"></pre>
          </div>
          </div>
        </div>
      </div>
      <div id="nlPapayaContainer" class="nl-papaya-container">
        <div id="nlPapayaParent" class="nl-papaya-parent"></div>
        <div class="nl-colorbar-container" id="nlColorbarContainer"></div>
        <div class="ui message warning nl-papaya-alert">
          <div class="header">You cannot add more than 8 overlays</div>
          <p>Please unselect an overlay to add a new one</p>
        </div>
      </div>
    </div>
    `
  return HTML
}

describe('SymbolsController', () => {
  let sc
  beforeEach(() => {
    document.body.innerHTML = getTestHTML()
    sc = new SymbolsController()
  })

  afterEach(() => {
    jest.clearAllMocks()
  })

  it('should create', () => {
    expect(sc).toBeDefined()
  })

  describe('setRouteEngine', () => {
    const engine = 'neuroquery'

    beforeEach(() => {
      $.ajax = mockJqueryGet.mockImplementation(() => {
        return $.Deferred().resolve(mockResults)
      })
    })

    it('should get engine symbols and display them', () => {
      sc.setRouteEngine(engine)
      expect($.ajax).toHaveBeenCalledWith({
        url: `${API_ROUTE.symbols}/${engine}`,
        type: 'get'
      })
      const items = $('.nl-symbols-dropdown .item')
      expect(items.length).toBe(3)
    })
  })

  describe('base symbols', () => {
    let mockFetchTableData
    let mockTableData
    const mockFunctions = {
      agg_count: { type: 'typing.Callable[[typing.Iterable], numpy.int64]', doc: 'Aggregate count function', function: true },
      superior_of: { type: 'typing.Callable[[neurolang.regions.Region, neurolang.regions.Region], bool]', doc: 'Superior_of function', function: true },
      agg_create_region: { type: 'typing.Callable[[typing.Iterable, typing.Iterable, typing.Iterable], neurolang.regions.ExplicitVBR]', doc: 'Create some brain region', function: true }
    }
    let symbols

    beforeEach(() => {
      mockTableData = {
        draw: 1,
        recordsTotal: 500,
        recordsFiltered: 500,
        data: mockResults.data.results.ActivationGivenTerm.values
      }
      mockFetchTableData = jest.spyOn(sc, 'fetchNewTableData').mockImplementationOnce((data, callback, settings) => callback(mockTableData))
      symbols = JSON.parse(JSON.stringify(mockResults.data))
      Object.assign(symbols.results, mockFunctions)
      sc.symbols = symbols
      sc._updateSymbolsList()
    })

    it('should display the base symbols and functions', () => {
      const items = $('.nl-symbols-dropdown .item')
      expect(items.length).toBe(6)
      const headers = $('.nl-symbols-dropdown .menu .header')
      expect(headers.length).toBe(2)
      expect(headers.first().text()).toBe('Base symbols')
      expect(headers.last().text()).toBe('Functions')
    })

    it('should show the details for the selected symbol or function', () => {
      // click on the first function item
      $('.nl-symbols-dropdown .item.function').first().trigger('click')

      expect($('.nl-symbols-table').is(':visible')).toBe(false)
      expect($('.nl-functions-msg .nl-functions-header').text()).toBe('agg_count')
      expect($('.nl-functions-msg .nl-functions-message').text()).toBe(`A function of type ${mockFunctions.agg_count.type}`)
      expect($('.nl-functions-msg .nl-functions-help').text()).toBe(mockFunctions.agg_count.doc)

      // click on last symbol item
      mockTableData = {
        draw: 2,
        recordsTotal: 500,
        recordsFiltered: 500,
        data: mockResults.data.results.TermInStudyTFIDF.values
      }
      mockFetchTableData.mockImplementationOnce((data, callback, settings) => {
        mockTableData.draw = data.draw
        callback(mockTableData)
      })
      $('.nl-symbols-dropdown .item.symbol').last().trigger('click')
      expect($('.nl-symbols-table tr').length).toBe(mockTableData.data.length + 1)
    })
  })

  describe('setQueryResults', () => {
    let mockFetchTableData
    let mockTableData
    beforeEach(() => {
      mockTableData = {
        draw: 1,
        recordsTotal: 500,
        recordsFiltered: 500,
        data: mockResults.data.results.ActivationGivenTerm.values
      }
      mockFetchTableData = jest.spyOn(sc, 'fetchNewTableData').mockImplementationOnce((data, callback, settings) => callback(mockTableData))
    })

    it('should add symbols to the symbols list', () => {
      sc.setQueryResults(mockResults)

      const items = $('.nl-symbols-dropdown .item')
      expect(items.length).toBe(3)
      expect(items.first().text()).toBe('ActivationGivenTerm')
      expect($(items.get(1)).text()).toBe('RegionImage')
      expect(items.last().text()).toBe('TermInStudyTFIDF')
    })

    it('should select the first item from the list', () => {
      sc.setQueryResults(mockResults)
      expect(sc.activeSymbol).toBe('ActivationGivenTerm')

      expect(mockFetchTableData).toHaveBeenCalled()
      // Expect 1 row per item + header
      expect($('#tabTable tr').length).toBe(mockTableData.data.length + 1)
    })

    describe('image results', () => {
      beforeEach(() => {
        sc.setQueryResults(mockResults)
        mockFetchTableData.mockClear()
        mockShowViewer.mockClear()
        mockHideViewer.mockClear()

        sc.tableData = mockResults.data
        mockTableData = {
          draw: 1,
          recordsTotal: 1,
          recordsFiltered: 1,
          data: mockResults.data.results.RegionImage.values
        }
        createMiniColorBar.mockReturnValueOnce(document.createElement('canvas'))
      })

      it('should show papaya viewer when tab has VBROverlay type', () => {
        // trigger click on 2nd menu item which contains images
        $('.nl-symbols-dropdown .item').eq(1).trigger('click')
        expect(mockShowViewer).toHaveBeenCalled()
        expect(mockHideViewer).not.toHaveBeenCalled()
        expect(mockFetchTableData).toHaveBeenCalled()
      })

      it('should show controls when displaying brain images', () => {
        // trigger click on 2nd menu item which contains images
        $('.nl-symbols-dropdown .item').eq(1).trigger('click')
        expect(mockShowViewer).toHaveBeenCalled()
        expect($('#tabTable tr').length).toBe(mockTableData.data.length + 1)
        expect($('.nl-vbr-overlay-switch').length).toBe(mockTableData.data.length)
        // Expect 2 buttons to center & show histograms. Buttons should be hidden
        expect($('.nl-vbr-overlay-center').length).toBe(mockTableData.data.length)
        expect($('.nl-vbr-overlay-center:visible').length).toBe(0)
        expect($('.nl-vbr-overlay-hist').length).toBe(mockTableData.data.length)
        expect($('.nl-vbr-overlay-hist:visible').length).toBe(0)
      })

      it('should call setImage on papayaViewer when image switch is checked', () => {
        // trigger click on 2nd menu item which contains images
        $('.nl-symbols-dropdown .item').eq(1).trigger('click')
        mockCanAdd.mockReturnValueOnce(true)
        mockAddImage.mockReturnValueOnce({ lut: 'Viridis' })

        // click on second image checkbox
        const chkbox = $('.nl-vbr-overlay-switch input').last()
        chkbox.prop('checked', true)
        chkbox.trigger('change')

        const expectedImage = mockResults.data.results.RegionImage.values[1][0]
        expect(mockCanAdd).toHaveBeenCalled()
        expect(mockAddImage).toHaveBeenCalledWith('image_1_0', expectedImage.image, expectedImage.min, expectedImage.max, expectedImage.q95)
      })

      it('should not add image if max images already added', () => {
        mockCanAdd.mockReturnValueOnce(false)

        // trigger click on 2nd menu item which contains images
        $('.nl-symbols-dropdown .item').eq(1).trigger('click')

        // click on first image checkbox
        const chkbox = $('.nl-vbr-overlay-switch input').first()
        chkbox.prop('checked', true)
        chkbox.trigger('change')

        expect(mockCanAdd).toHaveBeenCalled()
        expect(mockAddImage).not.toHaveBeenCalled()
        expect(chkbox.prop('checked')).toBe(false)
      })

      it('should set the color and hide histogram for region labels', () => {
        mockCanAdd.mockReturnValueOnce(true)
        mockAddImage.mockReturnValueOnce({ hex: '#f44336' })
        // trigger click on 2nd menu item which contains images
        $('.nl-symbols-dropdown .item').eq(1).trigger('click')
        // click on first image checkbox
        const chkbox = $('.nl-vbr-overlay-switch input').first()
        chkbox.prop('checked', true)
        chkbox.trigger('change')

        const expectedImage = mockResults.data.results.RegionImage.values[0][0]
        expect(mockAddImage).toHaveBeenCalledWith('image_0_0', expectedImage.image, expectedImage.min, expectedImage.max, expectedImage.q95)
        expect($('.nl-vbr-overlay-hist').first().is(':visible')).toBe(false)
        expect($('.nl-vbr-overlay-switch label').first().css('color')).toBe('rgb(244, 67, 54)') // #f44336 = rgb(244, 67, 54)
      })

      it('should set center when center btn is clicked', () => {
        // trigger click on 2nd menu item which contains images
        $('.nl-symbols-dropdown .item').eq(1).trigger('click')

        // click on first image checkbox
        mockCanAdd.mockReturnValueOnce(true)
        mockAddImage.mockReturnValueOnce({ lut: 'Viridis' })
        const chkbox = $('.nl-vbr-overlay-switch input').first()
        chkbox.prop('checked', true)
        chkbox.trigger('change')

        // trigger click on first image center btn
        $('.nl-vbr-overlay-center').first().trigger('click')
        expect(mockCenter).toHaveBeenCalledWith(mockResults.data.results.RegionImage.values[0][0].center)
      })

      it('should call showHistogram when histogram btn is clicked', () => {
        // trigger click on 2nd menu item which contains images
        $('.nl-symbols-dropdown .item').eq(1).trigger('click')

        // click on second image checkbox
        mockCanAdd.mockReturnValueOnce(true)
        mockAddImage.mockReturnValueOnce({ lut: 'Viridis' })
        const chkbox = $('.nl-vbr-overlay-switch input').last()
        chkbox.prop('checked', true)
        chkbox.trigger('change')

        // trigger click on second image histogram btn
        $('.nl-vbr-overlay-hist').last().trigger('click')
        expect(mockHist).toHaveBeenCalledWith('image_1_0')
      })
    })
  })

  describe('fetchNewTableData', () => {
    const engine = 'neuroquery'
    let data
    let callback

    beforeEach(() => {
      data = { draw: 1, order: [], start: 0, length: 25 }
      callback = jest.fn()
      $.ajax = mockJqueryGet.mockImplementation(() => {
        return $.Deferred().resolve(mockResults)
      })
      sc.results = mockResults.data
    })

    it('should send query to status endpoint when active symbol is a query result', () => {
      const activeSymbol = 'ActivationGivenTerm'
      sc.activeSymbol = activeSymbol
      const queryData = {
        symbol: activeSymbol,
        start: data.start,
        length: data.length
      }
      sc.fetchNewTableData(data, callback, null)
      expect($.ajax).toHaveBeenCalledWith({
        url: `${API_ROUTE.status}/${mockResults.data.uuid}`,
        type: 'get',
        data: queryData
      })
    })

    it('should send query to symbol endpoint when active symbol is an engine symbol', () => {
      sc.symbols = sc.results
      sc.results = undefined
      sc.engine = engine
      const activeSymbol = 'TermInStudyTFIDF'
      sc.activeSymbol = activeSymbol
      const queryData = {
        symbol: activeSymbol,
        start: data.start,
        length: data.length
      }
      sc.fetchNewTableData(data, callback, null)
      expect($.ajax).toHaveBeenCalledWith({
        url: `${API_ROUTE.symbols}/${engine}`,
        type: 'get',
        data: queryData
      })
    })

    it('should call callback with result data', () => {
      const activeSymbol = 'ActivationGivenTerm'
      sc.activeSymbol = activeSymbol
      const resultData = {
        draw: data.draw,
        recordsTotal: mockResults.data.results.ActivationGivenTerm.size,
        recordsFiltered: mockResults.data.results.ActivationGivenTerm.size,
        data: mockResults.data.results.ActivationGivenTerm.values
      }
      sc.fetchNewTableData(data, callback, null)
      expect(callback).toHaveBeenCalledWith(resultData)
    })
  })
})
