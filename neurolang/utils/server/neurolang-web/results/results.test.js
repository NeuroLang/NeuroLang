import '../tests/tests.setup'
import '@testing-library/jest-dom/extend-expect'
import $ from '../jquery-bundler'
import 'datatables.net'
import dt from 'datatables.net-se'
import 'semantic-ui-css'
import { ResultsController } from './results'

const mockShowViewer = jest.fn()
const mockHideViewer = jest.fn()
const mockAddImage = jest.fn()
const mockCanAdd = jest.fn()
const mockCenter = jest.fn()
const mockHist = jest.fn()

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
  })
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
        values: [[2.0, 77.0, 46.0, 0.0015321803045817697], [4.0, 63.0, 34.0, 0.00023572711035194132], [5.0, 64.0, 38.0, 0.000419401098292473], [8.0, 54.0, 38.0, 0.001412803281281338], [12.0, 61.0, 40.0, 0.0018112978377986602], [12.0, 69.0, 36.0, 0.00045956587380234065], [14.0, 38.0, 40.0, 0.0018165763699549008], [14.0, 44.0, 32.0, 0.0003255290865078548], [14.0, 45.0, 54.0, 0.0008038844221973434], [14.0, 46.0, 32.0, 0.0014595104937104011]]
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
        values: [[{ image: 'someEncodedImageData', hash: '89er8798awre', center: [], max: 0.008, min: 0 }], [{ image: 'anotherEncodedImage', hash: '456eras8098ert', center: [], max: 0.009, min: 0.00002 }]]
      }
    }
  }
}

function getTestHTML () {
  const HTML = `
    <div id="queryContainer" class="ui grid container">
    </div>
    <div class="ui container nl-results-viewer-container">
        <div id="resultsContainer" class="nl-results-container">
        <div class="ui top attached tabular menu nl-results-tabs">
        </div>
        <div class="ui bottom attached tab segment active">
            <table class='ui compact striped table nl-result-table' id='tabTable' width="100%"></table>
        </div>
        </div>

        <div id="nlPapayaContainer" class="nl-papaya-container">
          <div id="nlPapayaParent" class="nl-papaya-parent"></div>
          
          <div class="ui message warning nl-papaya-alert">
              <div class="header">You cannot add more than 8 overlays</div>
              <p>Please unselect an overlay to add a new one</p>
          </div>
        </div>
    </div>
    `
  return HTML
}

describe('ResultsController', () => {
  let rc
  beforeEach(() => {
    document.body.innerHTML = getTestHTML()
    rc = new ResultsController()
  })

  it('should create', () => {
    expect(rc).toBeDefined()
  })

  describe('showQueryResults', () => {
    let mockFetchTableData
    let mockTableData
    beforeEach(() => {
      mockShowViewer.mockClear()
      mockHideViewer.mockClear()
      mockTableData = {
        draw: 1,
        recordsTotal: 500,
        recordsFiltered: 500,
        data: mockResults.data.results.ActivationGivenTerm.values
      }
      mockFetchTableData = jest.spyOn(rc, 'fetchNewTableData').mockImplementation((data, callback, settings) => callback(mockTableData))
    })

    it('should create tabs for all symbols', () => {
      rc.showQueryResults(mockResults)

      const items = $('.nl-results-tabs .item')
      expect(items.length).toBe(3)
      expect(items.first().text()).toBe('ActivationGivenTerm')
      expect($(items.get(1)).text()).toBe('TermInStudyTFIDF')
      expect(items.last().text()).toBe('RegionImage')
    })

    it('should display the first tab', () => {
      rc.showQueryResults(mockResults)
      expect(rc.activeSymbol).toBe('ActivationGivenTerm')

      expect(mockFetchTableData).toHaveBeenCalled()
      // Expect 1 row per item + header
      expect($('#tabTable tr').length).toBe(mockTableData.data.length + 1)
    })

    describe('image results', () => {
      beforeEach(() => {
        rc.showQueryResults(mockResults)
        mockFetchTableData.mockClear()
        mockShowViewer.mockClear()
        mockHideViewer.mockClear()

        rc.tableData = mockResults.data
        mockTableData = {
          draw: 1,
          recordsTotal: 1,
          recordsFiltered: 1,
          data: mockResults.data.results.RegionImage.values
        }
      })

      afterEach(() => {
        jest.clearAllMocks()
      })

      it('should show papaya viewer when tab has VBROverlay type', () => {
        // trigger click on last tab which contains images
        $('.nl-results-tabs .item').last().trigger('click')
        expect(mockShowViewer).toHaveBeenCalled()
        expect(mockHideViewer).not.toHaveBeenCalled()
        expect(mockFetchTableData).toHaveBeenCalled()
      })

      it('should show controls when displaying brain images', () => {
        // trigger click on last tab which contains images
        $('.nl-results-tabs .item').last().trigger('click')
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
        // trigger click on last tab which contains images
        $('.nl-results-tabs .item').last().trigger('click')
        mockCanAdd.mockReturnValueOnce(true)

        // click on second image checkbox
        const chkbox = $('.nl-vbr-overlay-switch input').last()
        chkbox.prop('checked', true)
        chkbox.trigger('change')

        const expectedImage = mockResults.data.results.RegionImage.values[1][0]
        expect(mockCanAdd).toHaveBeenCalled()
        expect(mockAddImage).toHaveBeenCalledWith('image_1_0', expectedImage.image, expectedImage.min, expectedImage.max)
      })

      it('should not add image if max images already added', () => {
        mockCanAdd.mockReturnValueOnce(false)

        // trigger click on last tab which contains images
        $('.nl-results-tabs .item').last().trigger('click')

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
        mockAddImage.mockReturnValueOnce('#f44336')
        // trigger click on last tab which contains images
        $('.nl-results-tabs .item').last().trigger('click')
        // click on first image checkbox
        const chkbox = $('.nl-vbr-overlay-switch input').first()
        chkbox.prop('checked', true)
        chkbox.trigger('change')

        const expectedImage = mockResults.data.results.RegionImage.values[0][0]
        expect(mockAddImage).toHaveBeenCalledWith('image_0_0', expectedImage.image, expectedImage.min, expectedImage.max)
        expect($('.nl-vbr-overlay-hist').first().is(':visible')).toBe(false)
        expect($('.nl-vbr-overlay-switch label').first().css('color')).toBe('rgb(244, 67, 54)') // #f44336 = rgb(244, 67, 54)
      })

      it('should set center when center btn is clicked', () => {
        // trigger click on last tab which contains images
        $('.nl-results-tabs .item').last().trigger('click')

        // click on first image checkbox
        mockCanAdd.mockReturnValueOnce(true)
        const chkbox = $('.nl-vbr-overlay-switch input').first()
        chkbox.prop('checked', true)
        chkbox.trigger('change')

        // trigger click on first image center btn
        $('.nl-vbr-overlay-center').first().trigger('click')
        expect(mockCenter).toHaveBeenCalledWith(mockResults.data.results.RegionImage.values[0][0].center)
      })

      it('should call showHistogram when histogram btn is clicked', () => {
        // trigger click on last tab which contains images
        $('.nl-results-tabs .item').last().trigger('click')

        // click on second image checkbox
        mockCanAdd.mockReturnValueOnce(true)
        const chkbox = $('.nl-vbr-overlay-switch input').last()
        chkbox.prop('checked', true)
        chkbox.trigger('change')

        // trigger click on second image histogram btn
        $('.nl-vbr-overlay-hist').last().trigger('click')
        expect(mockHist).toHaveBeenCalledWith('image_1_0')
      })
    })
  })
})
