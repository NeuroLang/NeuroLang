import '../tests/tests.setup'
import '@testing-library/jest-dom/extend-expect'
import $ from '../jquery-bundler'
import 'datatables.net'
import dt from 'datatables.net-se'
import { ResultsController } from './results'

const mockShowViewer = jest.fn()
const mockHideViewer = jest.fn()

jest.mock('../papaya/viewer', () => ({
  PapayaViewer: jest.fn().mockImplementation(() => {
    return { showViewer: mockShowViewer, hideViewer: mockHideViewer }
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
      ActivationGivenTerm: { row_type: ["<class 'int'>", "<class 'int'>", "<class 'int'>", "<class 'float'>"], columns: ['i', 'j', 'k', 'PROB'], size: 27697, values: [[2.0, 77.0, 46.0, 0.0015321803045817697], [4.0, 63.0, 34.0, 0.00023572711035194132], [5.0, 64.0, 38.0, 0.000419401098292473], [8.0, 54.0, 38.0, 0.001412803281281338], [12.0, 61.0, 40.0, 0.0018112978377986602], [12.0, 69.0, 36.0, 0.00045956587380234065], [14.0, 38.0, 40.0, 0.0018165763699549008], [14.0, 44.0, 32.0, 0.0003255290865078548], [14.0, 45.0, 54.0, 0.0008038844221973434], [14.0, 46.0, 32.0, 0.0014595104937104011], [14.0, 46.0, 38.0, 0.0010778211028046204], [14.0, 48.0, 34.0, 0.0022749369671648755], [14.0, 48.0, 45.0, 0.0005455797760476706], [14.0, 49.0, 46.0, 0.00040957704408603273], [14.0, 52.0, 36.0, 0.0003928925850236022], [14.0, 52.0, 38.0, 0.0019686322640022234], [14.0, 52.0, 53.0, 0.0012841697619640415], [14.0, 54.0, 30.0, 0.0024709079363821277], [14.0, 54.0, 44.0, 0.002730697165150301], [14.0, 56.0, 44.0, 0.002730697165150301], [14.0, 56.0, 48.0, 0.002730697165150301], [14.0, 58.0, 29.0, 0.0006092024906175322], [14.0, 58.0, 36.0, 0.0018165763699549008], [14.0, 59.0, 37.0, 0.0012076905898642518], [14.0, 60.0, 42.0, 0.0022749369671648755]] },
      TermInStudyTFIDF: { row_type: ["<class 'str'>", "<class 'float'>", "<class 'neurolang.frontend.neurosynth_utils.StudyID'>"], columns: ['0', '1', '2'], size: 1049299, values: [['001', 0.055394216111399996, '9862924'], ['001', 0.09387570522489999, '11595392'], ['001', 0.0689931709903, '12077009'], ['001', 0.0996940021344, '12725761'], ['001', 0.09198243946769999, '12880904'], ['001', 0.0903373983476, '12958082'], ['001', 0.199392155694, '14561452'], ['001', 0.0735499288657, '14741317'], ['001', 0.10293863273399999, '14741643'], ['001', 0.079606876159, '15036060']] }
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
            <a class="item active" data-tab="first">First</a>
            <a class="item" data-tab="second">Second</a>
            <a class="item" data-tab="third">Third</a>
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
    beforeEach(() => {
      mockShowViewer.mockClear()
      mockHideViewer.mockClear()
    })

    it('should create tabs for all symbols', () => {
      rc.showQueryResults(mockResults)

      expect($('.nl-results-tabs .item').length).toBe(2)
      expect($('.nl-results-tabs .item').first().text()).toBe('ActivationGivenTerm')
      expect($('.nl-results-tabs .item').last().text()).toBe('TermInStudyTFIDF')
    })
  })
})
