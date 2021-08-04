import '../tests/tests.setup'
import '@testing-library/jest-dom/extend-expect'
import $ from '../jquery-bundler'
import { QueryManager } from './query'
import WS from 'jest-websocket-mock'
import * as Results from '../results/results'

jest.mock('../results/results', () => ({
  showQueryResults: jest.fn(),
  hideQueryResults: jest.fn()
}))

function getTestHTML () {
  const HTML = `
    <div id="queryContainer" class="ui grid container">
        <div class="sixteen wide column">
        <div class="ui raised segment query-segment">
            <div class="ui top attached label">Write your query on neurological data</div>
            <div class="code-mirror-container">
            <textarea class="ui input" id="queryTextArea">union(region_union(r)) :- destrieux(..., r)</textarea>
            </div>
        </div>
        </div>
        <div class="two wide column">
        <button type="button" class="ui button" id="runQueryBtn">Run query</button>
        </div>
        <div class="fourteen wide column">
        <div class="ui message nl-query-alert" id="queryAlert">
            <div class="header nl-query-header">
            </div>
            <p class="nl-query-message"></p>
            <pre class="nl-query-help"></pre>
        </div>
        </div>
    </div>
    `
  return HTML
}

describe('QueryManager', () => {
  let qm
  beforeEach(() => {
    document.body.innerHTML = getTestHTML()
    qm = new QueryManager()
  })

  it('should create', () => {
    expect(qm).toBeDefined()
  })

  describe('query submission', () => {
    let wsserver
    beforeEach(async () => {
      wsserver = new WS('ws://localhost:8888/v1/statementsocket', { jsonProtocol: true })
    })

    afterEach(() => {
      WS.clean()
    })

    it('should listen for click events on button', () => {
      const spy = jest.spyOn(qm, '_submitQuery')

      $('#runQueryBtn').trigger('click')

      expect(spy).toHaveBeenCalledTimes(1)
    })

    it('should create a websocket and send the query to the server', async () => {
      $('#runQueryBtn').trigger('click')

      await wsserver.connected
      const expectedQuery = { query: 'union(region_union(r)) :- destrieux(..., r)' }
      await expect(wsserver).toReceiveMessage(expectedQuery)
      expect(wsserver).toHaveReceivedMessages([expectedQuery])

      // Check queryBtn is disabled and alert message is displayed
      expect($('#runQueryBtn').is(':disabled')).toBe(true)
    })

    it('should set alert message when query is running', async () => {
      $('#runQueryBtn').trigger('click')

      await wsserver.connected
      const expectedQuery = { query: 'union(region_union(r)) :- destrieux(..., r)' }
      await expect(wsserver).toReceiveMessage(expectedQuery)
      wsserver.send({ status: 'ok', data: { done: false, running: true, cancelled: false } })

      expect($('.nl-query-message').text()).toBe('Results will display below when available..')
      expect($('.nl-query-header').text()).toBe('Your query is running')
      expect($('#runQueryBtn').is(':disabled')).toBe(true)
    })

    it('should set alert message when query fails', async () => {
      $('#runQueryBtn').trigger('click')

      await wsserver.connected
      const expectedQuery = { query: 'union(region_union(r)) :- destrieux(..., r)' }
      await expect(wsserver).toReceiveMessage(expectedQuery)
      const errorName = 'neurolang.exceptions.InvalidQueryError'
      const message = 'Your query is invalid'
      const errorDoc = 'This error is raised when the query is invalid. You should try to make it valid...'
      wsserver.send({ status: 'ok', data: { done: true, running: false, cancelled: false, message, errorName, errorDoc } })

      await wsserver.closed
      expect(qm.socket.readyState).toBe(WebSocket.CLOSED)
      expect($('.nl-query-message').text()).toBe(message)
      expect($('.nl-query-header').text()).toBe(errorName)
      expect($('.nl-query-help').text()).toBe(errorDoc)
      expect($('#runQueryBtn').is(':disabled')).toBe(false)
    })

    it('should show results when query succeeds', async () => {
      $('#runQueryBtn').trigger('click')

      await wsserver.connected
      const expectedQuery = { query: 'union(region_union(r)) :- destrieux(..., r)' }
      await expect(wsserver).toReceiveMessage(expectedQuery)
      const results = { ans: { size: 15, row_type: 'Tuple[float, str]', values: [[0, 1, 2]] } }
      const response = { status: 'ok', data: { done: true, running: false, cancelled: false, results } }
      wsserver.send(response)

      await wsserver.closed
      expect(qm.socket.readyState).toBe(WebSocket.CLOSED)
      expect(Results.showQueryResults).toHaveBeenCalledWith(response)
      expect($('#runQueryBtn').is(':disabled')).toBe(false)
    })
  })
})
