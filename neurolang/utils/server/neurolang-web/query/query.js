import 'codemirror/lib/codemirror.css'
import 'codemirror/theme/xq-light.css'
import 'codemirror/addon/display/autorefresh'
import 'codemirror/mode/python/python'
import CodeMirror from 'codemirror'
import './query.css'
import $ from '../jquery-bundler'
import { hideQueryResults, showQueryResults } from '../results/results'
import { API_ROUTE } from '../constants'

/// Initialize the query box with the CodeMirror plugin
const queryTextArea = document.querySelector('#queryTextArea')
const editor = CodeMirror.fromTextArea(queryTextArea, {
  mode: 'python',
  theme: 'xq-light',
  autoRefresh: true,
  lineNumbers: true,
  lineWrapping: true,
  gutters: [
    'CodeMirror-linenumbers',
    { className: 'marks', style: 'width: .9em' }
  ]
})

/// Query Button & Alert box
const runQueryBtn = $('#runQueryBtn')
const queryAlert = $('#queryAlert')

/**
 * Class to manage query submission.
 */
class QueryManager {
  submitQuery () {
    const query = editor.getValue()
    const msg = { query: query }
    // create a new WebSocket and set the event listeners on it
    this.socket = new WebSocket(API_ROUTE.statementsocket)
    this.socket.onerror = this._onerror
    this.socket.onmessage = (event) => this._onmessage(event)
    this.socket.onopen = () => {
      runQueryBtn.addClass('loading')
      runQueryBtn.prop('disabled', true)
      hideQueryResults()
      this.socket.send(JSON.stringify(msg))
    }
  }

  _onerror (event) {
    setAlert('warning', 'An error occured while connecting to the server.')
  }

  _onmessage (event) {
    const msg = JSON.parse(event.data)
    if (!('status' in msg) || msg.status !== 'ok') {
      setAlert('error', msg, 'An error occured while submitting your query.')
      this._finishQuery()
    } else {
      if (msg.data.cancelled) {
        // query was cancelled
        setAlert('warning', undefined, 'The query was cancelled.')
        this._finishQuery()
      } else if (msg.data.done) {
        if ('errorName' in msg.data) {
          // query returned an error
          const errorDoc = 'errorDoc' in msg.data ? msg.data.errorDoc : undefined
          setAlert('error', msg.data.message, msg.data.errorName, errorDoc)
          this._finishQuery()
        } else {
          // query was sucessfull
          clearAlert()
          this._finishQuery()
          showQueryResults(msg)
        }
      } else {
        // query is either still running or has not yet started
        setAlert('info', 'Results will display below when available..', 'Your query is running')
      }
    }
  }

  _finishQuery () {
    this.socket.close()
    runQueryBtn.removeClass('loading')
    runQueryBtn.prop('disabled', false)
  }
}

/// Create a QueryManager and bind the Query Button to the submit query method
const qm = new QueryManager()
runQueryBtn.on('click', () => qm.submitQuery())

/**
 * Set a message in the message box below the query box.
 * @param {string} style the style class for the message (info, error, success, warning)
 * @param {*} content the content for the message
 * @param {*} header the header for the message
 */
function setAlert (style, content, header, help) {
  const qHeader = queryAlert.find('.nl-query-header')
  if (typeof header !== 'undefined') {
    qHeader.text(header)
  } else {
    qHeader.empty()
  }
  const qMsg = queryAlert.find('.nl-query-message')
  if (typeof content !== 'undefined') {
    qMsg.text(content)
  } else {
    qMsg.empty()
  }
  const qHelp = queryAlert.find('.nl-query-help')
  if (typeof help !== 'undefined') {
    qHelp.text(help)
    qHelp.show()
  } else {
    qHelp.empty()
    qHelp.hide()
  }
  queryAlert.removeClass('info error warning success')
  queryAlert.addClass(style)
  queryAlert.show()
}

function clearAlert () {
  queryAlert.hide()
}
