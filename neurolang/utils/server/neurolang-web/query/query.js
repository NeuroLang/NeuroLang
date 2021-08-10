import 'codemirror/lib/codemirror.css'
import 'codemirror/theme/xq-light.css'
import 'codemirror/addon/display/autorefresh'
import 'codemirror/mode/python/python'
import CodeMirror from 'codemirror'
import './query.css'
import $ from '../jquery-bundler'
import { SymbolsController } from '../symbols/symbols'
import { API_ROUTE } from '../constants'

/**
 * Class to manage query submission.
 */
export class QueryController {
  constructor () {
    this.engine = 'destrieux'
    /// Initialize the query box with the CodeMirror plugin
    this.queryTextArea = document.querySelector('#queryTextArea')
    this.editor = CodeMirror.fromTextArea(this.queryTextArea, {
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
    this.runQueryBtn = $('#runQueryBtn')
    this.queryAlert = $('#queryAlert')
    this.runQueryBtn.on('click', () => this._submitQuery())

    /// Results Manager
    this.sc = new SymbolsController()
  }

  /**
   * Set the active engine
   * @param {*} engine
   */
  setRouteEngine (engine, query) {
    this.engine = engine
    this.editor.setValue(query)
    this._clearAlert()
    this.sc.hide()
    this.sc.setRouteEngine(engine)
  }

  _submitQuery () {
    const query = this.editor.getValue()
    const msg = { query: query }
    if (this.engine) {
      msg.engine = this.engine
    }
    // create a new WebSocket and set the event listeners on it
    this.socket = new WebSocket(API_ROUTE.statementsocket)
    this.socket.onerror = this._onerror
    this.socket.onmessage = (event) => this._onmessage(event)
    this.socket.onopen = () => {
      this.runQueryBtn.addClass('loading')
      this.runQueryBtn.prop('disabled', true)
      this.socket.send(JSON.stringify(msg))
    }
  }

  _onerror (event) {
    this._setAlert('warning', 'An error occured while connecting to the server.')
  }

  _onmessage (event) {
    const msg = JSON.parse(event.data)
    if (!('status' in msg) || msg.status !== 'ok') {
      this._setAlert('error', msg, 'An error occured while submitting your query.')
      this._finishQuery()
    } else {
      if (msg.data.cancelled) {
        // query was cancelled
        this._setAlert('warning', undefined, 'The query was cancelled.')
        this._finishQuery()
      } else if (msg.data.done) {
        if ('errorName' in msg.data) {
          // query returned an error
          const errorDoc = 'errorDoc' in msg.data ? msg.data.errorDoc : undefined
          this._setAlert('error', msg.data.message, msg.data.errorName, errorDoc)
          this._finishQuery()
        } else {
          // query was sucessfull
          this._clearAlert()
          this._finishQuery()
          this.sc.setQueryResults(msg)
        }
      } else {
        // query is either still running or has not yet started
        this._setAlert('info', 'Results will display below when available..', 'Your query is running')
      }
    }
  }

  _finishQuery () {
    this.socket.close()
    this.runQueryBtn.removeClass('loading')
    this.runQueryBtn.prop('disabled', false)
  }

  /**
 * Set a message in the message box below the query box.
 * @param {string} style the style class for the message (info, error, success, warning)
 * @param {*} content the content for the message
 * @param {*} header the header for the message
 */
  _setAlert (style, content, header, help) {
    const qHeader = this.queryAlert.find('.nl-query-header')
    if (typeof header !== 'undefined') {
      qHeader.text(header)
    } else {
      qHeader.empty()
    }
    const qMsg = this.queryAlert.find('.nl-query-message')
    if (typeof content !== 'undefined') {
      qMsg.text(content)
    } else {
      qMsg.empty()
    }
    const qHelp = this.queryAlert.find('.nl-query-help')
    if (typeof help !== 'undefined') {
      qHelp.text(help)
      qHelp.show()
    } else {
      qHelp.empty()
      qHelp.hide()
    }
    this.queryAlert.removeClass('info error warning success')
    this.queryAlert.addClass(style)
    this.queryAlert.show()
  }

  _clearAlert () {
    this.queryAlert.hide()
  }
}
