import 'codemirror/lib/codemirror.css'
import 'codemirror/theme/xq-light.css'
import 'codemirror/addon/hint/show-hint.css'
import 'codemirror/addon/display/autorefresh'
import 'codemirror/addon/hint/show-hint'

import './datalog'
import CodeMirror from 'codemirror'

import './query.css'
import $ from '../jquery-bundler'
import { SymbolsController } from '../symbols/symbols'
import { API_ROUTE } from '../constants'


function myCompletions(context) {
  let word = context.matchBefore(/\w*/)
  if (word.from == word.to && !context.explicit)
    return null
  return {
    from: word.from,
    options: [
      {label: "match", type: "keyword"},
      {label: "hello", type: "variable", info: "(World)"},
      {label: "magic", type: "text", apply: "⠁⭒*.✩.*⭒⠁", detail: "macro"}
    ]
  }
}

/**
 * Class to manage query submission.
 */
export class QueryController {
  constructor () {
    this.engine = 'destrieux'
    /// Initialize the query box with the CodeMirror plugin
    this.queryTextArea = document.querySelector('#queryTextArea')
    this.editor = CodeMirror.fromTextArea(this.queryTextArea, {
      extraKeys: { "Ctrl-Space": "autocomplete" },
      hintOptions: {hint: () => this.completion() },
      mode: 'datalog',
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
    this.inputFileReader = $('#queryFileInput')
    this.inputFileReader.on('change', () => this._loadCodeAsFile())
    this.downloadLink = $('.nl-query-download')
    this.downloadLink.on('click', () => this._saveCodeAsFile())
    $('.nl-query-upload').on('click', (e) => {
      this.inputFileReader.trigger('click')
      e.preventDefault()
    })

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

  completion() {
    const cm = this.editor
    const query = cm.getValue().trim().replace('"', '\"')

    console.log("Looking for completion")
    console.log(cm)

    const cursor = cm.getCursor()
    const start = cursor.ch
    const end = cursor.ch

    const msg = {
      query: query,
      engine: this.engine,
      line: cursor.line,
      character: start,
    }

    return new Promise(function (accept) {
      setTimeout(function () {
        console.log("Sending request for completions %s", query)
        $.ajax({
          url: `${API_ROUTE.completion}`,
          type: 'get',
          data: msg
        }).done((data) => {
          console.log(data)
          if ('status' in data && data.status === 'ok') {
            // Fetching the symbols done with success
            const completions = data.data.completions.map(
              function (item) {
                if (item == "NEW IDENTIFIER") {
                  return {text: "", displayText: "-- insert a new identifier --"}
                } else {
                  return item
                }
              }
            )
            console.log("Completions received: %s", completions)
            return accept({
              list: completions,
              from: CodeMirror.Pos(cursor.line, start),
              to: CodeMirror.Pos(cursor.line, end),
            })
          } else {
            console.log('No completion available ')
            console.log(data)
            return accept({
              list: [],
              from: CodeMirror.Pos(cursor.line, start),
              to: CodeMirror.Pos(cursor.line, end),            
            })
          }
        })
       },
        100
      )
    })
  }

  _submitQuery () {
    const query = this.editor.getValue().trim()
    if (query) {
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
          this._clearAlert()
          // query returned an error
          const errorDoc = 'errorDoc' in msg.data ? msg.data.errorDoc : undefined
          this._setAlert('error', msg.data.message, msg.data.errorName, errorDoc)
          if ('line_info' in msg.data) {
            this._setEditorMarks(msg.data.line_info)
          }
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

  _setEditorMarks (lineInfo) {
    const from = { line: lineInfo.line, ch: lineInfo.col === 0 ? 0 : lineInfo.col - 1 }
    const to = { line: lineInfo.line, ch: lineInfo.col + 1 }
    const marker = $(`<div data-position="bottom center" data-content="${lineInfo.text}">❌</div>`)
    marker.css('color', '#822')
    marker.popup()
    this.editor.setGutterMarker(lineInfo.line, 'marks', marker[0])
    this.editor.markText(from, to, {
      css: 'text-decoration: red double underline; text-underline-offset: .3em;'
    })
  }

  _saveCodeAsFile () {
    this.downloadLink.attr('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(this.editor.getValue()))
    this.downloadLink.attr('download', 'export.neurolang')
  }

  _loadCodeAsFile () {
    const reader = new FileReader()
    reader.onload = (e) => {
      this.editor.setValue(e.target.result)
    }
    reader.onerror = (e) => {
      $('body').toast({
        class: 'error',
        message: 'An error occured while reading your file !'
      })
    }
    reader.readAsText(this.inputFileReader.get(0).files[0])
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
    this.editor.clearGutter('marks')
    this.editor.getAllMarks().forEach((elt) => elt.clear())
  }
}
