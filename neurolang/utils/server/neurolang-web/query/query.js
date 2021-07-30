import 'codemirror/lib/codemirror.css'
import 'codemirror/theme/xq-light.css'
import 'codemirror/addon/display/autorefresh'
import 'codemirror/mode/python/python'
import CodeMirror from 'codemirror'
import './query.css'
import $ from 'jquery'
import { showQueryResults } from '../results/results'
import { API_ROUTE } from '../constants'

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

const runQueryBtn = $('#runQueryBtn')
const queryAlert = $('#queryAlert')

class QueryManager {
  constructor () {
    this.socket = new WebSocket(API_ROUTE.statementsocket)
  }

  submitQuery () {
    const query = editor.getValue()
    const msg = { query: query }
    this.socket.send(JSON.stringify(msg))
    this.socket.onmessage = this._onmessage
  }

  _onmessage (event) {
    const msg = JSON.parse(event.data)
    if (msg.status !== 'ok') {
      setAlert('An error occured while submitting your query.')
    } else {
      if (msg.data.cancelled) {
        // query was cancelled
        setAlert('The query was cancelled.')
      } else if (msg.data.done) {
        if ('errorName' in msg.data) {
          // query returned an error
          setQueryError(msg)
        } else {
          // query was sucessfull
          setAlert()
          showQueryResults(msg)
        }
      } else {
        // query is either still running or has not yet started
        setAlert('Your query is running. Results will display below when available..')
      }
    }
  }
}

const qm = new QueryManager()
runQueryBtn.on('click', () => qm.submitQuery())

function setQueryError (data) {
  setAlert(data.data.message, true)
}

function setAlert (msg, error = false) {
  queryAlert.removeClass('error')
  queryAlert.removeClass('info')
  if (typeof msg !== 'undefined') {
    const styleClass = error ? 'error' : 'info'
    queryAlert.addClass(styleClass)
    queryAlert.find('.content p').text(msg)
    // queryAlert.innerHTML = `<div class="content"><p>${msg}</p></div>`
    queryAlert.show()
  } else {
    queryAlert.hide()
  }
}
