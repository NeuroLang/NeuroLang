import 'codemirror/lib/codemirror.css'
import 'codemirror/theme/xq-light.css'
import 'codemirror/addon/display/autorefresh'
import 'codemirror/mode/python/python'
import CodeMirror from 'codemirror'
import './query.css'
import $ from 'jquery'
import { showQueryResults } from '../results/results'

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

const runQueryBtn = document.querySelector('#runQueryBtn')
const queryAlert = document.querySelector('#queryAlert')
runQueryBtn.addEventListener('click', submitQuery)

function submitQuery () {
  // disable query btn
  runQueryBtn.disable = true
  // submit query
  const query = editor.getValue()
  $.post('http://localhost:8888/v1/statement', { query: query })
    .done(function (data) {
      if (data.status === 'ok') {
        setAlert('Your query is running. Results will display below when available..')
        setTimeout(() => pollResults(data.data.uuid), 2000)
      } else {
        setAlert('An error occured while submitting your query.')
      }
    }).fail(function () {
      setAlert('An error occured while submitting your query.')
    }).always(function () {
      runQueryBtn.disable = false
    })
}

function pollResults (queryId) {
  $.get(`http://localhost:8888/v1/status/${queryId}`)
    .done(function (data) {
      if (data.data.cancelled) {
        // query was cancelled
        setAlert('The query was cancelled.')
      } else if (data.data.done) {
        if ('errorName' in data.data) {
          // query returned an error
          setQueryError(data)
        } else {
          // query was sucessfull
          setAlert()
          showQueryResults(queryId, data)
        }
      } else {
        // query is either still running or has not yet started
        setTimeout(() => pollResults(queryId), 3000)
      }
    })
}

function setQueryError (data) {
  setAlert(data.data.message, true)
}

function setAlert (msg, error = false) {
  if (typeof msg !== 'undefined') {
    const styleClass = error ? 'alert-danger' : 'alert-primary'
    queryAlert.innerHTML = `<div class="alert ${styleClass}" role="alert">${msg}</div>`
  } else {
    queryAlert.innerHTML = ''
  }
}
