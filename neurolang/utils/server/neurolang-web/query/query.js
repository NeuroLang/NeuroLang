import 'codemirror/lib/codemirror.css'
import 'codemirror/theme/xq-light.css'
import 'codemirror/addon/display/autorefresh'
import './datalog'
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
    this.tokens = []
    this.curSymbols = []
    this.curFunctions = []
    this.curCommands = []
    this.autocompletionSymbols = {}
    this.engine = 'destrieux'
    /// Initialize the query box with the CodeMirror plugin
    this.queryTextArea = document.querySelector('#queryTextArea')
    this.editor = CodeMirror.fromTextArea(this.queryTextArea, {
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

    this.editor.on("keydown", (cm, event) => {
      console.log(`keydown detected ${event.key}`);
      if (event.shiftKey && event.key == 'Tab') {
        // Prevent the default behaviour of the tab key
        event.preventDefault();
        //this._requestAutocomplete(contentToCursor);
        this._requestAutocomplete();
      }
    });

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

    /// Atach the change event listener to the right facet here to make sure we avoid duplicates.
    /// Otherwise the event lister would have to be removed before its creation.
    document.getElementById("rightFacet").addEventListener('change', (event) => {
        this._handleRightFacetClick(event);
    });

    /// Autocompletion Manager
    //this.ac = new AutocompletionController()

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

  _requestAutocomplete() {
    console.log("Requesting autocomplete for:");
    //console.log(content);
    // get the entire text from the CodeMirror instance
    const allText = this.editor.getValue();

    // get the cursor's current position
    const cursorPos = this.editor.getCursor();

    // get the line number where the cursor is
    const lineNumber = cursorPos.line;

    // get the position in the whole text of the first character of that line
    const lineStartPos = this.editor.indexFromPos({line: lineNumber, ch: 0});

    // get the position in the whole text of the cursor
    const cursorIndex = this.editor.indexFromPos(cursorPos);

    $.post(API_ROUTE.autocompletion, {text: allText, engine: this.engine, line: lineNumber, startpos: lineStartPos, endpos: cursorIndex}, data => {
        var facets = JSON.parse(data.tokens)
        this._displayFacets(facets);    // Display the facets based on the tokens
    });
  }

  _displayFacets(facets_obj) {
    const leftFacet = document.getElementById("leftFacet");
    const rightFacet = document.getElementById("rightFacet");
    const facetsContainer = document.getElementById("facetsContainer");
//    const okButton = document.getElementById("okButton");
    const myDropdown = this.sc.dropdown;

    // Clear previous facet items
    leftFacet.innerHTML = "";
    rightFacet.innerHTML = "";

    // Add 'All' option to the left facet
    let deselectOption = document.createElement("option")
    deselectOption.textContent = "All"
    leftFacet.appendChild(deselectOption)

    // Add facets_obj keys to left facet
    for (let key of Object.keys(facets_obj)) {
        let option = document.createElement("option")
        option.textContent = key;
        leftFacet.appendChild(option);
    }

    // Make sure to remove any previous event listener to avoid duplicates
    if (this._currentLeftFacetHandler) {
  	    leftFacet.removeEventListener('change', this._currentLeftFacetHandler);
	}

    // Create a new handler function that has access to facets_obj
	this._currentLeftFacetHandler = (event) => this._handleLeftFacetClick(event, facets_obj);

	// Attach the new event listener
	leftFacet.addEventListener("change", this._currentLeftFacetHandler);

    // Display the facets container
    facetsContainer.style.display = 'flex';         // overrides the 'display: none;' from the CSS.

    // This will always run, whether there are tokens or not
    this.editor.on("cursorActivity", () => {
        facetsContainer.style.display = 'none';     // set the facets container back to be hidden
    });
  }

  _handleLeftFacetClick(event, facets_obj) {
    const leftFacet = document.getElementById("leftFacet");
    const rightFacet = document.getElementById("rightFacet");
    const facetsContainer = document.getElementById("facetsContainer");

    // Retrieve the selected option
    const selectedKey = event.target.value;

    // Clear previous options
    while (rightFacet.firstChild) {
        rightFacet.removeChild(rightFacet.firstChild);
    }

    if (selectedKey === "All") {
        for (let key of Object.keys(facets_obj)) {
            for (let value of facets_obj[key]) {
                let option = document.createElement("option")
                option.value = value;
                option.textContent = value;
                rightFacet.appendChild(option);
            }
        }
    } else if (facets_obj[selectedKey]) {
        for (let value of facets_obj[selectedKey]) {
            const option = document.createElement("option")
            option.value = value;
            option.textContent = value;
            rightFacet.appendChild(option);
        }
    }

    this.editor.focus();
  }

  _handleRightFacetClick(event) {
    const rightFacet = document.getElementById("rightFacet")
    const selectedValue = rightFacet.value;
    const leftFacet = document.getElementById("leftFacet")
    const selectedKey = leftFacet.value

    // Synchronize with dropdown if the selected key in the left facet is one of the specified values
//    if (['commands', 'functions', 'base symbols', 'query symbols'].includes(selectedKey)) {
//  	    // this.sc.dropdown is a jQuery object wrapping the dropdown element
//  	    this.sc.dropdown.dropdown('set selected', selectedValue);
//    }

    const dropdownItems = []
    this.sc.dropdown.find('.menu .item').each(function() {
        dropdownItems.push($(this).data('value'));
    });
    if (dropdownItems.includes(selectedValue)) {
  	    // this.sc.dropdown is a jQuery object wrapping the dropdown element
  	    this.sc.dropdown.dropdown('set selected', selectedValue);
    }

    if (selectedValue && selectedValue !== "All") {
        console.log(`selectedValue && selectedValue !== "All"`)

        const cursorPos = this.editor.getCursor();              // get the cursor position in the CodeMirror editor
        this.editor.replaceRange(selectedValue, cursorPos);     // insert the selected value at the current cursor position
        const endPos = { line: cursorPos.line, ch: cursorPos.ch + selectedValue.length }    // calculate the end position based on the length of the inserted value

        // check if the selected value starts with '<' and ends with '>'
        if (selectedValue.startsWith('<') && selectedValue.endsWith('>')) {
            // select the text that was just inserted
            this.editor.setSelection(cursorPos, endPos)
        } else {
            // Move cursor to end of inserted value
            this.editor.setCursor(endPos);
        }
    }

    // Hide the facets and 'ok' button as they are no longer needed
    const facetsContainer = document.getElementById("facetsContainer");
    facetsContainer.style.display = 'none';     // set the facets container back to be hidden

    // Refocus the CodeMirror editor to keep the cursor visible in the textarea
    this.editor.focus();
  }
}
