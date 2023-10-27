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
        // Prevent the defalt behaviour of the tab key
        event.preventDefault();
        // Get content from the start to the cursor position
        console.log("Shift + Tab detected");
        const cursorPosition = this.editor.getCursor();
        const contentToCursor = this.editor.getRange({line: 0, ch: 0}, cursorPosition);

        this._requestAutocomplete(contentToCursor);
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

    this.okButton = $('#okButton')
    this.okButton.on('click', () => this._handleOkButtonClick());

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

  _requestAutocomplete(content) {
    console.log("Requesting autocomplete for:");
    console.log(content);
//    console.log(`this.engine before : ${this.engine}`);
    // const actEng = this.router.getActive.Engine();
//    var activeEngine = ""
//    if (this.engine) {
//        activeEngine = this.engine
//      }
//    console.log(`this.engine before : ${this.engine}`);
//    console.log(`activeEngine : ${activeEngine}`)
//    console.log(`this.sc.engine ${this.sc.engine}`);
    $.post(API_ROUTE.autocompletion, {text: content, engine: this.engine }, data => {
        // if (data.tokens && data.tokens.length > 0) {
        //     this._showTooltipAtCursor(data.tokens);
        // }
        console.log(`avant - type of data.tokens : ${Object.prototype.toString.call(data.tokens)}`)
        console.log(`avant - data.tokens         : ${data.tokens}`)
        var tokens_to_json = JSON.parse(data.tokens)
        console.log(`avant - type of tokens_to_json : ${JSON.stringify(tokens_to_json, null, 2)}`)
        console.log(`avant - tokens_to_json         : ${tokens_to_json}`)
        //var tmp_tokens = JSON.stringify(data.tokens, null, 2)
        //console.log(`avant - type of tmp_tokens : ${Object.prototype.toString.call(tmp_tokens)}`)
        //console.log(`avant - tmp_tokens         : ${tmp_tokens}`)
        var retrieved_symbols = this._addSymbolsInAutocompletionTokens();
//        console.log(`retrieved_symbols - type : ${Object.prototype.toString.call(retrieved_symbols)}`)
//        console.log(`retrieved_symbols - value : ${JSON.stringify(retrieved_symbols, null, 2)}`)
        //const facets = Object.assign({}, {"tokens" : data.tokens}, retrieved_symbols)
        var facets = Object.assign({}, tokens_to_json, retrieved_symbols)
//        console.log(`facets - type : ${Object.prototype.toString.call(facets)}`)
//        console.log(`facets - value : ${JSON.stringify(facets, null, 2)}`)
        facets = this._cleanAcceptedTokens(facets)

        this._displayFacets(facets);    // Display the facets based on the tokens
        //this._showTooltipAtCursor(data.tokens);  // No tokens passed, so it won't show the tooltip but will set up the cursor activity listener to hide facets
    });
  }

  _cleanAcceptedTokens(facets_obj) {
    var keys_to_remove = []
    var clean_json_object = {}

    for (let key of Object.keys(facets_obj)) {
        if(facets_obj[key].length != 0) {
            //keys_to_remove.push(key)
            clean_json_object[key] = facets_obj[key]
        }
    }
    return clean_json_object
  }

  //_addSymbolsInAutocompletionTokens(tokens) {
  _addSymbolsInAutocompletionTokens() {
    console.log(`__in _addSymbolsInAutocompletionTokens()___`)
    //console.log("Displaying tokens:", tokens);

    // ajouter les elememts ici
    var curSymbols = Object.keys(window.qc.sc.symbols.results)
    .filter(elt => !window.qc.sc.symbols.results[elt].function && !window.qc.sc.symbols.results[elt].command)
    var curFunctions = Object.keys(window.qc.sc.symbols.results)
    .filter(elt => window.qc.sc.symbols.results[elt].function)
    var curCommands = Object.keys(window.qc.sc.symbols.results)
    .filter(elt => window.qc.sc.symbols.results[elt].command)

    this.curSymbols = curSymbols
    this.curFunctions = curFunctions
    this.curCommands = curCommands

    console.log(`Symbols : ${curSymbols}`)
    console.log(`Symbols type : ${Object.prototype.toString.call(curSymbols)}`)
    console.log(`Functions : ${curFunctions}`)
    console.log(`Functions type : ${Object.prototype.toString.call(curFunctions)}`)
    console.log(`Commands : ${curCommands}`)
    console.log(`Commands type : ${Object.prototype.toString.call(curCommands)}`)

    //return [].concat(tokens, curSymbols, curFunctions, curCommands)

    return {
        "symbols" : curSymbols,
        "functions" : curFunctions,
        "commands" : curCommands
    }
  }

  _displayFacets(facets_obj) {
    const leftFacet = document.getElementById("leftFacet");
    const rightFacet = document.getElementById("rightFacet");
    const facetsContainer = document.getElementById("facetsContainer");
    const okButton = document.getElementById("okButton");

    // Clear previous facet items
    leftFacet.innerHTML = "";
    rightFacet.innerHTML = "";

    // Add deselect to the left facet
    const deselectItem = document.createElement("div");
    deselectItem.textContent = "deselect";
    deselectItem.classList.add("facet-item");
    leftFacet.appendChild(deselectItem);

    // Add facets_obj keys to left facet
    for (let key of Object.keys(facets_obj)) {
        const item = document.createElement("div");
        item.textContent = key;
        item.classList.add("facet-item");
        leftFacet.appendChild(item);
    }

    // Handle left facet item click
    leftFacet.addEventListener("click", (e) => {
        const itemContent = e.target.textContent;
        rightFacet.innerHTML = ""; // Clear previous items

        if (itemContent === "deselect") {
            for (let key of Object.keys(facets_obj)) {
                for (let value of facets_obj[key]) {
                    const item = document.createElement("div");
                    item.textContent = value;
                    item.classList.add("facet-item");
                    rightFacet.appendChild(item);
                }
            }
        } else {
            for (let value of facets_obj[itemContent]) {
                const item = document.createElement("div");
                item.textContent = value;
                item.classList.add("facet-item");
                rightFacet.appendChild(item);
            }
        }
    });

    // Handle right facet item click
    rightFacet.addEventListener('click', this._handleRightFacetClick.bind(this));

    // Handle 'ok' button click
    //okButton.removeEventListener("click", this._handleOkButtonClick.bind(this))
    //okButton.addEventListener("click", this._handleOkButtonClick.bind(this));

    // Display the facets container
    //facetsContainer.removeAttribute("hidden");
    facetsContainer.style.display = 'flex';         // overrides the 'display: none;' from the CSS.

    // This will always run, whether there are tokens or not
    this.editor.on("cursorActivity", () => {
        //const facetsContainer = document.getElementById("facetsContainer");
        //facetsContainer.setAttribute("hidden", true);
        facetsContainer.style.display = 'none';     // set the facets container back to be hidden
    });
  }

  _handleRightFacetClick(event) {
    // Check if the clicked element is an item within the right facet
    if (event.target.classList.contains('facet-item')) {
        // Deselect previously selected item
        const previouslySelected = event.currentTarget.querySelector('.selected');
        if (previouslySelected) {
            previouslySelected.classList.remove('selected');
        }

        // Mark the clicked item as selected
        event.target.classList.add('selected');
    }
  }

  _handleOkButtonClick() {
    console.log(`___in _handleOkButtonClick() -> clicked___`)
    const rightFacet = document.getElementById("rightFacet");
    const selectedItem = rightFacet.querySelector(".selected");

    console.log(`right facet : ${rightFacet}`)
    console.log(`selected item : ${selectedItem}`)
    console.log(`selected item.textContent : ${selectedItem.textContent}`)

    if (selectedItem && selectedItem.textContent !== "deselect") {
        console.log(`  selectedItem && selectedItem.textContent !== "deselect"`)
        //const cursorPos = this.editor.getCursor();
        //this.editor.replaceRange(selectedItem.textContent, cursorPos);

        const selectedValue = selectedItem.textContent;
    	const doc = this.editor.getDoc();

    	console.log(`  selectedValue : ${selectedValue}`)


    	// Insert the value at the current cursor position.
    	const cursor = doc.getCursor();
    	doc.replaceRange(selectedValue, cursor);
    	//doc.replaceRange("Test insertion", cursor)

    	// Move cursor to end of inserted value
    	const endPos = { line: cursor.line, ch: cursor.ch + selectedValue.length };
    	doc.setCursor(endPos);

    }

    //// Hide the facets and 'ok' button
    //const facetsContainer = document.getElementById("facetsContainer");
    //facetsContainer.style.display = 'none';     // set the facets container back to be hidden
  }


  _showTooltipAtCursor(tokens) {
    console.log(`___in _showTooltipAtCursor()___`)
    console.log("Displaying tokens:", tokens);

    if (tokens.length > 0) {
        const cursorPos = this.editor.cursorCoords(true, "page");
        const tooltip = $('<div class="autocomplete-tooltip"></div>').text(tokens.join(", ")).css({
            top: cursorPos.top + "px",
            left: cursorPos.left + "px"
        }).appendTo(document.body);

        // Close the tooltip on cursor movement or click outside
        this.editor.on("cursorActivity", () => {
            tooltip.remove();
        });
        $(document).on('click', () => {
            tooltip.remove();
        });
    }

    // This will always run, whether there are tokens or not
    this.editor.on("cursorActivity", () => {
        const facetsContainer = document.getElementById("facetsContainer");
        facetsContainer.setAttribute("hidden", true);
    });
  }
}
