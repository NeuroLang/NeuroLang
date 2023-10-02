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
 * Class to manage query autocompletion.
 */
class QueryAutocompletionController {
	constructor() {
    	this.textarea = document.getElementById('textareaId');
    	this.initEventListeners();
	}

	initEventListeners() {
    	this.textarea.addEventListener('keydown', this.handleKeyDown.bind(this));
	}

	async handleKeyDown(event) {
    	if (event.key === 'Tab') {
        	event.preventDefault();
        	const lastLine = this.getLastLineOfTextarea();
        	const tokens = await this.fetchTokens(lastLine);
        	this.displayTokens(tokens);
    	}
	}

	getLastLineOfTextarea() {
    	const lines = this.textarea.value.split('\n');
    	return lines[lines.length - 1];
	}

	async fetchTokens(code) {
    	const response = await fetch(API_ROUTE.autocompletion, {
        	method: 'POST',
        	body: JSON.stringify({ code: code }),
        	headers: { 'Content-Type': 'application/json' },
    	});
    	const data = await response.json();
    	return data.tokens;
	}

	displayTokens(tokens) {
    	const tokenString = tokens.join(', ');
    	this.textarea.title = tokenString;
	}
}

