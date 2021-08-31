/**
 * Setup script for tests.
 *
 * Defines window.URL.createObjectURL which is not yet implemented by JEST
 * but which Plotly.js requires
 * 
 * Also defines window.$ so that the jquery-bundler.js module which we use
 * also works during tests.
 */
import $ from 'jquery'

function noOp () { }
if (typeof window.URL.createObjectURL === 'undefined') {
  Object.defineProperty(window.URL, 'createObjectURL', { value: noOp })
}

window.$ = window.jQuery = $