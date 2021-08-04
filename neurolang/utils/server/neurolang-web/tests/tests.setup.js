/**
 * Setup script for tests.
 *
 * Defines window.URL.createObjectURL which is not yet implemented by JEST
 * but which Plotly.js requires
 */
function noOp () { }
if (typeof window.URL.createObjectURL === 'undefined') {
  Object.defineProperty(window.URL, 'createObjectURL', { value: noOp })
}
