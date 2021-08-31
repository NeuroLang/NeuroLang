/**
 * jQuery is imported as a global $ variable, defined as window.$. In order to be
 * able to import jQuery using es6 modules imports, we simply export window.$ in
 * this module.
 * This avoids having multiple instances of jQuery defined both in window scope,
 * and per module.
 *
 * See https://stackoverflow.com/questions/34338411/how-to-import-jquery-using-es6-syntax
 */

// import 'jquery'
// window.$ = window.jQuery = $
// export default $

export default window.$
