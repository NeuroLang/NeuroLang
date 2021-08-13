/**
 * We want to import $ from jQuery, and set the global variables $ and jQuery to
 * be this.
 * This needs to be done before we import other modules which might depend on jQuery
 * and would access it from window.$
 * See https://stackoverflow.com/questions/34338411/how-to-import-jquery-using-es6-syntax
 */

// import 'jquery'
// window.$ = window.jQuery = $
// export default $

export default window.$
