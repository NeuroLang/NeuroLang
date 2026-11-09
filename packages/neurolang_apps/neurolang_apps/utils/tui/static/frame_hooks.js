/*
 * What a page shown in a frame of the REPL in the browser does with it, inlined
 * in the brain viewer and in the graph of a query (see doc/_static/repl/
 * frames.js). The page takes the theme the REPL sends it, a message, as the
 * attribute data-theme of its html tag, which its styles follow, and tells a
 * 'neurolang-theme' event to its script; and when framed it says how tall it
 * is, so that its frame is as tall and has no scrollbars of its own. On its
 * own, with no theme sent, it follows the system's.
 */
(function () {
  var root = document.documentElement;
  window.addEventListener('message', function (event) {
    var data = event.data;
    if (event.source !== window.parent || !data ||
        data.type !== 'neurolang-theme' ||
        (data.theme !== 'light' && data.theme !== 'dark')) return;
    root.dataset.theme = data.theme;
    window.dispatchEvent(new CustomEvent('neurolang-theme', { detail: data.theme }));
  });
  if (window.parent === window || !window.ResizeObserver) return;
  root.style.overflow = 'hidden';
  var said = -1;
  new ResizeObserver(function () {
    var height = Math.ceil(root.getBoundingClientRect().height);
    if (height === said) return;
    said = height;
    parent.postMessage({ type: 'neurolang-height', height: height }, '*');
  }).observe(root);
})();
