// A stand-in for the engine of the REPL page, to look at its live states
// without Pyodide or the data: it replaces `Worker` and `fetch` and replays
// canned messages of the protocol of worker.js. Classic script, injected in
// the head by build.mjs before the page's own. The state shown is the
// `?state=NAME` of the address (see STATES below).
(() => {
  const state = new URLSearchParams(location.search).get("state") || "empty";
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const $ = (selector) => document.querySelector(selector);

  // --- canned content (Neurosynth-like) ---
  const TERMS = ["pain", "working memory", "reward", "fear", "language comprehension", "attention", "motor", "emotion regulation"];
  const TITLES = [
    "Neural correlates of the anticipation and experience of thermal pain in healthy volunteers",
    "Working memory load modulates prefrontal and parietal activity across a lifespan sample of 412 adults, with a very long title to test truncation",
    "Reward prediction errors in the ventral striatum",
  ];
  const rows = (count) => Array.from({ length: count }, (_, i) => [
    18000000 + i * 3719, TERMS[i % TERMS.length], Math.max(0.02, +(0.93 - i * 0.0149).toFixed(4)), 3 + ((i * 7) % 41), TITLES[i % TITLES.length],
  ]);
  const study = (i) => ({ title: TITLES[i % 3], authors: "Smith J, Doe A", year: 2008 + (i % 12), journal: "NeuroImage", doi: `10.1016/j.neuroimage.2008.${100 + i}` });
  const table = (title, block, count = 60, total = 1240) => ({
    title, block, columns: ["pmid", "term", "prob", "n_peaks", "title"], kinds: ["int", "str", "float", "int", "str"],
    rows: rows(count), total, truncated: total > count, nifti: null, nifti_reason: "This result is not an image",
    study_links: { columns: [0], records: Object.fromEntries(Array.from({ length: count }, (_, i) => [18000000 + i * 3719, study(i)])) },
  });
  const QUERY = 'ans(pmid, term, prob) :- PROB(pmid, term) :- term_in_study_tfidf(pmid, term, tfidf) & (tfidf > 0.01)';
  const bigPage = (label) => `<!doctype html><html data-theme="dark"><body style="margin:0;font:14px system-ui;background:#0f1a2e;color:#e8eef9;height:420px;display:grid;place-items:center">${label}</body></html>`;
  const reply = (extra = {}) => ({ html: "<pre>Query ran in 0.31 s: 1240 rows.</pre>", viewers: [], failed: false, tables: [table("", 0)], prompt: "", mode: "datalog", predicates: [], ratable: true, graph: "", ...extra });

  const ready = {
    type: "ready", banner: "<pre>NeuroLang 0.0.9 - Neurosynth 0.7\nstudy(pmid)  term_in_study_tfidf(pmid, term, tfidf)  peak_reported(pmid, x, y, z)</pre>", source: "", providers: [{ id: "provider-a", label: "Provider A", model: "model-x", base_url: "https://api.example.org" }],
    examples: { datalog: [QUERY, "ans(s) :- term_in_study_tfidf(s, 'pain', t) & (t > 0.05)", "ans(region, p) :- PROB(region) :- activation(region) // peak_reported(s, x, y, z)"], squall: [] },
    questions: ["Which studies mention pain?", "Which regions are most probably active in studies of working memory?"],
    questionTopics: { "Which terms are mentioned by the studies, and how often?": "Studies and terms", "Which studies mention either language or speech?": "Studies and terms", "How many studies report a peak in the left amygdala?": "Encoding: from a term to the brain", "What is the Bayes factor of fear for the activation of the left amygdala?": "Decoding: from the brain to terms" },
    questionQueries: { "Which studies mention pain?": { datalog: "ans(s) :- term_in_study_tfidf(s, 'pain', t) & (t > 0.05)" }, "How many studies report a peak in the left amygdala?": { datalog: "ans(n) :- n = count(pmid)" } },
    moreQuestions: ["Which terms are mentioned by the studies, and how often?", "Which studies mention either language or speech?", "How many studies report a peak in the left amygdala?", "What is the Bayes factor of fear for the activation of the left amygdala?"],
    predicates: [{ name: "term_in_study_tfidf", signature: "term_in_study_tfidf(pmid, term, tfidf)", kind: "stored", description: "TF-IDF of a term in the abstract of a study.", arguments: [{ sort: "study", type: "int", columns: ["pmid"] }, { sort: "term", type: "str", columns: ["term"] }, { sort: "score", type: "float", columns: ["tfidf"] }] }, { name: "peak_reported", signature: "peak_reported(pmid, x, y, z)", kind: "stored", description: "A peak a study reports.", arguments: [] }],
    commands: [{ name: ".view", usage: ".view", description: "Show the last result in the brain viewer." }, { name: ".help", usage: ".help [topic]", description: "Help." }],
    canRate: true, assistant: true,
  };
  const noKey = { ...ready, assistant: false, canRate: false };

  // The steps of an assistant run, as neurolang_apps.utils.tui.assistant sends them.
  const ASK = { type: "agent", event: "ask", question: "Which studies mention pain?", provider: "provider-a", model: "model-x", strategy: "react", language: "datalog" };
  const STEP1 = { type: "agent", event: "step", thought: "I should look at the relations of the engine first.", action: "list_relations", arguments: {}, observation: "term_in_study_tfidf(pmid, term, tfidf)\npeak_reported(pmid, x, y, z)\nstudy(pmid)" };
  const STEP2 = { type: "agent", event: "step", thought: "Now a query for the studies whose abstract has the term pain.", action: "run_query", arguments: { query: "ans(pmid) :- term_in_study_tfidf(pmid, 'pain', tfidf) & (tfidf > 0.05)" }, observation: "412 rows; first rows:\n18001234\n18004953\n18008672" };
  const PENDING = { type: "agent", event: "reply", text: "Let me check the result with a verification query.", tool_calls: [{ name: "run_query", arguments: { query: "ans(n) :- n = count(pmid)" } }], usage: { input: 900, output: 80 }, seconds: 2.1 };
  const DONE = { type: "agent", event: "done", valid: true, calls: 3, tokens: 4210, seconds: 8.4, query: "ans(pmid) :- term_in_study_tfidf(pmid, 'pain', tfidf) & (tfidf > 0.05)", explanation: "Selects the studies whose abstract has a TF-IDF above 0.05 for the term pain.", back_translation: "The studies for which pain has a TF-IDF above 0.05.", verified: true, verification: "" };
  const NONE = { ...DONE, valid: false, query: "", error: "The engine has no data on gene expression: it holds the studies of Neurosynth and their peaks only." };

  // --- states: what the worker sends at start (`start`), what it answers a
  // statement (`answer`), and what the visitor does once the page is ready (`drive`).
  const click = (selector) => $(selector).click();
  const run = (text) => { $("#input").value = text; $("#input").dispatchEvent(new Event("input")); click("#run"); };
  const ask = (text = "Which studies mention pain?") => { $("#question").value = text; click("#ask-run"); };
  const STATES = {
    boot: { start: null },
    empty: {},
    nokey: { ready: noKey },
    fatal: { start: { type: "error", message: "Failed to fetch bundle.json: NetworkError when attempting to fetch resource." } },
    running: { answer: null, drive: () => run(QUERY) },
    table: { answer: () => [reply()], drive: () => run(QUERY) },
    truncated: { answer: () => [reply({ tables: [table("", 0, 20, 1240)] })], drive: () => run(QUERY) },
    plot_bars: { answer: () => [reply()], drive: async () => { run(QUERY); await wait(300); click(".rt-plot-toggle"); } },
    plot_line: { answer: () => [reply()], drive: async () => { run(QUERY); await wait(300); click(".rt-plot-toggle"); await wait(100); const k = $(".plot-controls select"); k.value = "line"; k.dispatchEvent(new Event("change")); } },
    plot_scatter: { answer: () => [reply()], drive: async () => { run(QUERY); await wait(300); click(".rt-plot-toggle"); await wait(100); const k = $(".plot-controls select"); k.value = "scatter"; k.dispatchEvent(new Event("change")); } },
    filter: { answer: () => [reply()], drive: async () => { run(QUERY); await wait(300); const f = $(".rt-filter"); f.value = "pain"; f.dispatchEvent(new Event("input")); } },
    sorted: { answer: () => [reply()], drive: async () => { run(QUERY); await wait(300); $(".rt-table th:nth-child(5) button").click(); } },
    answers: { answer: () => [reply({ tables: [table("by_term", 0, 40, 40), table("by_region", 1, 25, 25), table("counts", 2, 8, 8)] })], drive: () => run("ans[by_term](...) :- ...") },
    menu_download: { answer: () => [reply()], drive: async () => { run(QUERY); await wait(300); $(".rt-menu > summary").click(); } },
    menu_more: { drive: () => $(".menu > summary").click() },
    focus_header: { answer: () => [reply()], drive: async () => { run(QUERY); await wait(300); $(".rt-table th:nth-child(4) button").focus(); } },
    focus_link: { answer: () => [reply()], drive: async () => { run(QUERY); await wait(300); $(".rt-study a").focus(); } },
    menu_escape: { drive: async () => { $(".menu > summary").click(); $("#docs > summary").click(); await wait(100); $("#docs-links a").focus(); document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })); } },
    menu_outside: { drive: async () => { $(".menu > summary").click(); await wait(100); $("#result-title").click(); } },
    menu_docs: { drive: () => $("#docs > summary").click() },
    brain: { answer: () => [reply({ viewers: [bigPage("brain viewer (mock)")], graph: bigPage("graph (mock)") })], drive: async () => { run(QUERY); await wait(300); click("#tab-brain"); } },
    graph: { answer: () => [reply({ viewers: [bigPage("brain viewer (mock)")], graph: bigPage("graph (mock)") })], drive: async () => { run(QUERY); await wait(300); click("#tab-graph"); } },
    // An image uploaded under a long name: its row is cut, and has the Decode button.
    uploads_long: {
      upload: (name) => ({ name, relation: "image", statement: `.image '${name}' --name image`, kind: "map", threshold: 3.1, cortex_query: "", query: "ans(concept, p) :- decoded(concept, p)" }),
      answer: () => [reply({ uploads: { used: 7234560, most: 209715200, files: [{ name: "autobiographical_memory_association-test_z_FDR_0.01.nii", size: 7234560, loaded: ["autobiographical_memory_association_test_z_fdr_0_01"] }] } })],
      drive: () => {
        const data = new DataTransfer();
        data.items.add(new File([new Uint8Array(8)], "autobiographical_memory_association-test_z_FDR_0.01.nii"));
        $("#upload-file").files = data.files;
        $("#upload-file").dispatchEvent(new Event("change"));
        setTimeout(() => $("#decode-offer-no").click(), 600); // the offer is put aside: the row keeps its Decode button
      },
    },
    // ... and the row's Decode button brings the offer back.
    uploads_decode_again: {
      upload: (name) => ({ name, relation: "image", statement: `.image '${name}' --name image`, kind: "map", threshold: 3.1, cortex_query: "", query: "ans(concept, p) :- decoded(concept, p)" }),
      answer: () => [reply({ uploads: { used: 7234560, most: 209715200, files: [{ name: "autobiographical_memory_association-test_z_FDR_0.01.nii", size: 7234560, loaded: ["autobiographical_memory_association_test_z_fdr_0_01"] }] } })],
      drive: () => {
        const data = new DataTransfer();
        data.items.add(new File([new Uint8Array(8)], "autobiographical_memory_association-test_z_FDR_0.01.nii"));
        $("#upload-file").files = data.files;
        $("#upload-file").dispatchEvent(new Event("change"));
        setTimeout(() => $("#decode-offer-no").click(), 600);
        setTimeout(() => $("#uploaded-list .upload-decode").click(), 900);
      },
    },
    stderr: { answer: () => [reply({ failed: true, tables: [], ratable: false, html: "<pre>Traceback (most recent call last):\n  File \"&lt;repl&gt;\", line 1\nUnsupportedProgramError: the query is not liftable, and the weighted model counting of 412 variables was stopped.</pre>" })], drive: () => run("ans(x) :- PROB(x) :- p(x) & q(x)") },
    query_error: { answer: () => [reply({ failed: true, tables: [], ratable: false, html: "<pre>Syntax error at 1:19: unexpected token ')' after a comparison that has no right-hand side, in a rule whose body is long enough for its error to be wider than the card\n  ans(pmid) :- term_in_study_tfidf(pmid, 'pain', t) & (t > )\n                                                      ^</pre>" })], drive: () => { run("ans(pmid) :- term_in_study_tfidf(pmid, 'pain', t) & (t > )"); } },
    assistant_live: { answer: null, events: [ASK, STEP1, PENDING], drive: () => ask() },
    assistant_done: { answer: () => [reply({ prompt: DONE.query, tables: [table("", 0, 60, 412)] })], events: [ASK, STEP1, STEP2, DONE], drive: () => ask() },
    assistant_none: { answer: () => [reply({ failed: true, tables: [], ratable: false, html: "<pre>No query.</pre>" })], events: [{ ...ASK, question: "Which genes are expressed in the amygdala?" }, STEP1, NONE], drive: () => ask("Which genes are expressed in the amygdala?") },
    rating_no: { answer: () => [reply({ prompt: DONE.query, tables: [table("", 0, 60, 412)] })], events: [ASK, STEP1, STEP2, DONE], drive: async () => { ask(); await wait(900); click("#rate-bad"); } },
    past: { answer: () => [reply({ prompt: DONE.query, tables: [table("", 0, 60, 412)] })], events: [ASK, STEP1, STEP2, DONE], drive: async () => { ask(); await wait(900); $("#asked").open = true; $("#live").open = true; } },
  };
  const spec = STATES[state];
  if (!spec) throw new Error(`unknown state ${state}`);

  // Screenshots are taken as soon as the page settles: no transition or animation is left half way.
  document.head.append(Object.assign(document.createElement("style"), { textContent: "*, *::before, *::after { transition: none !important; animation-duration: .001s !important; }" }));

  // --- the stand-ins ---
  const origFetch = window.fetch.bind(window);
  const json = (value, ok = true) => Promise.resolve({ ok, json: async () => value, text: async () => JSON.stringify(value) });
  window.fetch = (url, options) => {
    const name = String(url).split("/").pop();
    if (name === "bundle.json") return json({ engines: { neurosynth: { label: "Neurosynth", data: "x.zip" }, neurostore: { label: "NeuroStore", data: "y.zip" } }, default: "neurosynth" });
    if (name === "feedback.json") return json({ url: "x", doc: "y" });
    if (name === "highlight.json") return json({}, false);
    if (String(url).startsWith("https://")) return Promise.reject(new TypeError("offline"));
    if (!String(url).startsWith("file:")) return origFetch(url, options);
    // fetch cannot read file: URLs; an XHR can, with --allow-file-access-from-files.
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open("GET", String(url));
      request.onload = () => resolve({ ok: true, text: async () => request.responseText, json: async () => JSON.parse(request.responseText) });
      request.onerror = () => reject(new TypeError("cannot read " + url));
      request.send();
    });
  };

  window.Worker = class {
    constructor() { setTimeout(() => this.#later(0), 0); }
    #send(data) { setTimeout(() => this.onmessage?.({ data }), 0); }
    #later() {}
    terminate() {}
    postMessage(message) {
      if (message.type === "start") {
        if (spec.start === null) return; // never ready
        if (spec.start) return this.#send(spec.start);
        this.#send({ type: "status", message: "Loading Python…" });
        setTimeout(() => { this.#send(spec.ready ?? ready); if (spec.drive) setTimeout(spec.drive, 400); }, 300);
        return;
      }
      if (message.type === "upload") return spec.upload && this.#send({ type: "uploaded", ...spec.upload(message.name) });
      if (message.type) return;
      if (spec.events) spec.events.forEach((event, i) => setTimeout(() => this.#send(event), 100 + i * 150));
      if (spec.answer === null) return; // never answers: the statement runs
      const replies = (spec.answer ?? (() => [reply({ html: "<pre>ok</pre>", tables: [] })]))();
      setTimeout(() => replies.forEach((r) => this.#send({ type: "reply", id: message.id, ...r })), 100 + (spec.events?.length ?? 0) * 150);
    }
  };
})();
