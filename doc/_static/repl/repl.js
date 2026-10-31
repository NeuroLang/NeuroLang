// The page of the in-browser REPL: a box for the AI assistant, a box for
// the code, and the result as a table or in the brain viewer. It shows what
// worker.js answers.
import { History } from "./history.js";
import { LlmSettings } from "./llm_settings.js";
import { imageButtons, renderTable } from "./table_view.js";
import { AgentLog, renderAgentLog, tickWaiting } from "./agent_view.js";
import { DOCS_URL, PAGES, topicAt } from "./docs.js";
import { Editor } from "./editor.js";
import { renderExamples, renderPredicates } from "./guide.js";
import { highlightNodes, loadWords } from "./highlight.js";
import { decodeShare, encodeShare } from "./share.js";

const $ = (id) => document.getElementById(id);
const [statusBox, out, input, question, history_] = ["status", "out", "input", "question", "history"].map($);
const llmForm = $("llm");
const results = $("results");
const DEFAULT_REPO = "NeuroLang/NeuroLang";
const busyControls = [input, question, $("run"), $("ask-run"), $("view")];
// The code box colours its text as the language of the mode select.
const editor = new Editor(input, $("mirror"), () => $("mode").value === "squall");
loadWords().then(() => editor.refresh()); // the words of the grammars, from highlight.json

const store = (kind) => {
  try {
    return window[kind];
  } catch {
    return null; // refused, e.g. in a private window
  }
};
const history = new History(store("localStorage"));
const llmSettings = new LlmSettings(store("localStorage"), store("sessionStorage"));
let worker = null;
let providers = []; // those of the AI assistant, from the worker

// The work of the AI assistant, shown as it comes from the worker.
const agentLog = new AgentLog();
const live = $("live");
const liveLog = $("live-log");
// What the query the assistant wrote does, above the box it is in: the model's
// explanation, and the query read back in English, with the check's doubts.
const summary = $("summary");
function hideSummary() {
  summary.hidden = true;
}
function showSummary({ valid, explanation, back_translation: reads, verified, verification }) {
  if (!valid || !(explanation || reads)) return hideSummary();
  $("summary-text").textContent = explanation ?? "";
  $("summary-reads").textContent = reads ? `Reads back as: ${reads}` : "";
  $("summary-warning").textContent = verified === false ? `The check found that the query may not answer the question: ${verification}` : "";
  summary.hidden = false;
}

function showAgent() {
  live.hidden = false;
  live.classList.toggle("running", agentLog.running);
  $("live-summary").textContent = agentLog.count ? `Assistant steps (${agentLog.count})` : "Assistant steps";
  const squall = agentLog.header?.language === "squall";
  const renderQuery = (text) => {
    const fragment = document.createDocumentFragment();
    fragment.append(...highlightNodes(text, squall));
    return fragment;
  };
  renderAgentLog(agentLog, liveLog, { renderQuery });
  if (agentLog.running) liveLog.scrollTop = liveLog.scrollHeight;
}

// A long step shows how long it has lasted, so that a slow one does not look stuck.
const step = { message: "", since: 0 };
const setStatus = (message, state = "") => {
  Object.assign(step, { message, since: Date.now() });
  Object.assign(statusBox, { textContent: message, className: state });
};
setInterval(() => {
  const seconds = Math.round((Date.now() - step.since) / 1000);
  if (statusBox.className === "busy" && seconds >= 5) statusBox.textContent = `${step.message} (${seconds} s)`;
  tickWaiting(liveLog);
}, 1000);
const html = (markup) => Object.assign(document.createElement("div"), { innerHTML: markup }); // rich escapes its text
const setBusy = (busy) => {
  document.body.toggleAttribute("data-busy", busy);
  busyControls.forEach((control) => (control.disabled = busy));
  if (!busy) input.focus();
};

// The tables of the last result, for the downloads of the brain tab, which
// shows a result that .view may have taken from an earlier reply.
let lastTables = [];

// The file of a result: the worker makes it, the page offers it to be saved.
function requestDownload(block, kind, value = "") {
  setBusy(true);
  setStatus("Preparing the file…", "busy");
  worker.postMessage({ type: "download", block, kind, value });
}

function saveFile({ name, mime, bytes }) {
  const link = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([bytes], { type: mime })), download: name });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

// The documentation: its pages in the menu, and what the keyword under the cursor is.
const link = (href, text) => Object.assign(document.createElement("a"), { href, target: "_blank", rel: "noopener", textContent: text });
$("docs-links").replaceChildren(...PAGES.map(([label, page]) => link(DOCS_URL + page, label)));

function showHint() {
  const topic = topicAt(input.value, input.selectionStart, $("mode").value === "squall");
  $("hint").replaceChildren(
    ...(topic ? [Object.assign(document.createElement("code"), { textContent: topic.word }), ` ${topic.text} · `, link(topic.url, "documentation ↗")] : []),
  );
}
["keyup", "click", "focus"].forEach((type) => input.addEventListener(type, showHint));

// The examples of the engine and its predicates, which put their text in the box.
let guide = { examples: {}, predicates: [] };
function showGuide() {
  if (!$("guide").open) return;
  const squall = $("mode").value === "squall";
  renderExamples($("guide-examples"), guide.examples[squall ? "squall" : "datalog"], (query) => {
    editor.value = query;
    hideSummary();
    input.focus();
  });
  renderPredicates($("guide-relations"), guide.predicates, $("guide-filter").value, (text) => editor.insert(text), squall);
}
$("guide").addEventListener("toggle", showGuide);
$("guide-filter").addEventListener("input", showGuide);

// A query shared as a link is put in the box, not run.
function restoreShared() {
  const shared = decodeShare(location.hash);
  if (!shared) return;
  if ($("mode").value !== shared.mode) {
    $("mode").value = shared.mode;
    run(`.mode ${shared.mode}`);
  }
  editor.value = shared.code;
}

function showHistory() {
  history_.replaceChildren(new Option("History", ""), ...[...history.entries].reverse().map((e) => new Option(e.replace(/\s+/g, " ").slice(0, 90), e)));
}

// Show *what* in the results, in the tab *tab*.
function show(tab, ...nodes) {
  $(tab).replaceChildren(...nodes);
  results.classList.add("has");
  results.dataset.tab = tab;
}

// The AI assistant: the form, and the key given to the worker.
const choices = () => {
  const data = Object.fromEntries(new FormData(llmForm));
  return { provider: data.provider, model: data.model.trim(), baseUrl: (data.baseUrl ?? "").trim(), key: data.key.trim(), remember: llmForm.elements.remember.checked };
};

function showProvider() {
  const preset = providers.find((p) => p.id === llmForm.elements.provider.value);
  llmForm.elements.model.placeholder = preset?.model || "model";
  llmForm.elements.baseUrl.hidden = !preset || Boolean(preset.base_url);
  llmForm.elements.baseUrl.required = !llmForm.elements.baseUrl.hidden;
}

function useKey() {
  $("llm-error").hidden = true;
  const chosen = choices();
  worker.postMessage({ type: "llm", ...chosen });
  Object.assign($("ask-run"), { title: "" }).classList.remove("ghost");
  $("llm-summary").textContent = `${providers.find((p) => p.id === chosen.provider)?.label ?? chosen.provider}, key in use`;
}

function setProviders(list) {
  providers = list;
  const saved = llmSettings.load();
  const { elements } = llmForm;
  elements.provider.replaceChildren(...list.map((p) => new Option(p.label, p.id)));
  if (list.some((p) => p.id === saved.provider)) elements.provider.value = saved.provider;
  Object.assign(elements.model, { value: saved.model });
  Object.assign(elements.baseUrl, { value: saved.baseUrl });
  Object.assign(elements.key, { value: saved.key });
  elements.remember.checked = saved.remember;
  elements.provider.disabled = !list.length;
  llmForm.querySelector("button").disabled = !list.length;
  showProvider();
  $("llm-summary").textContent = list.length ? "bring your own API key" : "not in this version of NeuroLang";
  if (list.length && saved.key) useKey();
}

// The version to run, from the page's address: ?repo=OWNER/NAME&ref=BRANCH.
function start() {
  worker?.terminate();
  const params = new URLSearchParams(location.search);
  const [repo, ref] = [params.get("repo") || DEFAULT_REPO, params.get("ref") || ""];
  Object.assign(version.elements.repo, { value: repo });
  Object.assign(version.elements.ref, { value: ref });
  $("version-summary").textContent = ref ? `${repo} at ${ref}` : "the build of this page";
  out.replaceChildren();
  results.classList.remove("has");
  agentLog.reset();
  live.hidden = true;
  setStatus("Starting…", "busy");
  setBusy(true);
  worker = new Worker("worker.js", { type: "module" });
  worker.postMessage({ type: "start", repo, ref, full: params.get("engine") === "full" });
  worker.onerror = (event) =>
    setStatus(`The REPL stopped: ${event.message || "the browser ended its worker, perhaps out of memory"}. Restart it.`, "error");
  worker.onmessage = ({ data }) => {
    if (data.type === "status") {
      setStatus(data.message, "busy");
    } else if (data.type === "error") {
      setStatus(`The REPL could not start: ${data.message}`, "error");
    } else if (data.type === "ready") {
      setStatus("Ready.");
      $("banner").replaceChildren(html(data.banner));
      setProviders(data.providers ?? []);
      guide = { examples: data.examples ?? {}, predicates: data.predicates ?? [] };
      if (!input.value && !decodeShare(location.hash)) $("guide").open = true; // examples to start from
      showGuide();
      setBusy(false);
      restoreShared();
    } else if (data.type === "file") {
      setBusy(false);
      if (data.error) {
        setStatus(`The file could not be made: ${data.error}`, "error");
      } else {
        saveFile(data);
        setStatus("Ready.");
      }
    } else if (data.type === "agent") {
      agentLog.apply(data);
      showAgent();
      if (data.event === "ask") hideSummary();
      if (data.event === "done") showSummary(data);
    } else if (data.type === "llm") {
      if (!data.ok) Object.assign($("llm-error"), { textContent: data.message, hidden: false });
    } else if (data.type === "reply") {
      setStatus(data.failed ? "Ready (the last statement failed)." : "Ready.");
      out.replaceChildren(html(data.html));
      if (data.predicates) guide.predicates = data.predicates;
      if (data.mode) $("mode").value = data.mode; // the assistant may have switched it
      showGuide();
      if (data.prompt) editor.value = data.prompt; // what the assistant wrote
      else editor.refresh();
      if (data.tables?.length) {
        lastTables = data.tables;
        show("table", ...data.tables.map((table) => renderTable(table, { download: requestDownload })));
      }
      if (data.viewers?.length) {
        const viewer = Object.assign(document.createElement("iframe"), { sandbox: "allow-scripts", srcdoc: data.viewers.at(-1), title: "NeuroLang brain viewer" });
        show("brain", ...[imageButtons(lastTables, requestDownload), viewer].filter(Boolean));
      }
      setBusy(false);
    }
  };
}

function run(text, { remember = false } = {}) {
  text = text.trim();
  if (!text) return;
  if (remember) {
    history.add(text);
    showHistory();
  }
  setBusy(true);
  setStatus("Running…", "busy");
  if (!text.startsWith(".ask") && !text.startsWith(".llm")) hideSummary();
  worker.postMessage({ text });
}

// The theme: light, dark or the computer's, which the page's head applied.
const THEME = "neurolang-repl-theme";
const system = matchMedia("(prefers-color-scheme: light)");
const themeButtons = [...document.querySelectorAll("#theme button")];
function applyTheme(choice) {
  const light = choice === "light" || (choice === "system" && system.matches);
  document.documentElement.dataset.theme = light ? "light" : "dark";
  document.documentElement.dataset.themeChoice = choice;
  themeButtons.forEach((button) => button.setAttribute("aria-pressed", button.dataset.theme === choice));
}
const chosen = () => store("localStorage")?.getItem(THEME) || "system";
for (const button of themeButtons) {
  button.addEventListener("click", () => {
    store("localStorage")?.setItem(THEME, button.dataset.theme);
    applyTheme(button.dataset.theme);
  });
}
system.addEventListener("change", () => applyTheme(chosen()));
applyTheme(chosen());

const version = $("version");
$("code").addEventListener("submit", (event) => {
  event.preventDefault();
  run(input.value, { remember: true });
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    run(input.value, { remember: true });
  }
});
$("ask").addEventListener("submit", (event) => {
  event.preventDefault();
  if (question.value.trim()) run(`.ask ${question.value}`);
});
$("share").addEventListener("click", async (event) => {
  const url = `${location.origin}${location.pathname}${location.search}#${encodeShare(editor.value, $("mode").value)}`;
  window.history.replaceState(null, "", url);
  try {
    await navigator.clipboard.writeText(url);
    event.target.textContent = "Copied";
    setTimeout(() => { event.target.textContent = "Copy link"; }, 1200);
  } catch {
    window.prompt("Copy this link", url); // no clipboard here, e.g. an unsecured address
  }
});
$("key").addEventListener("click", () => ($("llm-details").open = !$("llm-details").open));
$("mode").addEventListener("change", (event) => {
  editor.refresh();
  showGuide();
  run(`.mode ${event.target.value}`);
});
$("view").addEventListener("click", () => run(".view"));
history_.addEventListener("change", () => {
  if (history_.value) {
    editor.value = history_.value;
    hideSummary();
  }
  history_.value = "";
  input.focus();
});
for (const tab of ["table", "brain"]) $(`tab-${tab}`).addEventListener("click", () => (results.dataset.tab = tab));
llmForm.elements.provider.addEventListener("change", showProvider);
llmForm.addEventListener("submit", (event) => {
  event.preventDefault();
  llmSettings.save(choices());
  useKey();
  $("llm-details").open = false;
});
$("llm-forget").addEventListener("click", () => {
  llmSettings.forget();
  Object.assign(llmForm.elements.key, { value: "" });
  llmForm.elements.remember.checked = false;
  worker.postMessage({ type: "llm", key: "" });
  Object.assign($("ask-run"), { title: "Add an API key to use the assistant" }).classList.add("ghost");
  $("llm-error").hidden = true;
  $("llm-summary").textContent = "bring your own API key";
});
$("restart").addEventListener("click", start);
version.addEventListener("submit", (event) => {
  event.preventDefault();
  const params = new URLSearchParams();
  const { repo, ref } = Object.fromEntries(new FormData(version));
  if (ref.trim()) {
    params.set("ref", ref.trim());
    if (repo.trim() !== DEFAULT_REPO) params.set("repo", repo.trim());
  }
  window.history.replaceState(null, "", params.size ? `?${params}` : location.pathname);
  start();
});
showHistory();
start();
