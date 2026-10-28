// The page of the in-browser REPL: a box for the AI assistant, a box for
// the code, and the result as a table or in the brain viewer. It shows what
// worker.js answers.
import { History } from "./history.js";
import { LlmSettings } from "./llm_settings.js";
import { renderTable } from "./table_view.js";

const $ = (id) => document.getElementById(id);
const [statusBox, out, input, question, history_] = ["status", "out", "input", "question", "history"].map($);
const llmForm = $("llm");
const results = $("results");
const DEFAULT_REPO = "NeuroLang/NeuroLang";
const busyControls = [input, question, $("run"), $("ask-run"), $("view")];

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

// A long step shows how long it has lasted, so that a slow one does not look stuck.
const step = { message: "", since: 0 };
const setStatus = (message, state = "") => {
  Object.assign(step, { message, since: Date.now() });
  Object.assign(statusBox, { textContent: message, className: state });
};
setInterval(() => {
  const seconds = Math.round((Date.now() - step.since) / 1000);
  if (statusBox.className === "busy" && seconds >= 5) statusBox.textContent = `${step.message} (${seconds} s)`;
}, 1000);
const html = (markup) => Object.assign(document.createElement("div"), { innerHTML: markup }); // rich escapes its text
const setBusy = (busy) => {
  busyControls.forEach((control) => (control.disabled = busy));
  if (!busy) input.focus();
};

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
      setBusy(false);
    } else if (data.type === "llm") {
      if (!data.ok) Object.assign($("llm-error"), { textContent: data.message, hidden: false });
    } else if (data.type === "reply") {
      setStatus(data.failed ? "Ready (the last statement failed)." : "Ready.");
      out.replaceChildren(html(data.html));
      if (data.prompt) input.value = data.prompt; // what the assistant wrote
      if (data.tables?.length) show("table", ...data.tables.map(renderTable));
      if (data.viewers?.length) {
        show("brain", Object.assign(document.createElement("iframe"), { sandbox: "allow-scripts", srcdoc: data.viewers.at(-1), title: "NeuroLang brain viewer" }));
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
$("key").addEventListener("click", () => ($("llm-details").open = !$("llm-details").open));
$("mode").addEventListener("change", (event) => run(`.mode ${event.target.value}`));
$("view").addEventListener("click", () => run(".view"));
history_.addEventListener("change", () => {
  if (history_.value) input.value = history_.value;
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
