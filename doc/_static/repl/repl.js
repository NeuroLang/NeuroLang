// The page of the in-browser REPL: it shows what worker.js answers.
import { History, ReverseSearch } from "./history.js";
import { renderTable } from "./table_view.js";

const statusBox = document.getElementById("status");
const log = document.getElementById("log");
const input = document.getElementById("input");
const run = document.getElementById("run");
const form = document.getElementById("form");
const version = document.getElementById("version");
const versionDetails = document.getElementById("version-details");
const versionSummary = document.getElementById("version-summary");
const search = document.getElementById("search");
const searchInput = document.getElementById("search-input");
const DEFAULT_REPO = "NeuroLang/NeuroLang";

function storage() {
  try {
    return window.localStorage;
  } catch {
    return null; // refused, e.g. in a private window
  }
}

const history = new History(storage());
let browsing = history.entries.length; // the entry Up and Down are on
let draft = ""; // what was typed before browsing
let reverse = null; // the Ctrl+R search under way
let worker = null;
let nextId = 0;

function append(node) {
  log.appendChild(node);
  node.scrollIntoView({ block: "end" });
}

function output(html) {
  const box = document.createElement("div");
  box.className = "entry-output";
  box.innerHTML = html; // rich escapes the text it renders
  return box;
}

function viewer(page) {
  const frame = document.createElement("iframe");
  frame.sandbox = "allow-scripts";
  frame.srcdoc = page;
  frame.title = "NeuroLang brain viewer";
  return frame;
}

// Show *message* in the status; a spinner while the REPL is busy.
function setStatus(message, state = "") {
  statusBox.textContent = message;
  statusBox.className = state;
}

function setReady(ready) {
  input.disabled = !ready;
  run.disabled = !ready;
  if (ready) input.focus();
}

// The version to run, from the page's address: ?repo=OWNER/NAME&ref=BRANCH.
function requestedVersion() {
  const params = new URLSearchParams(location.search);
  return { repo: params.get("repo") || DEFAULT_REPO, ref: params.get("ref") || "" };
}

function start() {
  if (worker) worker.terminate();
  const { repo, ref } = requestedVersion();
  version.elements.repo.value = repo;
  version.elements.ref.value = ref;
  versionSummary.textContent = ref ? `${repo} at ${ref}` : "the build of this page";
  versionDetails.open = Boolean(ref);
  log.replaceChildren();
  setStatus("Starting…", "busy");
  setReady(false);
  worker = new Worker("worker.js", { type: "module" });
  worker.postMessage({ type: "start", repo, ref });
  worker.onmessage = ({ data }) => {
    if (data.type === "status") {
      setStatus(data.message, "busy");
    } else if (data.type === "error") {
      setStatus(`The REPL could not start: ${data.message}`, "error");
    } else if (data.type === "ready") {
      setStatus("Ready.");
      const source = document.createElement("pre");
      source.className = "entry-input";
      source.textContent = `NeuroLang: ${data.source}`;
      append(source);
      append(output(data.banner));
      setReady(true);
    } else if (data.type === "reply") {
      setStatus(data.failed ? "Ready (the last statement failed)." : "Ready.");
      append(output(data.html));
      for (const table of data.tables ?? []) append(renderTable(table));
      for (const page of data.viewers) append(viewer(page));
      setReady(true);
    }
  };
}

function submit() {
  closeSearch(true);
  const text = input.value.trim();
  if (!text) return;
  history.add(text);
  browsing = history.entries.length;
  const echo = document.createElement("pre");
  echo.className = "entry-input";
  echo.textContent = `nl(neurosynth:DL)> ${text}`;
  append(echo);
  input.value = "";
  setReady(false);
  setStatus("Running…", "busy");
  worker.postMessage({ id: nextId++, text });
}

// Up on the first line of the input, and Down on its last, browse the
// history, as Alt+Up and Alt+Down do anywhere in it.
function browse(direction) {
  if (browsing === history.entries.length) draft = input.value;
  const target = browsing + direction;
  if (target < 0 || target > history.entries.length) return;
  browsing = target;
  input.value = history.entries[target] ?? draft;
  input.setSelectionRange(input.value.length, input.value.length);
}

function openSearch() {
  if (input.disabled || reverse) return;
  reverse = new ReverseSearch(history, input.value);
  search.classList.add("open");
  searchInput.value = "";
  searchInput.focus();
  showMatch();
}

// The search shows its match in the input, as a shell shows it at its prompt.
function showMatch() {
  input.value = reverse.text;
  search.classList.toggle("failing", Boolean(reverse.query) && !reverse.found);
}

// Close the search, keeping the match in the input, or the draft if *cancel*.
function closeSearch(cancel = false) {
  if (!reverse) return;
  if (cancel) input.value = reverse.draft;
  reverse = null;
  search.classList.remove("open", "failing");
  if (!input.disabled) input.focus();
}

searchInput.addEventListener("input", () => {
  reverse.type(searchInput.value);
  showMatch();
});
searchInput.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  if (event.ctrlKey && key === "r") {
    event.preventDefault();
    reverse.older();
    showMatch();
  } else if (event.ctrlKey && key === "s") {
    event.preventDefault();
    reverse.newer();
    showMatch();
  } else if (key === "arrowup" || key === "arrowdown") {
    event.preventDefault();
    if (key === "arrowup") reverse.older(); else reverse.newer();
    showMatch();
  } else if (key === "escape" || (event.ctrlKey && key === "g")) {
    event.preventDefault();
    closeSearch(true);
  } else if (key === "enter" || key === "tab") {
    event.preventDefault();
    closeSearch();
  }
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submit();
});
input.addEventListener("keydown", (event) => {
  // Up and Down browse the history from the first and the last line.
  const onFirstLine = !input.value.slice(0, input.selectionStart).includes("\n");
  const onLastLine = !input.value.slice(input.selectionEnd).includes("\n");
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    submit();
  } else if (event.ctrlKey && event.key.toLowerCase() === "r") {
    event.preventDefault(); // not the browser's reload
    openSearch();
  } else if (event.key === "ArrowUp" && (event.altKey || onFirstLine)) {
    event.preventDefault();
    browse(-1);
  } else if (event.key === "ArrowDown" && (event.altKey || (onLastLine && browsing < history.entries.length))) {
    event.preventDefault();
    browse(1);
  }
});
document.getElementById("history").addEventListener("click", openSearch);
const copyCommand = document.getElementById("copy-command");
copyCommand.addEventListener("click", async () => {
  await navigator.clipboard?.writeText(document.getElementById("command").textContent);
  copyCommand.textContent = "Copied";
  setTimeout(() => { copyCommand.textContent = "Copy"; }, 1200);
});
document.getElementById("restart").addEventListener("click", start);
version.addEventListener("submit", (event) => {
  event.preventDefault();
  const params = new URLSearchParams();
  const { repo, ref } = Object.fromEntries(new FormData(version));
  if (ref.trim()) {
    params.set("ref", ref.trim());
    if (repo.trim() !== DEFAULT_REPO) params.set("repo", repo.trim());
  }
  const query = params.toString();
  window.history.replaceState(null, "", query ? `?${query}` : location.pathname);
  start();
});
for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => {
    input.value = button.textContent;
    input.focus();
  });
}
start();
