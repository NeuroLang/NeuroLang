// The page of the in-browser REPL: it shows what worker.js answers.
const statusBox = document.getElementById("status");
const log = document.getElementById("log");
const input = document.getElementById("input");
const run = document.getElementById("run");
const form = document.getElementById("form");
const inputs = [];
let historyIndex = 0;
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

function setReady(ready) {
  input.disabled = !ready;
  run.disabled = !ready;
  if (ready) input.focus();
}

function start() {
  if (worker) worker.terminate();
  log.replaceChildren();
  statusBox.className = "";
  setReady(false);
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "status") {
      statusBox.textContent = data.message;
    } else if (data.type === "error") {
      statusBox.className = "error";
      statusBox.textContent = `The REPL could not start: ${data.message}`;
    } else if (data.type === "ready") {
      statusBox.textContent = "Ready.";
      append(output(data.banner));
      setReady(true);
    } else if (data.type === "reply") {
      statusBox.textContent = data.failed ? "Ready (the last statement failed)." : "Ready.";
      append(output(data.html));
      for (const page of data.viewers) append(viewer(page));
      setReady(true);
    }
  };
}

function submit() {
  const text = input.value.trim();
  if (!text) return;
  inputs.push(text);
  historyIndex = inputs.length;
  const echo = document.createElement("pre");
  echo.className = "entry-input";
  echo.textContent = `nl(neurosynth:DL)> ${text}`;
  append(echo);
  input.value = "";
  setReady(false);
  statusBox.textContent = "Running…";
  worker.postMessage({ id: nextId++, text });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submit();
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    submit();
  } else if (event.altKey && event.key === "ArrowUp" && historyIndex > 0) {
    input.value = inputs[--historyIndex];
  } else if (event.altKey && event.key === "ArrowDown" && historyIndex < inputs.length) {
    input.value = inputs[++historyIndex] ?? "";
  }
});
document.getElementById("restart").addEventListener("click", start);
for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => {
    input.value = button.textContent;
    input.focus();
  });
}
start();
