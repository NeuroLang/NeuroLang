// The lists of the page that put text in the code box: the examples of the
// engine and its predicates. The page gives the box; these only draw the lists.

function element(tag, properties = {}, ...children) {
  const node = Object.assign(document.createElement(tag), properties);
  node.append(...children);
  return node;
}

// The text a predicate puts in the box: its name, applied (in Datalog).
export const predicateText = (predicate, squall) => (squall ? predicate.name : `${predicate.name}(`);

// Whether *predicate* matches the *filter*, the words of which are all in its name,
// signature, kind or description.
export function matches(predicate, filter) {
  const text = `${predicate.name} ${predicate.signature} ${predicate.kind} ${predicate.description ?? ""}`.toLowerCase();
  return filter.toLowerCase().split(/\s+/).filter(Boolean).every((word) => text.includes(word));
}

// Fill *list* with a button per example, of the *title*; a click gives its text to *pick*.
// The question the box of the assistant suggests: one of *questions* at
// random (random() in [0, 1)), when the box is *current*ly empty and the
// assistant can answer (*ready*); "" otherwise.
export function suggestedQuestion(questions, current, ready, random = Math.random) {
  if (!ready || current.trim() || !questions.length) return "";
  return questions[Math.min(questions.length - 1, Math.floor(random() * questions.length))];
}

// The query a first visit starts with: the first of the *examples* of the
// language *mode*, when the *box* is empty and the browser has no history of
// queries run here (*visited*); "" otherwise.
export function startingQuery(examples, mode, { box, visited }) {
  if (box.trim() || visited) return "";
  return examples[mode]?.[0] ?? "";
}

// *shown*(query) is the text of its button, e.g. laid out (layout.js).
export function renderExamples(list, queries, pick, title = "Put this query in the box", shown = (query) => query) {
  list.replaceChildren(...(queries ?? []).map((query) => {
    const button = element("button", { type: "button", className: "guide-item", textContent: shown(query), title });
    button.addEventListener("click", () => pick(query));
    return element("li", {}, button);
  }));
}

const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;
const KIND_TEXT = {
  stored: "a table of the engine's data",
  derived: "defined by rules of the engine",
  session: "defined in this session",
};

// The columns of a predicate, one by one, as its declaration gives them: the
// *name* of each, the *sort* of what it holds and its *type*.
export function columnDocs(predicate) {
  return (predicate.arguments ?? []).flatMap((argument) =>
    argument.columns.map((name) => ({ name, sort: argument.sort, type: argument.type })));
}

// A query for every row of *predicate*, in Datalog, or null if its columns, or its name, are not plain.
export function showAllQuery(predicate) {
  const columns = columnDocs(predicate).map((column) => column.name);
  if (!columns.length || !IDENTIFIER.test(predicate.name) || !columns.every((name) => IDENTIFIER.test(name))) return null;
  const variables = columns.join(", ");
  return `ans(${variables}) :- ${predicate.name}(${variables})`;
}

// Whether the *command* matches the *filter*, the words of which are all in its usage or its description.
export function matchesCommand(command, filter) {
  const text = `${command.usage} ${command.description}`.toLowerCase();
  return filter.toLowerCase().split(/\s+/).filter(Boolean).every((word) => text.includes(word));
}

// The documentation panels that are open, by key, for the lists to show them open again when they are drawn again.
const opened = new Set();

// A button that shows or hides the *panel* after it, and says so to a screen reader; the panel of
// *key* stays open when the list is drawn again.
function disclosure(label, panel, key) {
  panel.hidden = !opened.has(key);
  const button = element("button", { type: "button", className: "guide-info", textContent: "ⓘ", title: label });
  button.setAttribute("aria-label", label);
  button.setAttribute("aria-expanded", String(!panel.hidden));
  button.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    opened[panel.hidden ? "delete" : "add"](key);
    button.setAttribute("aria-expanded", String(!panel.hidden));
  });
  return button;
}

// What is known of *predicate*: what it holds, its columns, where it comes from, and ways to use it.
function predicateDoc(predicate, squall, { put, replace }) {
  const columns = columnDocs(predicate);
  const all = squall ? null : showAllQuery(predicate);
  const actions = [
    element("button", { type: "button", className: "ghost", textContent: "Put in the box", onclick: () => put(predicateText(predicate, squall)) }),
    ...(all ? [element("button", { type: "button", className: "ghost", textContent: "Show all its rows", onclick: () => replace(all) })] : []),
  ];
  return element("div", { className: "guide-doc" },
    element("p", { textContent: predicate.description || "No description yet." }),
    ...(columns.length ? [element("ul", {}, ...columns.map((column) => element("li", {},
      element("code", { textContent: column.name }), ` · ${column.sort}, ${column.type}`)))] : []),
    element("p", { className: "note", textContent: `${predicate.signature} — ${KIND_TEXT[predicate.kind] ?? predicate.kind}` }),
    element("div", { className: "row" }, ...actions));
}

// Fill *list* with the predicates that match *filter*, each with its documentation under a button; a
// click on the predicate gives its text to *pick*, and the buttons of its documentation use *replace* to
// put a whole query in the box.
export function renderPredicates(list, predicates, filter, pick, squall = false, limit = 200, replace = pick) {
  const shown = predicates.filter((predicate) => matches(predicate, filter));
  list.replaceChildren(...shown.slice(0, limit).map((predicate) => {
    const button = element("button", {
      type: "button", className: "guide-item", title: predicate.description || "Put this predicate in the box",
    }, element("code", { textContent: predicate.signature }), element("span", { className: "guide-kind", textContent: predicate.kind }));
    button.addEventListener("click", () => pick(predicateText(predicate, squall)));
    const doc = predicateDoc(predicate, squall, { put: pick, replace });
    return element("li", {}, button, disclosure(`Documentation of ${predicate.name}`, doc, `relation:${predicate.name}`), doc);
  }));
  if (shown.length > limit) list.append(element("li", { className: "note", textContent: `…and ${shown.length - limit} more: filter them.` }));
  return shown.length;
}

// Fill *list* with the commands that match *filter*, each with what it does under a button; a click on
// one gives its name to *pick*, and *help*(topic) runs ``.help topic``.
export function renderCommands(list, commands, filter, pick, help) {
  const shown = commands.filter((command) => matchesCommand(command, filter));
  list.replaceChildren(...shown.map((command) => {
    const button = element("button", { type: "button", className: "guide-item", title: command.description },
      element("code", { textContent: command.usage }));
    button.addEventListener("click", () => pick(command.name));
    const doc = element("div", { className: "guide-doc" },
      element("p", { textContent: command.description }),
      ...(command.details && command.details !== command.description ? [element("p", { className: "note", textContent: command.details })] : []),
      ...(command.topic ? [element("button", { type: "button", className: "ghost", textContent: `Full help: .help ${command.topic}`, onclick: () => help(command.topic) })] : []));
    return element("li", {}, button, disclosure(`Documentation of ${command.name}`, doc, `command:${command.name}`), doc);
  }));
  return shown.length;
}
