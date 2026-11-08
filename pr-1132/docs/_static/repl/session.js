// The statements a session has run, to save it, open it, or run again what it
// defined after the engine is restarted (a stopped query ends the worker).

const FORMAT = "neurolang-repl-session";
const MAX_ENTRIES = 500;
const MAX_TEXT = 20000;
const MODES = new Set(["datalog", "squall"]);

// Whether *text* defines a rule, whose head is not the answer: the statements
// that leave something in the session. A query only asks.
export function definesRules(text, mode = "datalog") {
  if (mode === "squall") return /\bdefine\s+as\b/i.test(text);
  return text.split("\n").some((line) => {
    const head = /^\s*([A-Za-z_]\w*)\s*\(.*(:-|←)/.exec(line);
    return head !== null && head[1] !== "ans";
  });
}

// The texts that give a session's rules back: ``.clear`` first if asked, the
// statements that define rules each in its language, and the language at the end.
export function replayTexts(entries, finalMode, { clear = false } = {}) {
  const texts = clear ? [".clear"] : [];
  let mode = null; // the language the box was in is not known: the first statement says its own
  for (const entry of entries.filter(({ text, mode: language }) => definesRules(text, language))) {
    if (entry.mode !== mode) texts.push(`.mode ${entry.mode}`);
    mode = entry.mode;
    texts.push(entry.text);
  }
  if (finalMode && finalMode !== mode) texts.push(`.mode ${finalMode}`);
  return texts;
}

export class Session {
  constructor() {
    this.entries = [];
  }

  // Keep *text*, run in *mode*, as the session's.
  add(text, mode) {
    const kept = text.trim();
    if (!kept || kept.startsWith(".") || !MODES.has(mode)) return;
    this.entries.push({ text: kept, mode });
    if (this.entries.length > MAX_ENTRIES) this.entries.shift();
  }

  replace(entries) {
    this.entries = entries.map((entry) => ({ ...entry }));
  }

  // The text of the file that saves the session, with what is in the code box.
  serialise(box, mode) {
    return JSON.stringify({ format: FORMAT, version: 1, mode, box, entries: this.entries }, null, 2);
  }
}

// The entry of a saved file, or an Error saying it is not one.
function parseEntry(entry) {
  if (typeof entry?.text !== "string" || entry.text.length > MAX_TEXT || !MODES.has(entry.mode)) {
    throw new Error("This session has a statement that is not valid.");
  }
  return { text: entry.text, mode: entry.mode };
}

// The JSON of a saved file, or an Error saying it is not a saved session.
function readSession(text) {
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("This file is not a saved session: it is not JSON.");
  }
  if (data?.format !== FORMAT || data.version !== 1 || !Array.isArray(data.entries)) {
    throw new Error("This file is not a saved NeuroLang session.");
  }
  if (data.entries.length > MAX_ENTRIES) throw new Error("This session has too many statements.");
  return data;
}

// The session of a saved file: its entries, the code box and the language, or
// an Error saying what is wrong with the file.
export function parseSession(text) {
  const data = readSession(text);
  const box = typeof data.box === "string" && data.box.length <= MAX_TEXT ? data.box : "";
  return { entries: data.entries.map(parseEntry), box, mode: MODES.has(data.mode) ? data.mode : "datalog" };
}
