// The questions asked in English to the assistant, each with the query it
// gave (or none), kept between visits: the history of the box of the
// assistant, beside that of the code box (history.js).
import { SETTINGS } from "./settings.js";

const KEY = "neurolang-repl-asked";

export class AskedHistory {
  // *storage* keeps the entries between visits (localStorage), if given;
  // *now* gives the time of an entry, in ms.
  constructor(storage = null, now = Date.now, limit = SETTINGS.history.questions) {
    this.storage = storage;
    this.now = now;
    this.limit = limit;
    this.entries = [];
    try {
      const saved = JSON.parse(storage?.getItem(KEY) ?? "[]");
      if (Array.isArray(saved)) this.entries = saved.filter((entry) => typeof entry?.question === "string");
    } catch {
      this.entries = [];
    }
  }

  // Keep the *question* with its *query* (its *language*, whether it is
  // *valid*, the assistant's *explanation*); not one without a question,
  // nor one the last repeats.
  add({ question, query = "", language = "datalog", valid = false, explanation = "" }) {
    question = question.trim();
    if (!question) return;
    const last = this.entries.at(-1);
    if (last?.question === question && last?.query === query) return;
    this.entries.push({ question, query, language, valid: Boolean(valid), explanation: explanation ?? "", at: this.now() });
    if (this.entries.length > this.limit) this.entries.splice(0, this.entries.length - this.limit);
    try {
      this.storage?.setItem(KEY, JSON.stringify(this.entries));
    } catch {
      // A full or refused storage only loses the next visit's history.
    }
  }

  // The entries, the newest first.
  newest() {
    return [...this.entries].reverse();
  }
}
