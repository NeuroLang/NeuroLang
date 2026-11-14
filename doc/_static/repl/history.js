// The history of what the REPL was given, kept between visits.
import { SETTINGS } from "./settings.js";

export class History {
  // *storage* keeps the entries between visits (localStorage), if given.
  constructor(storage = null, key = "neurolang-repl-history", limit = SETTINGS.history.queries) {
    this.storage = storage;
    this.key = key;
    this.limit = limit;
    this.entries = [];
    try {
      const saved = JSON.parse(storage?.getItem(key) ?? "[]");
      if (Array.isArray(saved)) this.entries = saved.filter((e) => typeof e === "string");
    } catch {
      this.entries = [];
    }
  }

  // Add what was given; an empty entry, and one the last repeats, are not kept.
  add(text) {
    const entry = text.trim();
    if (!entry || this.entries.at(-1) === entry) return;
    this.entries.push(entry);
    if (this.entries.length > this.limit) this.entries.splice(0, this.entries.length - this.limit);
    try {
      this.storage?.setItem(this.key, JSON.stringify(this.entries));
    } catch {
      // A full or refused storage only loses the next visit's history.
    }
  }
}
