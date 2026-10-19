// The history of what the REPL was given, and the search through it with
// Ctrl+R as in a shell: the most recent entry holding the typed text, then
// the older ones at each Ctrl+R and the newer ones at Ctrl+S.

export class History {
  // *storage* keeps the entries between visits (localStorage), if given.
  constructor(storage = null, key = "neurolang-repl-history", limit = 500) {
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

  // The index of the nearest entry holding *query*, before or after *from*
  // (excluded) according to *direction*, -1 for older and +1 for newer;
  // -1 if there is none.
  find(query, from, direction) {
    const needle = query.toLowerCase();
    for (let i = from + direction; i >= 0 && i < this.entries.length; i += direction) {
      if (this.entries[i].toLowerCase().includes(needle)) return i;
    }
    return -1;
  }
}

// One search through a history, begun on the *draft* the user was typing.
export class ReverseSearch {
  constructor(history, draft = "") {
    this.history = history;
    this.draft = draft;
    this.query = "";
    this.index = history.entries.length; // before the newest entry
    this.found = false;
  }

  // The entry found, else what was being typed.
  get text() {
    return this.found ? this.history.entries[this.index] : this.draft;
  }

  // Search the most recent entry holding *query*.
  type(query) {
    this.query = query;
    this.index = this.history.entries.length;
    this.found = false;
    return this.step(-1);
  }

  older() {
    return this.step(-1);
  }

  newer() {
    return this.step(1);
  }

  // Move to the next match in *direction*; stay where it is if there is none.
  step(direction) {
    const index = this.history.find(this.query, this.index, direction);
    if (index < 0) return false;
    this.index = index;
    this.found = true;
    return true;
  }
}
