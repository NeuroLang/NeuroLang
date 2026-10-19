// The model of the table viewer of the REPL's results: the rows of a
// result (see neurolang.utils.tui.web.table_payload) filtered, sorted and
// paged, without the page. table_view.js shows it.

const collator = new Intl.Collator(undefined, { numeric: true });

// How the cells of a column of each kind are ordered; missing ones last.
function compareValues(kind, a, b) {
  if (kind === "int" || kind === "float") return Number(a) - Number(b);
  if (kind === "bool") return Number(a) - Number(b);
  return collator.compare(String(a), String(b));
}

export function cellText(value) {
  if (value === null || value === undefined) return "";
  return String(value);
}

function csvField(text) {
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export class TableModel {
  constructor(payload) {
    this.title = payload.title ?? "";
    this.columns = payload.columns;
    this.kinds = payload.kinds;
    this.rows = payload.rows;
    this.total = payload.total;
    this.truncated = payload.truncated;
    this.filterText = "";
    this.sort = null; // { column: index, direction: "asc" | "desc" }
    this.order = this.rows.map((_, index) => index);
    this.texts = null; // the lower-case text of each row's cells, when needed
  }

  // The number of rows the filter keeps.
  get visibleCount() {
    return this.order.length;
  }

  // Keep the rows holding every word of *text*, in any cell, or, for a word
  // ``column:text``, in the cell of the column starting with ``column``.
  setFilter(text) {
    this.filterText = text;
    this.refresh();
  }

  // Cycle the sort on *column*: ascending, descending, none.
  toggleSort(column) {
    if (this.sort === null || this.sort.column !== column) {
      this.sort = { column, direction: "asc" };
    } else if (this.sort.direction === "asc") {
      this.sort = { column, direction: "desc" };
    } else {
      this.sort = null;
    }
    this.refresh();
  }

  refresh() {
    let order = this.rows.map((_, index) => index);
    for (const word of this.filterText.toLowerCase().split(/\s+/).filter(Boolean)) {
      const match = this.matcher(word);
      order = order.filter(match);
    }
    if (this.sort !== null) {
      const { column, direction } = this.sort;
      const kind = this.kinds[column];
      const sign = direction === "asc" ? 1 : -1;
      const cell = (index) => this.rows[index][column];
      order.sort((a, b) => {
        const x = cell(a), y = cell(b);
        if (x === null || x === undefined) return (y === null || y === undefined) ? a - b : 1;
        if (y === null || y === undefined) return -1;
        return sign * compareValues(kind, x, y) || a - b;
      });
    }
    this.order = order;
  }

  // The function telling whether a row, by its index, matches *word*.
  matcher(word) {
    const split = word.indexOf(":");
    if (split > 0) {
      const name = word.slice(0, split);
      const column = this.columns.findIndex((c) => c.toLowerCase().startsWith(name));
      if (column >= 0) {
        const text = word.slice(split + 1);
        return (index) => cellText(this.rows[index][column]).toLowerCase().includes(text);
      }
    }
    if (this.texts === null) {
      this.texts = this.rows.map((row) => row.map(cellText).join("\t").toLowerCase());
    }
    return (index) => this.texts[index].includes(word);
  }

  pageCount(size) {
    return Math.max(1, Math.ceil(this.visibleCount / size));
  }

  // The rows of page *index* (from 0) of *size* rows: each with its number
  // in the result, from 1, and its cells.
  page(index, size) {
    const last = this.pageCount(size) - 1;
    const clamped = Math.min(Math.max(index, 0), last);
    const start = clamped * size;
    const rows = this.order.slice(start, start + size).map((i) => ({ number: i + 1, cells: this.rows[i] }));
    return { index: clamped, start, end: start + rows.length, rows };
  }

  // The smallest and the largest of a numeric column, or null.
  range(column) {
    if (this.kinds[column] !== "int" && this.kinds[column] !== "float") return null;
    let low = Infinity, high = -Infinity;
    for (const row of this.rows) {
      const value = Number(row[column]);
      if (row[column] === null || Number.isNaN(value)) continue;
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
    return low <= high ? { low, high } : null;
  }

  // The visible rows, header first, as text separated by *separator*.
  toDelimited(separator, quote = (text) => text) {
    const lines = [this.columns.map(quote).join(separator)];
    for (const index of this.order) {
      lines.push(this.rows[index].map((v) => quote(cellText(v))).join(separator));
    }
    return lines.join("\n") + "\n";
  }

  toTSV() {
    return this.toDelimited("\t", (text) => text.replace(/[\t\n\r]+/g, " "));
  }

  toCSV() {
    return this.toDelimited(",", csvField);
  }
}
