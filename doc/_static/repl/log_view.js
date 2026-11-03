// What the log of a successful query shows when it is folded: its last line,
// which says how long the query took.

export function logSummary(text) {
  return text.split("\n").map((line) => line.trim()).filter(Boolean).at(-1) ?? "";
}
