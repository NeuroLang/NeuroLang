// What the log of a query shows when it is folded: its last line, which
// says how long the query took, or why it failed.

export function logSummary(text) {
  return text.split("\n").map((line) => line.trim()).filter(Boolean).at(-1) ?? "";
}
