// The tabs of the results: which one an arrow key moves to, in the order of
// the *tabs* that are shown, as the keyboard model of a tab list has it.

export function tabAfter(tabs, current, key) {
  if (key === "Home") return tabs[0] ?? null;
  if (key === "End") return tabs.at(-1) ?? null;
  const step = { ArrowRight: 1, ArrowLeft: -1 }[key];
  const index = tabs.indexOf(current);
  if (step === undefined || index < 0) return null;
  return tabs[(index + step + tabs.length) % tabs.length];
}
