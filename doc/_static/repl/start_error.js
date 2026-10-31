// What the page tells when the engine does not start: in plain words, what
// to do, with the technical message kept apart.

export function explainStartError(message) {
  const text = String(message ?? "");
  if (/<!doctype|not valid json|unexpected token '</i.test(text)) {
    return { summary: "The files of the engine were not found on the server: this is a problem of the site, not of your query.", detail: text };
  }
  if (/failed to fetch|networkerror|load failed|network/i.test(text)) {
    return { summary: "The engine could not be downloaded. Check your connection and try again.", detail: text };
  }
  if (/memory/i.test(text)) {
    return { summary: "The browser ran out of memory while loading the engine. Close other tabs and try again.", detail: text };
  }
  return { summary: "The engine could not start. Try again.", detail: text };
}
