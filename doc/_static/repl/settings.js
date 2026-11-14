// The settings of the page, read from settings.yaml: the one place its
// limits and sizes are written. The page fetches the file; the tests, which
// run in Node, read it from the disk. A module reads SETTINGS.section.name.

// A scalar of the YAML the file is written in: a number, true, false, null,
// a list [a, b], a quoted or a plain string.
function scalar(text) {
  text = text.trim();
  if (text.startsWith("[") && text.endsWith("]")) {
    const inner = text.slice(1, -1).trim();
    return inner ? inner.split(",").map(scalar) : [];
  }
  if (/^(["']).*\1$/.test(text)) return text.slice(1, -1);
  if (text === "true" || text === "false") return text === "true";
  if (text === "null" || text === "~" || text === "") return null;
  return /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(text) ? Number(text) : text;
}

// The line *line* without its comment: from a # outside quotes.
function uncommented(line) {
  let quote = "";
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (quote) {
      if (char === quote) quote = "";
    } else if (char === '"' || char === "'") {
      quote = char;
    } else if (char === "#" && (index === 0 || /\s/.test(line[index - 1]))) {
      return line.slice(0, index);
    }
  }
  return line;
}

// The mapping of the YAML *text*: mappings nested by their indentation, of
// the scalars above.
export function parseSettings(text) {
  const root = {};
  const stack = [{ indent: -1, map: root }];
  for (const raw of text.split("\n")) {
    const line = uncommented(raw).replace(/\s+$/, "");
    if (!line.trim()) continue;
    const indent = line.length - line.trimStart().length;
    const [, key, value] = /^\s*([^:]+?):\s*(.*)$/.exec(line) ?? [];
    if (key === undefined) throw new Error(`settings.yaml: not "key: value": ${raw}`);
    while (indent <= stack.at(-1).indent) stack.pop();
    const parent = stack.at(-1).map;
    if (value === "") {
      parent[key] = {};
      stack.push({ indent, map: parent[key] });
    } else {
      parent[key] = scalar(value);
    }
  }
  return root;
}

const file = new URL("./settings.yaml", import.meta.url);
const text = typeof window === "undefined" && typeof process !== "undefined"
  ? (await import("node:fs")).readFileSync(file, "utf8")
  : await (await fetch(file, { cache: "no-cache" })).text();

export const SETTINGS = parseSettings(text);
