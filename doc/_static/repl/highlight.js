// The syntax highlighting of Datalog and SQUALL queries, as the terminal
// does it: a port of neurolang_apps/utils/query_highlight.py, whose tests
// are run against this one's (tui/tests/test_repl_page.py), so that the two
// cannot drift. A line is split into fragments of the kinds comment, keyword,
// operator, predicate, string and number ("" for the rest).

// The words of the Datalog grammar, until highlight.json (which
// doc/make_repl_bundle.py writes from query_highlight.py) is loaded.
export const DATALOG_DEFAULTS = {
  datalog_keywords: ["AGGREGATE", "EXISTS", "MARG", "PROB", "SUCC", "ans", "exists", "False", "True", "lambda", "st"],
  datalog_operators: [
    ":-", "←", "-:", "→", "~", "¬", "//", "|", "::", ":~:", "^", ":=", "==", "!=", "<",
    "<=", ">", ">=", "=", "&", "∧", "∃",
  ],
  squall_keywords: [],
};

let words = toSets(DATALOG_DEFAULTS);

function toSets(json) {
  return Object.fromEntries(Object.entries({ ...DATALOG_DEFAULTS, ...json }).map(([key, list]) => [key, new Set(list)]));
}

export function setWords(json) {
  words = toSets(json);
}

// The words of the grammars, from highlight.json next to the page; the
// defaults stay when it is not there.
export async function loadWords(url = "highlight.json") {
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (response.ok) setWords(await response.json());
  } catch {
    // the defaults do
  }
}

const TOKEN = new RegExp(
  String.raw`(?<string>"[^"]*"?|'[^']*'?)` +
    String.raw`|(?<comment>%.*)` +
    String.raw`|(?<number>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)` +
    String.raw`|(?<word>[A-Za-z_][A-Za-z0-9_]*)(?<call>\s*\()?` +
    String.raw`|(?<operator>:~:|:-|::|:=|-:|//|==|<=|>=|!=|\*\*|[^\w\s])`,
  "gu",
);

function wordKind(word, call, squall) {
  if (squall) return words.squall_keywords.has(word) ? "keyword" : "";
  if (words.datalog_keywords.has(word)) return "keyword";
  return call ? "predicate" : "";
}

// The [kind, text] fragments of a *line* of a query.
export function* tokens(line, squall = false) {
  let end = 0;
  for (const match of line.matchAll(TOKEN)) {
    if (match.index > end) yield ["", line.slice(end, match.index)];
    const { string, comment, number, word, call, operator } = match.groups;
    if (word !== undefined) {
      yield [wordKind(word, call !== undefined, squall), word];
      if (call !== undefined) yield ["", call];
    } else if (operator !== undefined) {
      yield [!squall && words.datalog_operators.has(operator) ? "operator" : "", operator];
    } else {
      yield [string !== undefined ? "string" : comment !== undefined ? "comment" : "number", match[0]];
    }
    end = match.index + match[0].length;
  }
  if (end < line.length) yield ["", line.slice(end)];
}

// The nodes showing *text*: its fragments, in a span of the class tk-KIND when
// they have a kind. The lines are kept as they are.
export function highlightNodes(text, squall = false) {
  const nodes = [];
  text.split("\n").forEach((line, index) => {
    if (index) nodes.push(document.createTextNode("\n"));
    for (const [kind, fragment] of tokens(line, squall)) {
      if (kind) {
        const span = document.createElement("span");
        span.className = `tk-${kind}`;
        span.textContent = fragment;
        nodes.push(span);
      } else {
        nodes.push(document.createTextNode(fragment));
      }
    }
  });
  return nodes;
}
