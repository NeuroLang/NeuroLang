// The layout of a Datalog query in the code box: a rule wider than the box
// keeps its head on the left and puts each atom of its body on a line of its
// own, after the ":-"; a long probability breaks before its condition. A line
// ending with "," or inside brackets continues the rule, for the parser and
// for the REPL's statements (statements.py), so the layout changes nothing
// of what runs.
import { SETTINGS } from "./settings.js";

const OPENING = "([{", CLOSING = ")]}";

// The places in *text* where *wanted*(index, depth) holds, outside strings,
// *depth* being that of the brackets.
function scan(text, wanted) {
  const found = [];
  let depth = 0, quote = "";
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quote) {
      if (char === quote) quote = "";
    } else if (char === "'" || char === '"') {
      quote = char;
    } else if (OPENING.includes(char)) {
      depth += 1;
    } else if (CLOSING.includes(char)) {
      depth -= 1;
    } else if (wanted(index, depth)) {
      found.push(index);
    }
  }
  return found;
}

// A *part* of a probability starting at *column*, a conjunction in brackets
// ("(a, b)" or "// (a, b)"), broken at its commas when it is wider than *width*,
// its atoms under the first.
function layoutConjunction(part, column, width) {
  const bracket = part.indexOf("(");
  if (column + part.length <= width || bracket < 0 || !["", "//"].includes(part.slice(0, bracket).trim())) return part;
  const commas = scan(part, (index, depth) => depth === 1 && part[index] === ",");
  const atoms = [0, ...commas.map((comma) => comma + 1)].map((from, index) => part.slice(from, commas[index]).trim());
  return atoms.join(`,\n${" ".repeat(column + bracket + 1)}`);
}

// A probability breaks before its "//", under the first character of what it
// conditions, *indent* being the column the body starts at; each side, still
// too wide, at the commas of its conjunction.
function layoutProbability(head, body, indent, width) {
  const [cut] = scan(body, (index, depth) => depth === 1 && body.startsWith("//", index));
  if (cut === undefined) return null;
  const open = body.lastIndexOf("[", cut);
  const column = indent + open + 1;
  const conditioned = layoutConjunction(body.slice(open + 1, cut).trimEnd(), column, width);
  const condition = layoutConjunction(body.slice(cut), column, width);
  return `${head}${body.slice(0, open + 1)}${conditioned}\n${" ".repeat(column)}${condition}`;
}

function layoutLine(line, width) {
  const start = line.trimStart();
  if (line.length <= width || start.startsWith(".") || start.startsWith("%")) return line;
  const [neck] = scan(line, (index, depth) => depth === 0 && line.startsWith(":-", index));
  if (neck === undefined) return line;
  const head = `${line.slice(0, neck).trimEnd()} :- `;
  const body = line.slice(neck + 2).trim();
  // Under the body, unless the head takes much of the line.
  const indent = head.length <= width * SETTINGS.layout.head_share ? head.length : SETTINGS.layout.short_indent;
  const commas = scan(body, (index, depth) => depth === 0 && body[index] === ",");
  if (!commas.length) return layoutProbability(head, body, indent, width) ?? line;
  const atoms = [0, ...commas.map((comma) => comma + 1)].map((from, index) => body.slice(from, commas[index]).trim());
  return head + atoms.join(`,\n${" ".repeat(indent)}`);
}

// *text* laid out for a box *width* characters wide, line by line.
export function layoutQuery(text, width) {
  return text.split("\n").map((line) => layoutLine(line, width)).join("\n");
}
