// A form for the common questions: each has fields, needs some predicates of
// the engine, and gives a Datalog query to complete or run. Nothing is run
// here: the query goes in the code box.

// *text* as a Datalog string, which has no escapes: in the quotes it does not
// have, or an Error if it has both, or a line break.
export function quote(text) {
  if (/[\r\n]/.test(text)) throw new Error("A term cannot have a line break.");
  if (!text.includes('"')) return `"${text}"`;
  if (!text.includes("'")) return `'${text}'`;
  throw new Error("A term cannot have both kinds of quotes.");
}

const TERM = { name: "term", label: "Term", placeholder: "pain", kind: "text" };

export const TEMPLATES = [
  {
    id: "studies-of-term", label: "Studies that mention a term", needs: ["mentions"], fields: [TERM],
    build: ({ term }) => `ans(s) :- mentions(s, ${quote(term)})`,
  },
  {
    id: "studies-of-both", label: "Studies that mention two terms", needs: ["mentions"],
    fields: [{ ...TERM, name: "first", label: "First term" }, { ...TERM, name: "second", label: "Second term", placeholder: "memory" }],
    build: ({ first, second }) => `ans(s) :- mentions(s, ${quote(first)}), mentions(s, ${quote(second)})`,
  },
  {
    id: "studies-of-one-not-other", label: "Studies that mention a term but not another", needs: ["mentions"],
    fields: [{ ...TERM, name: "first", label: "Mentioned term" }, { ...TERM, name: "second", label: "Term left out", placeholder: "memory" }],
    build: ({ first, second }) => `ans(s) :- mentions(s, ${quote(first)}), ~mentions(s, ${quote(second)})`,
  },
  {
    id: "studies-by-weight", label: "Studies that mention a term a lot", needs: ["weighted_mention"],
    fields: [TERM, { name: "weight", label: "Weight above", placeholder: "0.1", kind: "number", value: "0.1" }],
    build: ({ term, weight }) => `ans(s, w) :- weighted_mention(s, ${quote(term)}, w), (w > ${weight})`,
  },
  {
    id: "peaks-of-term", label: "Peaks reported by the studies of a term", needs: ["mentions", "peak_reported"], fields: [TERM],
    build: ({ term }) => `ans(i, j, k, s) :- mentions(s, ${quote(term)}), peak_reported(i, j, k, s)`,
  },
  {
    id: "terms-of-study", label: "Terms that a study mentions", needs: ["mentions"],
    fields: [{ name: "study", label: "Study (PubMed id)", placeholder: "9862924", kind: "integer" }],
    build: ({ study }) => `ans(t) :- mentions(${study}, t)`,
  },
];

// The templates whose predicates are all among the *names* of the engine's.
export function availableTemplates(names) {
  const known = new Set(names);
  return TEMPLATES.filter((template) => template.needs.every((name) => known.has(name)));
}

// The query of *template* for the *values* of its fields (strings), or an Error
// telling which field is wrong.
export function buildQuery(template, values) {
  const cleaned = {};
  for (const field of template.fields) {
    const value = String(values[field.name] ?? "").trim();
    if (!value) throw new Error(`${field.label} is empty.`);
    if (field.kind === "integer" && !/^\d+$/.test(value)) throw new Error(`${field.label} must be a whole number.`);
    if (field.kind === "number" && !/^\d+(\.\d+)?$/.test(value)) throw new Error(`${field.label} must be a number.`);
    cleaned[field.name] = value;
  }
  return template.build(cleaned);
}
