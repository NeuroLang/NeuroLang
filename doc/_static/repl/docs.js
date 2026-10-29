// Where the syntax of the languages is documented: the pages of the
// documentation, and what to tell about the keyword under the cursor.
import { tokens } from "./highlight.js";

export const DOCS_URL = "https://neurolang.github.io/";

export const PAGES = [
  ["Datalog tutorial", "tutorial_datalog.html"],
  ["SQUALL tutorial", "tutorial_squall.html"],
  ["The query assistant", "query_assistant.html"],
  ["Concepts", "concepts.html"],
  ["All the documentation", ""],
];

const DATALOG = "tutorial_datalog.html";
// The keywords and operators of Datalog, with the section of the tutorial that
// explains them (a test checks the headings exist).
export const TOPICS = {
  PROB: ["PROB[query] = p is the probability of a query over a probabilistic choice", `${DATALOG}#prob-marginal-probability`],
  MARG: ["MARG[query] = p is the marginal probability of a conjunction", `${DATALOG}#marg-marginal-probability-of-a-conjunction`],
  SUCC: ["SUCC is the success probability of a query", `${DATALOG}#succ-success-probability`],
  AGGREGATE: ["AGGREGATE[group](body @ f(x)) = r summarises x with count, sum, mean, percentile…", `${DATALOG}#aggregation`],
  exists: ["exists(x; p(x)) is an existential predicate", `${DATALOG}#existential-predicates`],
  EXISTS: ["EXISTS is an existential predicate", `${DATALOG}#existential-predicates`],
  ans: ["ans(…) is the query: its rows are the result", `${DATALOG}#the-query-predicate-ans`],
  ":-": ["head :- body: the head holds when the body does", `${DATALOG}#rules-and-the-operator`],
  "~": ["~ is negation", `${DATALOG}#negation`],
  "//": ["a // b conditions the probability of a on b", `${DATALOG}#prob-marginal-probability`],
};
const SQUALL_TOPIC = ["A keyword of SQUALL, the controlled English", "tutorial_squall.html"];

// The [text, link] to show about the keyword or operator at *position* of *text*, if any.
export function topicAt(text, position, squall = false) {
  let offset = 0;
  for (const line of text.split("\n")) {
    if (position <= offset + line.length) {
      const column = position - offset;
      let start = 0;
      for (const [kind, fragment] of tokens(line, squall)) {
        const end = start + fragment.length;
        if (start <= column && column <= end && (kind === "keyword" || kind === "operator")) {
          const topic = squall ? SQUALL_TOPIC : TOPICS[fragment];
          if (topic) return { word: fragment, text: topic[0], url: DOCS_URL + topic[1] };
        }
        start = end;
      }
      return null;
    }
    offset += line.length + 1;
  }
  return null;
}
