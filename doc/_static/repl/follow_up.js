// The query that goes on from a result. A result is not a relation of the
// session (``ans`` cannot be used in a body), so the query that made it is
// given again with its ``ans`` head renamed, and a new ``ans`` rule on it,
// with the columns of the result as variables, for the visitor to complete.

const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;
const ANSWER_HEAD = /^(\s*)ans\(/gm;

export function followUpQuery({ columns }, query, squall = false) {
  if (squall || !query || !columns.length || !columns.every((column) => IDENTIFIER.test(column))) return null;
  if (!ANSWER_HEAD.test(query)) return null;
  ANSWER_HEAD.lastIndex = 0;
  const variables = columns.join(", ");
  return `${query.trimEnd().replace(ANSWER_HEAD, "$1result(")}\nans(${variables}) :- result(${variables}), `;
}
