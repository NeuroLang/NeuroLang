// The queries that go on from a result. A result is not a relation of the
// session (``ans`` cannot be used in a body), so the query that made it is
// given again with its ``ans`` head renamed, and a new ``ans`` rule on it,
// for the visitor to complete or run.

const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;
const ANSWER_HEAD = /^(\s*)ans\(/gm;
const VOXEL = ["i", "j", "k"];

// *query* with the heads of its answers renamed ``result``, or null if it has none.
function asResult(query) {
  if (!query || !ANSWER_HEAD.test(query)) return null;
  ANSWER_HEAD.lastIndex = 0;
  return query.trimEnd().replace(ANSWER_HEAD, "$1result(");
}

// The names for the voxel's coordinates and a count that no column has.
function freshNames(columns) {
  const taken = new Set(columns);
  const fresh = (name) => {
    let candidate = name;
    while (taken.has(candidate)) candidate += "_";
    taken.add(candidate);
    return candidate;
  };
  return { coordinates: VOXEL.map(fresh), count: fresh("n") };
}

// The buttons of a result: each has a *label*, a *title* and the *text* of the
// query it puts in the box. *peaks* tells that the engine has the peaks the
// studies report, from which a map of the studies is made.
export function followUps({ columns, kinds = [] }, query, { squall = false, peaks = false } = {}) {
  const result = squall ? null : asResult(query);
  if (result === null || !columns.length || !columns.every((column) => IDENTIFIER.test(column))) return [];
  const variables = columns.join(", ");
  const actions = [{
    label: "Use in a query",
    title: "Put this query in the box again with a new answer on its result",
    text: `${result}\nans(${variables}) :- result(${variables}), `,
  }];
  if (peaks) {
    const { coordinates: [i, j, k], count } = freshNames(columns);
    const studies = columns.filter((_, index) => kinds[index] === "int").slice(0, 3);
    for (const study of studies) {
      actions.push({
        label: studies.length === 1 ? "Map the studies" : `Map the studies in ${study}`,
        title: "Count, for each voxel, the studies of this result that report a peak there, to see it in the brain",
        text: `${result}\nans(${i}, ${j}, ${k}, ${count}) :- AGGREGATE[${i}, ${j}, ${k}]` +
          `(result(${variables}), peak_reported(${i}, ${j}, ${k}, ${study}) @ count(${study})) = ${count}`,
      });
    }
  }
  return actions;
}
