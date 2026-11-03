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

// The action that goes on from the result with an answer of its own, to be completed.
function useAction(result, variables) {
  return {
    label: "Use in a query",
    title: "Put this query in the box again with a new answer on its result",
    text: `${result}\nans(${variables}) :- result(${variables}), `,
  };
}

// The action that maps the studies of the *study* column: for each voxel, how many of them report a peak there.
function mapAction(result, variables, study, label, names) {
  const [i, j, k] = names.coordinates;
  return {
    label,
    title: "Count, for each voxel, the studies of this result that report a peak there, to see it in the brain",
    text: `${result}\nans(${i}, ${j}, ${k}, ${names.count}) :- AGGREGATE[${i}, ${j}, ${k}]` +
      `(result(${variables}), peak_reported(${i}, ${j}, ${k}, ${study}) @ count(${study})) = ${names.count}`,
  };
}

// The actions that map the studies of the integer columns of the result (three at most), one for each.
function mapActions({ columns, kinds }, result, variables) {
  const names = freshNames(columns);
  const studies = columns.filter((_, index) => (kinds ?? [])[index] === "int").slice(0, 3);
  const label = (study) => (studies.length === 1 ? "Map the studies" : `Map the studies in ${study}`);
  return studies.map((study) => mapAction(result, variables, study, label(study), names));
}

// Whether *columns* can be the variables of a rule: there are some, and they are all plain names.
const plain = (columns) => columns.length > 0 && columns.every((column) => IDENTIFIER.test(column));

// The buttons of a result: each has a *label*, a *title* and the *text* of the
// query it puts in the box. *peaks* tells that the engine has the peaks the
// studies report, from which a map of the studies is made.
export function followUps(payload, query, { squall, peaks } = {}) {
  const { columns } = payload;
  const result = squall ? null : asResult(query);
  if (result === null || !plain(columns)) return [];
  const variables = columns.join(", ");
  const actions = [useAction(result, variables)];
  return peaks ? actions.concat(mapActions(payload, result, variables)) : actions;
}
