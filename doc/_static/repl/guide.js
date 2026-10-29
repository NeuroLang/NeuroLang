// The lists of the page that put text in the code box: the examples of the
// engine and its predicates. The page gives the box; these only draw the lists.

function element(tag, properties = {}, ...children) {
  const node = Object.assign(document.createElement(tag), properties);
  node.append(...children);
  return node;
}

// The text a predicate puts in the box: its name, applied (in Datalog).
export const predicateText = (predicate, squall) => (squall ? predicate.name : `${predicate.name}(`);

// Whether *predicate* matches the *filter*, the words of which are all in its name,
// signature, kind or description.
export function matches(predicate, filter) {
  const text = `${predicate.name} ${predicate.signature} ${predicate.kind} ${predicate.description ?? ""}`.toLowerCase();
  return filter.toLowerCase().split(/\s+/).filter(Boolean).every((word) => text.includes(word));
}

// Fill *list* with a button per example; a click gives its text to *pick*.
export function renderExamples(list, queries, pick) {
  list.replaceChildren(...(queries ?? []).map((query) => {
    const button = element("button", { type: "button", className: "guide-item", textContent: query, title: "Put this query in the box" });
    button.addEventListener("click", () => pick(query));
    return element("li", {}, button);
  }));
}

// Fill *list* with the predicates that match *filter*; a click gives the text of the predicate to *pick*.
export function renderPredicates(list, predicates, filter, pick, squall = false, limit = 200) {
  const shown = predicates.filter((predicate) => matches(predicate, filter));
  list.replaceChildren(...shown.slice(0, limit).map((predicate) => {
    const button = element("button", {
      type: "button", className: "guide-item", title: predicate.description || "Put this predicate in the box",
    }, element("code", { textContent: predicate.signature }), element("span", { className: "guide-kind", textContent: predicate.kind }));
    button.addEventListener("click", () => pick(predicateText(predicate, squall)));
    return element("li", {}, button);
  }));
  if (shown.length > limit) list.append(element("li", { className: "note", textContent: `…and ${shown.length - limit} more: filter them.` }));
  return shown.length;
}
