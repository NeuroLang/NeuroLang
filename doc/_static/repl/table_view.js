// The table viewer of a REPL result: a table to sort by clicking a column,
// filter, page through, copy and download. The rows are the model's
// (table_model.js).
import { TableModel, cellText } from "./table_model.js";
import { citation, doiUrl, pmcUrl, pubmedUrl, studyColumns } from "./study_links.js";

const PAGE_SIZES = [20, 50, 100, 500];
const NUMERIC = new Set(["int", "float"]);

function element(tag, properties = {}, ...children) {
  const node = Object.assign(document.createElement(tag), properties);
  node.append(...children);
  return node;
}

function download(name, text, type) {
  const link = element("a", {
    href: URL.createObjectURL(new Blob([text], { type })),
    download: name,
  });
  link.click();
  URL.revokeObjectURL(link.href);
}

// The number as the table shows it: no more than 10 significant digits.
function shown(value, kind) {
  if (kind === "float" && typeof value === "number") return String(Number(value.toPrecision(10)));
  return cellText(value);
}

// The image of a result as the menu names it: its value column, or its mask.
const imageName = (value) => (value === null ? "NIfTI image (mask)" : `NIfTI image: ${value}`);

// The menu of the files a result can be saved as, all its rows as text and its
// NIfTI image; *download*(block, kind, value) asks the worker for the file.
function downloadMenu(payload, download) {
  const item = (text, kind, value = "", reason = "") => {
    const button = element("button", { type: "button", textContent: text, disabled: Boolean(reason), title: reason });
    button.addEventListener("click", () => {
      menu.open = false;
      download(payload.block, kind, value);
    });
    return button;
  };
  const images = Array.isArray(payload.nifti)
    ? payload.nifti.map((value) => item(imageName(value), "nifti", value ?? ""))
    : [item("NIfTI image", "nifti", "", payload.nifti_reason || "This result is not an image")];
  const menu = element("details", { className: "rt-menu" },
    element("summary", { textContent: "Download", title: "Save the whole result as a file" }),
    element("div", { className: "rt-menu-items" }, item("All rows, CSV", "csv"), item("All rows, TSV", "tsv"), ...images));
  return menu;
}

// Buttons for the NIfTI images of the *payloads* (the tables of a reply), for the brain tab.
export function imageButtons(payloads, download) {
  const buttons = payloads.flatMap((payload) => (Array.isArray(payload.nifti)
    ? payload.nifti.map((value) => {
      const label = [payload.title, value].filter(Boolean).join(": ");
      const button = element("button", {
        type: "button", className: "ghost", textContent: label ? `Download NIfTI: ${label}` : "Download NIfTI",
        title: "Save the result as a NIfTI image (.nii.gz)",
      });
      button.addEventListener("click", () => download(payload.block, "nifti", value ?? ""));
      return button;
    })
    : []));
  return buttons.length ? element("div", { className: "row brain-bar" }, ...buttons) : null;
}

// A link that opens in a new tab.
const outside = (href, text, title, className = "") =>
  element("a", { href, target: "_blank", rel: "noopener", textContent: text, title, className });

// The bar of controls of a table: its title and count, the filter, the size and the pages, the copy and
// the downloads, and the buttons of the queries that go on from it. They are returned for the table to listen to.
function controlBar(payload, { download, followUps, use }) {
  const title = payload.title ? element("strong", { textContent: payload.title }) : "";
  const count = element("span", { className: "rt-count" });
  const filter = element("input", {
    type: "search", className: "rt-filter", spellcheck: false,
    placeholder: "Filter rows", title: "Words in any column; name:text for one column",
  });
  filter.setAttribute("aria-label", "Filter the rows");
  const size = element("select", { className: "rt-size" },
    ...PAGE_SIZES.map((n) => element("option", { value: n, textContent: `Show ${n}` })));
  size.setAttribute("aria-label", "Rows per page");
  const previous = element("button", { type: "button", textContent: "‹", title: "Previous page" });
  previous.setAttribute("aria-label", "Previous page");
  const next = element("button", { type: "button", textContent: "›", title: "Next page" });
  next.setAttribute("aria-label", "Next page");
  const range = element("span", { className: "rt-range" });
  const copy = element("button", { type: "button", className: "rt-copy", textContent: "Copy", title: "Copy the rows shown as tab-separated text" });
  const csv = element("button", { type: "button", textContent: "CSV", title: "Download the rows shown as CSV" });
  const goOn = (followUps && use ? followUps(payload) : []).map(({ label, title: hint, text }) =>
    element("button", { type: "button", textContent: label, title: hint, onclick: () => use(text) }));
  const menu = download && payload.block !== undefined ? [downloadMenu(payload, download)] : [];
  const bar = element("header", { className: "rt-bar" }, title, count, filter, size, previous, range, next, copy, csv, ...goOn, ...menu);
  return { bar, count, filter, size, previous, range, next, copy, csv };
}

// A study: its PubMed record, and its DOI and PubMed Central record when they are known.
function studyCell(value, record = {}, found = {}) {
  const doi = record.doi ?? found.doi;
  const text = citation(record);
  return element("td", { className: "rt-num rt-study", title: text },
    outside(pubmedUrl(value), String(value), text || "The PubMed record of the study"),
    ...(doi ? [outside(doiUrl(doi), "DOI", `DOI ${doi}`, "rt-ref")] : []),
    ...(found.pmcid ? [outside(pmcUrl(found.pmcid), "PMC", `${found.pmcid}, found with Europe PMC`, "rt-ref")] : []));
}

// The cell of a *value* of a column of the *kind*, with a bar behind it if it is a probability.
function valueCell(value, kind, probability) {
  const text = shown(value, kind);
  const cell = element("td", { className: NUMERIC.has(kind) ? "rt-num" : "", textContent: text, title: text });
  if (probability && value !== null) {
    cell.classList.add("rt-prob");
    cell.style.setProperty("--bar", `${(Number(value) * 100).toFixed(1)}%`);
  }
  return cell;
}

// What the count of a table says: the rows the filter keeps, or that it is cut.
function countText(model) {
  if (model.filterText) return `${model.visibleCount} of ${model.rows.length} rows`;
  return model.truncated ? `${model.total} rows (the first ${model.rows.length} shown)` : "";
}

// *followUps*(payload) gives the queries that go on from the result, each with a
// *label*, a *title* and its *text*, which *use*(text) puts in the code box.
// *pmc*, if given, looks the PubMed Central records of the studies up (see
// study_links.js) for the pages it shows.
export function renderTable(payload, { download = null, followUps = null, use = null, pmc = null } = {}) {
  const model = new TableModel(payload);
  let pageIndex = 0;
  let pageSize = PAGE_SIZES[0];

  const { bar, count, filter, size, previous, range, next, copy, csv } = controlBar(payload, { download, followUps, use });
  const head = element("tr");
  const body = element("tbody");
  const table = element("table", { className: "rt-table" }, element("thead", {}, head), body);
  const root = element("section", { className: "rt" }, bar, element("div", { className: "rt-scroll" }, table));

  const arrows = { asc: " ▲", desc: " ▼" };
  function renderHead() {
    head.replaceChildren(element("th", { className: "rt-number", textContent: "#" }));
    model.columns.forEach((name, column) => {
      const sorted = model.sort?.column === column ? model.sort.direction : null;
      const extent = model.range(column);
      const th = element("th", {
        className: NUMERIC.has(model.kinds[column]) ? "rt-num" : "",
        title: extent ? `${name}: from ${extent.low} to ${extent.high}` : name,
      }, element("button", { type: "button", textContent: name + (arrows[sorted] ?? "") }));
      if (sorted) th.setAttribute("aria-sort", sorted === "asc" ? "ascending" : "descending");
      th.firstChild.addEventListener("click", () => {
        model.toggleSort(column);
        pageIndex = 0;
        render();
      });
      head.append(th);
    });
  }

  const studies = new Set(studyColumns(payload));
  const records = payload.study_links?.records ?? {};

  // The PubMed Central records of the studies of the page, which are shown when they come.
  function findRecords(page) {
    const ids = page.rows.flatMap(({ cells }) => [...studies].map((column) => cells[column]))
      .filter((id) => Number.isInteger(id));
    if (!pmc || !ids.length) return;
    const before = pmc.known.size;
    pmc.lookup(ids).then(() => {
      if (pmc.known.size > before) renderBody({ lookUp: false });
    });
  }

  function renderBody({ lookUp = true } = {}) {
    const page = model.page(pageIndex, pageSize);
    pageIndex = page.index;
    if (lookUp) findRecords(page);
    body.replaceChildren(...page.rows.map(({ number, cells }) => element("tr", {},
      element("td", { className: "rt-number", textContent: number }),
      ...cells.map((value, column) => {
        if (studies.has(column) && Number.isInteger(value)) return studyCell(value, records[value], pmc?.known.get(String(value)));
        return valueCell(value, model.kinds[column], column === model.probability);
      }))));
    const total = model.visibleCount;
    count.textContent = countText(model);
    range.textContent = total ? `${page.start + 1}–${page.end} of ${total}` : "no rows";
    previous.disabled = page.index === 0;
    next.disabled = page.end >= total;
  }

  function render() {
    renderHead();
    renderBody();
  }

  filter.addEventListener("input", () => {
    model.setFilter(filter.value);
    pageIndex = 0;
    renderBody();
  });
  size.addEventListener("change", () => {
    pageSize = Number(size.value);
    pageIndex = 0;
    renderBody();
  });
  previous.addEventListener("click", () => { pageIndex -= 1; renderBody(); });
  next.addEventListener("click", () => { pageIndex += 1; renderBody(); });
  copy.addEventListener("click", async () => {
    await navigator.clipboard?.writeText(model.toTSV());
    copy.textContent = "Copied";
    setTimeout(() => { copy.textContent = "Copy"; }, 1200);
  });
  csv.addEventListener("click", () => download("neurolang-result.csv", model.toCSV(), "text/csv"));

  render();
  return root;
}
