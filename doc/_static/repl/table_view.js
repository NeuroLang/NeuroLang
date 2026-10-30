// The table viewer of a REPL result: a table to sort by clicking a column,
// filter, page through, copy and download. The rows are the model's
// (table_model.js).
import { TableModel, cellText } from "./table_model.js";

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

export function renderTable(payload, { download = null } = {}) {
  const model = new TableModel(payload);
  let pageIndex = 0;
  let pageSize = PAGE_SIZES[0];

  const title = payload.title ? element("strong", { textContent: payload.title }) : "";
  const count = element("span", { className: "rt-count" });
  const filter = element("input", {
    type: "search", className: "rt-filter", spellcheck: false,
    placeholder: "Filter rows (name:text for one column)",
  });
  filter.setAttribute("aria-label", "Filter the rows");
  const size = element("select", { className: "rt-size" },
    ...PAGE_SIZES.map((n) => element("option", { value: n, textContent: `${n} rows` })));
  size.setAttribute("aria-label", "Rows per page");
  const previous = element("button", { type: "button", textContent: "‹", title: "Previous page" });
  const next = element("button", { type: "button", textContent: "›", title: "Next page" });
  const range = element("span", { className: "rt-range" });
  const copy = element("button", { type: "button", textContent: "Copy", title: "Copy the rows shown as tab-separated text" });
  const csv = element("button", { type: "button", textContent: "CSV", title: "Download the rows shown as CSV" });
  const head = element("tr");
  const body = element("tbody");
  const table = element("table", { className: "rt-table" }, element("thead", {}, head), body);
  const root = element("section", { className: "rt" },
    element("header", { className: "rt-bar" }, title, count, filter, size, previous, range, next, copy, csv,
      ...(download && payload.block !== undefined ? [downloadMenu(payload, download)] : [])),
    element("div", { className: "rt-scroll" }, table));

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

  function renderBody() {
    const page = model.page(pageIndex, pageSize);
    pageIndex = page.index;
    body.replaceChildren(...page.rows.map(({ number, cells }) => element("tr", {},
      element("td", { className: "rt-number", textContent: number }),
      ...cells.map((value, column) => {
        const text = shown(value, model.kinds[column]);
        return element("td", {
          className: NUMERIC.has(model.kinds[column]) ? "rt-num" : "", textContent: text, title: text,
        });
      }))));
    const total = model.visibleCount;
    count.textContent = model.filterText
      ? `${total} of ${model.rows.length} rows`
      : `${model.total} row${model.total === 1 ? "" : "s"}` +
        (model.truncated ? ` (the first ${model.rows.length} shown)` : "");
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
