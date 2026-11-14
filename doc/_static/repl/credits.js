// The credits of the REPL (credits.html): for each engine of the page, the
// sources of its data and how to acknowledge them, as engines.yaml's
// `acknowledgements` give them (bundle.json carries them; `.credits` prints
// those of the session's engine).
import { bundleEngines } from "./engines.js";

// A DOI written doi:10.…, or an address; neither ends with the stop or the
// parenthesis that follows it in the text.
const LINK = /doi:\s?(10\.\d{4,9}\/[^\s;,()]*[^\s;,.()])|(https?:\/\/[^\s;,()]*[^\s;,.()])/g;

// The pieces of *text*: { text } and, for a DOI or an address, { text, href }.
export function creditParts(text) {
  const parts = [];
  let at = 0;
  for (const match of text.matchAll(LINK)) {
    if (match.index > at) parts.push({ text: text.slice(at, match.index) });
    const [whole, doi, url] = match;
    parts.push({ text: whole, href: doi ? `https://doi.org/${doi}` : url });
    at = match.index + whole.length;
  }
  if (at < text.length) parts.push({ text: text.slice(at) });
  return parts;
}

// The engines of *bundle*, in its order: { name, label, texts }; a bundle of
// before the credits has no texts.
export function creditSections(bundle) {
  return Object.entries(bundleEngines(bundle)).map(([name, entry]) => ({
    name, label: entry.label, texts: entry.acknowledgements ?? [],
  }));
}

// The texts of an engine to paste in a paper: a paragraph each.
export function creditsText(section) {
  return [`${section.label} (${section.name}):`, ...section.texts].join("\n\n");
}

// The page of the credits of *bundle* in *main* (an element of *doc*).
export function renderCredits(bundle, main, doc = document) {
  const make = (tag, props = {}, ...children) => {
    const node = Object.assign(doc.createElement(tag), props);
    node.append(...children);
    return node;
  };
  const sections = creditSections(bundle);
  const index = make("ul", { className: "engines" }, ...sections.map(
    (s) => make("li", {}, make("a", { href: `#${s.name}`, textContent: s.label }))));
  main.append(make("nav", { ariaLabel: "Engines" }, index));
  for (const section of sections) {
    const copy = make("button", { type: "button", className: "quiet", textContent: "Copy" });
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(creditsText(section));
        copy.textContent = "Copied";
      } catch {
        copy.textContent = "Select the text to copy it";
      }
    });
    const list = section.texts.length
      ? make("ol", {}, ...section.texts.map((text) => make("li", {}, ...creditParts(text).map(
        (part) => (part.href ? make("a", { href: part.href, textContent: part.text }) : part.text)))))
      : make("p", { className: "note", textContent: "This build of the page names no source for this engine: .credits in the terminal does." });
    main.append(make("section", { id: section.name },
      make("div", { className: "head" },
        make("h2", {}, section.label, " ", make("code", { textContent: section.name })),
        ...(section.texts.length ? [copy] : [])),
      list));
  }
}
