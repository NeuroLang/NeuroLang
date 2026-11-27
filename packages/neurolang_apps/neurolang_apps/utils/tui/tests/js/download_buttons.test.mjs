import assert from "node:assert/strict";
import test from "node:test";

// The least of a DOM that the buttons of the brain tab are made of.
const node = (tag) => ({
  tag, children: [], listeners: {},
  append(...children) { this.children.push(...children); },
  addEventListener(type, listener) { this.listeners[type] = listener; },
});
globalThis.document = { createElement: node };

const { imageButtons } = await import("../../../../../../../doc/_static/repl/table_view.js");

test("the brain tab offers the streamlines, the network and their layers, not only images", () => {
  const asked = [];
  const bar = imageButtons([
    { block: 0, title: "seen", nifti: [null], files: [] },
    { block: 1, title: "tracts", nifti: null, files: [{ kind: "trk", label: "Streamlines, TRK" }, { kind: "layer", label: "Viewer layer, NIfTI" }] },
    { block: 2, title: "pairs", nifti: null, files: [{ kind: "network", label: "Network edges, CSV" }] },
  ], (...args) => asked.push(args));
  assert.deepEqual(bar.children.map((b) => b.textContent), [
    "Download NIfTI: seen",
    "Download Streamlines, TRK: tracts",
    "Download Viewer layer, NIfTI: tracts",
    "Download Network edges, CSV: pairs",
  ]);
  bar.children.forEach((b) => b.listeners.click());
  assert.deepEqual(asked, [[0, "nifti", ""], [1, "trk", ""], [1, "layer", ""], [2, "network", ""]]);
});

test("a result of streamlines alone still has its bar", () => {
  assert.ok(imageButtons([{ block: 0, nifti: null, files: [{ kind: "trk", label: "Streamlines, TRK" }] }], () => {}));
});
