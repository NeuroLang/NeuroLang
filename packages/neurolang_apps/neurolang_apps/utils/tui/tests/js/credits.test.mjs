import assert from "node:assert/strict";
import test from "node:test";
import { creditParts, creditSections, creditsText } from "../../../../../../../doc/_static/repl/credits.js";

test("a DOI links to doi.org, without the stop after it", () => {
  assert.deepEqual(creditParts("NeuroImage 54:313-327 (2011). doi:10.1016/j.neuroimage.2010.07.033."), [
    { text: "NeuroImage 54:313-327 (2011). " },
    { text: "doi:10.1016/j.neuroimage.2010.07.033", href: "https://doi.org/10.1016/j.neuroimage.2010.07.033" },
    { text: "." },
  ]);
});

test("an address links to itself, without the parenthesis that closes it", () => {
  assert.deepEqual(creditParts("(licence; https://www.nitrc.org/projects/iit/); LUT"), [
    { text: "(licence; " },
    { text: "https://www.nitrc.org/projects/iit/", href: "https://www.nitrc.org/projects/iit/" },
    { text: "); LUT" },
  ]);
  assert.deepEqual(creditParts("RecoBundles (doi:10.6084/m9.figshare.7375883): G")[1],
    { text: "doi:10.6084/m9.figshare.7375883", href: "https://doi.org/10.6084/m9.figshare.7375883" });
});

test("a text without a link is one piece", () => {
  assert.deepEqual(creditParts("SciRepEval. EMNLP 2023"), [{ text: "SciRepEval. EMNLP 2023" }]);
});

test("each engine of the bundle has its section, in order, and an old bundle none of the texts", () => {
  const bundle = {
    default: "neurosynth",
    engines: {
      neurosynth: { label: "Neurosynth", data: "a.zip", acknowledgements: ["Neurosynth: Yarkoni T."] },
      neurostore: { label: "NeuroStore", data: "b.zip" },
    },
  };
  assert.deepEqual(creditSections(bundle), [
    { name: "neurosynth", label: "Neurosynth", texts: ["Neurosynth: Yarkoni T."] },
    { name: "neurostore", label: "NeuroStore", texts: [] },
  ]);
  assert.deepEqual(creditSections({ data: "n.zip" }), [{ name: "neurosynth", label: "Neurosynth", texts: [] }]);
});

test("the texts to copy are a paragraph each, under the engine", () => {
  assert.equal(creditsText({ name: "neurosynth", label: "Neurosynth", texts: ["A.", "B."] }),
    "Neurosynth (neurosynth):\n\nA.\n\nB.");
});
