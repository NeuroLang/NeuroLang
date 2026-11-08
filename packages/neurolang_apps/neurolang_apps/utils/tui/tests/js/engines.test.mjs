import assert from "node:assert/strict";
import test from "node:test";
import { bundleEngines, dataParts, defaultEngine, engineChoice, joinParts, withEngine } from "../../../../../../../doc/_static/repl/engines.js";

const bundle = {
  wheels: [], default: "neurosynth",
  engines: {
    neurosynth: { label: "Neurosynth", data: "a.zip", files: {} },
    neurostore: { label: "NeuroStore", data: "b.zip", files: {} },
  },
};
const params = (query) => new URLSearchParams(query);

test("a bundle from before the choice has the engine neurosynth", () => {
  const old = { wheels: [], data: "n.zip", files: { x: ["y"] } };
  assert.deepEqual(bundleEngines(old), { neurosynth: { label: "Neurosynth", data: "n.zip", files: { x: ["y"] } } });
  assert.equal(defaultEngine(old), "neurosynth");
});

test("the engine asked for is started, else the default", () => {
  assert.deepEqual(engineChoice(params("engine=neurostore"), bundle), { name: "neurostore", full: false });
  assert.deepEqual(engineChoice(params(""), bundle), { name: "neurosynth", full: false });
  assert.deepEqual(engineChoice(params("engine=nowhere"), bundle), { name: "neurosynth", full: false });
});

test("engine=full is the default engine with its analyses", () => {
  assert.deepEqual(engineChoice(params("engine=full"), bundle), { name: "neurosynth", full: true });
});

test("an engine that the bundle left out is not offered", () => {
  const partial = { ...bundle, engines: { neurosynth: bundle.engines.neurosynth } };
  assert.deepEqual(engineChoice(params("engine=neurostore"), partial), { name: "neurosynth", full: false });
});

test("the address keeps the other options and names a non-default engine", () => {
  const next = withEngine(params("ref=main&repo=a/b"), "neurostore", bundle);
  assert.equal(next.get("engine"), "neurostore");
  assert.equal(next.get("ref"), "main");
  const back = withEngine(next, "neurosynth", bundle);
  assert.equal(back.has("engine"), false);
  assert.equal(back.get("repo"), "a/b");
});

test("the data of an engine is a zip or its parts, joined", () => {
  assert.deepEqual(dataParts({ data: "a.zip" }), ["a.zip"]);
  assert.deepEqual(dataParts({ data: ["a.zip.0", "a.zip.1"] }), ["a.zip.0", "a.zip.1"]);
  const one = new Uint8Array([1, 2]);
  assert.equal(joinParts([one]), one);
  assert.deepEqual([...joinParts([one, new Uint8Array([3]), new Uint8Array([4, 5])])], [1, 2, 3, 4, 5]);
});
