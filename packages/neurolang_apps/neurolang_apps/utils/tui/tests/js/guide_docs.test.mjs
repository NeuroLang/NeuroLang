import assert from "node:assert/strict";
import test from "node:test";
import { columnDocs, matchesCommand, showAllQuery } from "../../../../../../../doc/_static/repl/guide.js";

const peaks = {
  name: "peak_reported", signature: "peak_reported(voxel(i, j, k), study(study_id))", kind: "stored",
  arguments: [
    { columns: ["i", "j", "k"], sort: "voxel", type: "int" },
    { columns: ["study_id"], sort: "study", type: "int" },
  ],
};

test("the columns of a predicate come one by one with their sort and type", () => {
  assert.deepEqual(columnDocs(peaks), [
    { name: "i", sort: "voxel", type: "int" }, { name: "j", sort: "voxel", type: "int" },
    { name: "k", sort: "voxel", type: "int" }, { name: "study_id", sort: "study", type: "int" },
  ]);
  assert.deepEqual(columnDocs({ name: "person" }), []);
});

test("every row of a predicate is a query of its columns", () => {
  assert.equal(showAllQuery(peaks), "ans(i, j, k, study_id) :- peak_reported(i, j, k, study_id)");
  assert.equal(showAllQuery({ name: "person", arguments: [] }), null);
  assert.equal(showAllQuery({ ...peaks, name: "dcterms:Title" }), null);
});

test("a command matches by the words of its usage and its description", () => {
  const view = { name: ".view", usage: ".view [COLUMN]", description: "Show the last result in the brain viewer" };
  assert.equal(matchesCommand(view, ""), true);
  assert.equal(matchesCommand(view, "brain view"), true);
  assert.equal(matchesCommand(view, "BRAIN"), true);
  assert.equal(matchesCommand(view, "sort"), false);
});
