import { test } from "node:test";
import assert from "node:assert/strict";

import { removedUploads, uploadRows, uploadsSummary } from "../../../../../../../doc/_static/repl/uploads.js";
import { withoutUploads } from "../../../../../../../doc/_static/repl/session.js";

const uploads = {
  files: [
    { name: "my scores.csv", size: 5_013_351, age: 9000, loaded: ["scores"] },
    { name: "roi.nii.gz", size: 820, age: 30, loaded: [] },
  ],
  used: 5_014_171,
  most: 200 << 20,
};

test("the uploads are summed up, with the room taken", () => {
  assert.equal(uploadsSummary(uploads), "Uploaded files (2): 4.8 MB of 200 MB used");
  assert.equal(uploadsSummary({ files: [], used: 0, most: 1 }), "");
  assert.equal(uploadsSummary(undefined), "");
});

test("each upload is a row: name, kind, size, what it was loaded as", () => {
  assert.deepEqual(uploadRows(uploads), [
    ["my scores.csv", "table", "4.8 MB", "scores"],
    ["roi.nii.gz", "image", "820 B", "not loaded"],
  ]);
  const tracts = { files: [{ name: "AF_L.trk", size: 820, age: 3, loaded: ["af_l"] }] };
  assert.deepEqual(uploadRows(tracts), [["AF_L.trk", "tractogram", "820 B", "af_l"]]);
});

test("the files a reply no longer lists are those removed", () => {
  const kept = ["my scores.csv", "roi.nii.gz", "old.csv"];
  assert.deepEqual(removedUploads(kept, uploads), ["old.csv"]);
  assert.deepEqual(removedUploads(kept, undefined), []); // a reply of an older version
});

test("the statements loading a removed file are not run again", () => {
  const entries = [
    { text: ".load 'old.csv' old", mode: "datalog" },
    { text: ".load old.csv", mode: "datalog" },
    { text: ".load 'my scores.csv' scores", mode: "datalog" },
    { text: "a(x) :- old(x)", mode: "datalog" },
    { text: ".image roi.nii.gz --name image", mode: "datalog" },
  ];
  assert.deepEqual(withoutUploads(entries, ["old.csv", "roi.nii.gz"]).map((e) => e.text), [
    ".load 'my scores.csv' scores", "a(x) :- old(x)",
  ]);
});
