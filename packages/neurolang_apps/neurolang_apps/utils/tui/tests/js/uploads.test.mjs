import { test } from "node:test";
import assert from "node:assert/strict";

import { uploadRows, uploadsSummary } from "../../../../../../../doc/_static/repl/uploads.js";

const uploads = {
  files: [
    { name: "my scores.csv", size: 5_013_351, age: 9000, loaded: ["scores"] },
    { name: "roi.nii.gz", size: 820, age: 30, loaded: [] },
  ],
  used: 5_014_171,
  most: 200 << 20,
};

test("the uploads are summed up, with the room taken", () => {
  assert.equal(uploadsSummary(uploads), "Uploaded files: 2, 4.8 MB of 200 MB");
  assert.equal(uploadsSummary({ files: [], used: 0, most: 1 }), "");
  assert.equal(uploadsSummary(undefined), "");
});

test("each upload is a row: name, kind, size, what it was loaded as", () => {
  assert.deepEqual(uploadRows(uploads), [
    ["my scores.csv", "table", "4.8 MB", "scores"],
    ["roi.nii.gz", "image", "820 B", "not loaded"],
  ]);
});
