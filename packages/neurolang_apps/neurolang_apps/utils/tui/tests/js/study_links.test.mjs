import assert from "node:assert/strict";
import test from "node:test";
import {
  PmcLookup, citation, doiUrl, europePmcUrl, parseEuropePmc, pmcUrl, pubmedUrl, studyColumns,
} from "../../../../../../../doc/_static/repl/study_links.js";

test("the links of a study", () => {
  assert.equal(pubmedUrl(9065511), "https://pubmed.ncbi.nlm.nih.gov/9065511/");
  assert.equal(doiUrl("10.1523/JNEUROSCI.17-07-02512.1997"), "https://doi.org/10.1523/JNEUROSCI.17-07-02512.1997");
  assert.equal(doiUrl("10.1002/(SICI)1097 x"), "https://doi.org/10.1002/(SICI)1097%20x");
  assert.equal(pmcUrl("PMC6573507"), "https://pmc.ncbi.nlm.nih.gov/articles/PMC6573507/");
});

test("the study columns are those the engine tells", () => {
  assert.deepEqual(studyColumns({ study_links: { columns: [0, 2], records: {} } }), [0, 2]);
  assert.deepEqual(studyColumns({}), []);
});

test("a citation has what the record has", () => {
  assert.equal(citation({ title: "T", authors: "A B", year: 1997, journal: "J" }), "T. A B (1997) J");
  assert.equal(citation({ title: "T" }), "T");
  assert.equal(citation({ title: "Anatomy of motor learning. ", authors: "A" }), "Anatomy of motor learning. A");
  assert.equal(citation({ authors: "A B", year: 1997 }), "A B (1997)");
  assert.equal(citation(), "");
});

test("the request asks for the studies by their PubMed id", () => {
  const url = new URL(europePmcUrl([1, 22]));
  assert.equal(url.origin + url.pathname, "https://www.ebi.ac.uk/europepmc/webservices/rest/search");
  assert.equal(url.searchParams.get("query"), "SRC:MED AND (EXT_ID:1 OR EXT_ID:22)");
  assert.equal(url.searchParams.get("pageSize"), "2");
});

test("the answer gives the PubMed Central id and the DOI, by PubMed id", () => {
  const answer = { resultList: { result: [
    { pmid: "9065511", pmcid: "PMC6573507", doi: "10.1/x" },
    { pmid: "5", doi: "10.2/y" },
    { pmid: "6" },
    { pmcid: "PMC1" },
  ] } };
  assert.deepEqual([...parseEuropePmc(answer)], [
    ["9065511", { pmcid: "PMC6573507", doi: "10.1/x" }], ["5", { pmcid: undefined, doi: "10.2/y" }],
  ]);
  assert.equal(parseEuropePmc({}).size, 0);
});

const respond = (calls, bodies) => async (url) => {
  calls.push(url);
  const body = bodies.shift();
  if (body instanceof Error) throw body;
  return { ok: body !== null, json: async () => body };
};

test("the lookup asks once for each study, fifty at a time, and remembers", async () => {
  const calls = [];
  const lookup = new PmcLookup(respond(calls, [
    { resultList: { result: [{ pmid: "1", pmcid: "PMC1" }] } }, { resultList: { result: [] } },
  ]));
  const ids = Array.from({ length: 60 }, (_, i) => i + 1);
  const known = await lookup.lookup(ids);
  assert.equal(calls.length, 2);
  assert.deepEqual(known.get("1"), { pmcid: "PMC1", doi: undefined });
  await lookup.lookup(ids);
  assert.equal(calls.length, 2);
});

test("a refused or failed request leaves the links as they are, and is not repeated", async () => {
  const calls = [];
  const lookup = new PmcLookup(respond(calls, [new Error("offline"), null]));
  assert.equal((await lookup.lookup([1, 2])).size, 0);
  assert.equal((await lookup.lookup([1, 2, 3])).size, 0); // only 3 is new
  assert.equal(calls.length, 2);
  assert.equal(new URL(calls[1]).searchParams.get("query"), "SRC:MED AND (EXT_ID:3)");
});

test("only numbers are asked for", async () => {
  const calls = [];
  const lookup = new PmcLookup(respond(calls, []));
  await lookup.lookup(["abc", "1 OR EXT_ID:2", ""]);
  assert.equal(calls.length, 0);
});
