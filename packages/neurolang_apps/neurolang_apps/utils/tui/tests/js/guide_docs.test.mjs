import assert from "node:assert/strict";
import test from "node:test";
import { columnDocs, exampleQuestion, queryOf, questionOf, filterQuestions, matchesCommand, showAllQuery, startingQuery, suggestedQuestion } from "../../../../../../../doc/_static/repl/guide.js";

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

test("an example question is suggested when the assistant can answer", () => {
  const questions = ["Which regions does pain activate?", "Where is reading?", "What about faces?"];
  assert.equal(suggestedQuestion(questions, "", true, () => 0), questions[0]);
  assert.equal(suggestedQuestion(questions, "", true, () => 0.5), questions[1]);
  assert.equal(suggestedQuestion(questions, "", true, () => 0.9999), questions[2]);
  // not without the assistant, over the visitor's question, nor without questions
  assert.equal(suggestedQuestion(questions, "", false), "");
  assert.equal(suggestedQuestion(questions, "my own question", true), "");
  assert.equal(suggestedQuestion([], "", true), "");
});

test("a first visit starts with the first example of the language in the box", () => {
  const examples = { datalog: ["ans(s) :- mentions(s, 'pain')", "ans(s) :- x(s)"], squall: ["define as ..."] };
  assert.equal(startingQuery(examples, "datalog", { box: "", visited: false }), examples.datalog[0]);
  assert.equal(startingQuery(examples, "squall", { box: "", visited: false }), examples.squall[0]);
  // Not over what is in the box, nor for someone who has run queries here before.
  assert.equal(startingQuery(examples, "datalog", { box: "ans(x) :- y(x)", visited: false }), "");
  assert.equal(startingQuery(examples, "datalog", { box: "", visited: true }), "");
  assert.equal(startingQuery({}, "datalog", { box: "", visited: false }), "");
});

test("the example of the box of the assistant is another question, at random", () => {
  const questions = ["a", "b", "c"];
  assert.equal(exampleQuestion(questions, "", () => 0), "a");
  assert.equal(exampleQuestion(questions, "a", () => 0), "b");
  assert.equal(exampleQuestion(questions, "", () => 0.999), "c");
  assert.equal(exampleQuestion(["a"], "a"), "");
  assert.equal(exampleQuestion([], ""), "");
});

test("the questions are filtered by all the words typed, in any case", () => {
  const questions = ["Which studies mention pain?", "Which regions are active in memory?", "How many studies?"];
  assert.deepEqual(filterQuestions(questions, "STUDIES which"), [questions[0]]);
  assert.deepEqual(filterQuestions(questions, "  "), questions);
  assert.deepEqual(filterQuestions(questions, "fear"), []);
});

test("a question and its query find each other, in the language chosen", () => {
  const queries = { "Which studies mention pain?": { datalog: "ans(s) :-\n  mentions(s, 'pain')", squall: "obtain every Study." }, "No query": {} };
  assert.equal(queryOf(queries, "Which studies mention pain?", "datalog"), queries["Which studies mention pain?"].datalog);
  assert.equal(queryOf(queries, "Which studies mention pain?", "squall"), "obtain every Study.");
  assert.equal(queryOf(queries, "No query", "datalog"), "");
  assert.equal(queryOf(queries, "unknown", "datalog"), "");
  assert.equal(questionOf(queries, "ans(s) :-   mentions(s, 'pain')", "datalog"), "Which studies mention pain?");
  assert.equal(questionOf(queries, "ans(s) :- other(s)", "datalog"), "");
});
