import assert from "node:assert/strict";
import test from "node:test";
import { frameHeight, themeMessage, themed } from "../../../../../../../doc/_static/repl/frames.js";

test("a page takes the theme of the REPL on its html tag", () => {
  const page = '<!DOCTYPE html>\n<html lang="en"><head><title>x</title></head><body><html>kept</body></html>';
  assert.equal(
    themed(page, "light"),
    '<!DOCTYPE html>\n<html lang="en" data-theme="light"><head><title>x</title></head><body><html>kept</body></html>',
  );
  assert.ok(themed("<html><body></body></html>", "dark").startsWith('<html data-theme="dark">'));
});

test("a page that has a theme is given the new one, and an unknown theme is dark", () => {
  const page = '<html lang="en" data-theme="light"><body></body></html>';
  assert.equal(themed(page, "dark"), '<html lang="en" data-theme="dark"><body></body></html>');
  assert.ok(themed(page, '"><script>').startsWith('<html lang="en" data-theme="dark">'));
});

test("the message of a theme", () => {
  assert.deepEqual(themeMessage("light"), { type: "neurolang-theme", theme: "light" });
  assert.deepEqual(themeMessage("sepia"), { type: "neurolang-theme", theme: "dark" });
});

test("the height a page reports is for the frame it comes from", () => {
  const first = { contentWindow: { id: 1 } };
  const second = { contentWindow: { id: 2 } };
  const message = (source, data) => ({ source, data });
  assert.deepEqual(
    frameHeight(message(second.contentWindow, { type: "neurolang-height", height: 612.2 }), [first, second]),
    { frame: second, height: 613 },
  );
});

test("a height that is not from a frame of the page, or is not one, is ignored", () => {
  const frame = { contentWindow: { id: 1 } };
  const from = frame.contentWindow;
  const height = (value) => ({ source: from, data: { type: "neurolang-height", height: value } });
  assert.equal(frameHeight({ source: { id: 9 }, data: { type: "neurolang-height", height: 300 } }, [frame]), null);
  assert.equal(frameHeight({ source: from, data: { type: "other", height: 300 } }, [frame]), null);
  assert.equal(frameHeight({ source: from, data: "neurolang-height" }, [frame]), null);
  assert.equal(frameHeight({ source: from }, [frame]), null);
  for (const value of ["tall", NaN, Infinity, -1, 0, 1e9, null]) assert.equal(frameHeight(height(value), [frame]), null);
});
