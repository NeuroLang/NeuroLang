import assert from "node:assert/strict";
import test from "node:test";
import { LlmSettings } from "../../../../../../../doc/_static/repl/llm_settings.js";

const memory = () => {
  const store = new Map();
  return {
    getItem: (k) => store.get(k) ?? null,
    setItem: (k, v) => store.set(k, v),
    removeItem: (k) => store.delete(k),
    dump: () => [...store.values()].join("\n"),
  };
};
const given = { provider: "anthropic", model: "m", baseUrl: "", key: "sk-secret", remember: false };

test("nothing is known before a key is given", () => {
  const settings = new LlmSettings(memory(), memory());
  assert.deepEqual(settings.load(), { provider: "", model: "", baseUrl: "", key: "", remember: false });
});

test("the key stays in the tab's session unless the visitor asks to remember it", () => {
  const durable = memory();
  const session = memory();
  new LlmSettings(durable, session).save(given);
  assert.ok(!durable.dump().includes("sk-secret"));
  assert.equal(new LlmSettings(durable, session).load().key, "sk-secret");
  assert.equal(new LlmSettings(durable, memory()).load().key, "", "a new tab has no key");
  assert.equal(new LlmSettings(durable, memory()).load().provider, "anthropic");
});

test("a remembered key lasts between visits, and is not left in the session", () => {
  const durable = memory();
  const session = memory();
  new LlmSettings(durable, session).save({ ...given, remember: true });
  assert.ok(!session.dump().includes("sk-secret"));
  const loaded = new LlmSettings(durable, memory()).load();
  assert.equal(loaded.key, "sk-secret");
  assert.equal(loaded.remember, true);
});

test("changing from remembering to not remembering removes the key from the disk", () => {
  const durable = memory();
  const session = memory();
  const settings = new LlmSettings(durable, session);
  settings.save({ ...given, remember: true });
  settings.save({ ...given, remember: false });
  assert.ok(!durable.dump().includes("sk-secret"));
});

test("forgetting removes the key everywhere but keeps the other choices", () => {
  const durable = memory();
  const session = memory();
  const settings = new LlmSettings(durable, session);
  settings.save({ ...given, remember: true });
  settings.forget();
  assert.ok(!durable.dump().includes("sk-secret") && !session.dump().includes("sk-secret"));
  assert.equal(settings.load().key, "");
  assert.equal(settings.load().provider, "anthropic");
});

test("broken or missing storage is ignored", () => {
  assert.equal(new LlmSettings({ getItem: () => "{nope", setItem() {}, removeItem() {} }, null).load().key, "");
  const settings = new LlmSettings(null, null);
  settings.save(given);
  assert.equal(settings.load().key, "");
});
