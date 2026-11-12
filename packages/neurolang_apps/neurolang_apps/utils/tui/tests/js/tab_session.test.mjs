import assert from "node:assert/strict";
import test from "node:test";
import { tabSession } from "../../../../../../../doc/_static/repl/tab_session.js";

// The sessionStorage of a tab.
class Storage {
  constructor(items = {}) {
    this.items = { ...items };
  }
  getItem(key) {
    return this.items[key] ?? null;
  }
  setItem(key, value) {
    this.items[key] = String(value);
  }
}

// Channels between the tabs of one browser, as BroadcastChannel: a message
// goes to the others.
function channels() {
  const all = [];
  return class {
    constructor() {
      all.push(this);
    }
    postMessage(data) {
      for (const other of all) if (other !== this) setTimeout(() => other.onmessage?.({ data }));
    }
  };
}

test("a tab keeps its id, which a reload finds", async () => {
  const storage = new Storage();
  const id = await tabSession(storage, channels());
  assert.match(id, /^[0-9a-f]{32}$/);
  assert.equal(await tabSession(storage, channels()), id);
});

test("a duplicated tab takes an id of its own", async () => {
  const Channel = channels();
  const first = new Storage();
  const id = await tabSession(first, Channel);
  const copy = new Storage(first.items); // the duplicate's copy
  const other = await tabSession(copy, Channel);
  assert.notEqual(other, id);
  assert.equal(copy.getItem("neurolang-session"), other);
  assert.equal(first.getItem("neurolang-session"), id);
});

test("without storage or channel a tab still has an id", async () => {
  const refused = {
    getItem() {
      throw new Error("SecurityError");
    },
    setItem() {
      throw new Error("SecurityError");
    },
  };
  assert.match(await tabSession(refused, null), /^[0-9a-f]{32}$/);
  assert.match(await tabSession(null, null), /^[0-9a-f]{32}$/);
  const Broken = class {
    constructor() {
      throw new Error("no channel");
    }
  };
  const storage = new Storage({ "neurolang-session": "abc" });
  assert.equal(await tabSession(storage, Broken), "abc");
});
