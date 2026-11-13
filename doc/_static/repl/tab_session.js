// The id of the session of a tab, for a server that keeps one per tab
// (neurolang-repl-serve --local --users): a reload keeps it; a duplicated
// tab, which has a copy of the tab's sessionStorage, takes a new one.

const KEY = "neurolang-session";
const CHANNEL = "neurolang-repl-tabs";

const newId = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, "0")).join("");

// The id of this tab, kept in *storage* (sessionStorage, or null when the
// browser refuses it). Another live tab with the same id answers on a
// BroadcastChannel (*Channel*) within *wait* ms: then this tab takes a new
// id. Without storage or channel, it does without.
export async function tabSession(storage, Channel = globalThis.BroadcastChannel, wait = 150) {
  let id = null;
  try {
    id = storage?.getItem(KEY) ?? null;
  } catch {
    // refused: an id for this page only
  }
  let channel = null;
  try {
    channel = Channel ? new Channel(CHANNEL) : null;
  } catch {
    // no channel: the id is kept as it is
  }
  let taken = false;
  if (channel) {
    channel.onmessage = ({ data }) => {
      if (data?.ask && data.ask === id) channel.postMessage({ mine: id });
      else if (data?.mine && data.mine === id) taken = true;
    };
    if (id) {
      channel.postMessage({ ask: id });
      await new Promise((resolve) => setTimeout(resolve, wait));
    }
  }
  if (!id || taken) {
    id = newId();
    try {
      storage?.setItem(KEY, id);
    } catch {
      // refused: an id for this page only
    }
  }
  return id;
}
