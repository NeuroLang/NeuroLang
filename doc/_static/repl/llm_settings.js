// What the visitor chose for the AI assistant, and their API key, kept in
// the browser. The key stays in the tab's session (sessionStorage) unless
// the visitor asks to remember it on this device (localStorage); it is
// never in the page's address, nor sent anywhere but to the provider.

const EMPTY = { provider: "", model: "", baseUrl: "", key: "", remember: false };

export class LlmSettings {
  // *durable* keeps the choices between visits, *session* the key of a tab.
  constructor(durable = null, session = null, name = "neurolang-repl-llm") {
    this.durable = durable;
    this.session = session;
    this.name = name;
    this.keyName = `${name}-key`;
  }

  load() {
    let saved = {};
    try {
      saved = JSON.parse(this.durable?.getItem(this.name) ?? "{}") ?? {};
    } catch {
      saved = {};
    }
    const text = (value) => (typeof value === "string" ? value : "");
    const remember = saved.remember === true;
    const store = remember ? this.durable : this.session;
    return {
      ...EMPTY,
      provider: text(saved.provider),
      model: text(saved.model),
      baseUrl: text(saved.baseUrl),
      remember,
      key: text(store?.getItem(this.keyName)),
    };
  }

  save({ provider, model, baseUrl, key, remember }) {
    this.durable?.setItem(this.name, JSON.stringify({ provider, model, baseUrl, remember }));
    this.forget();
    (remember ? this.durable : this.session)?.setItem(this.keyName, key);
  }

  // Forget the key, wherever it is kept.
  forget() {
    this.durable?.removeItem(this.keyName);
    this.session?.removeItem(this.keyName);
  }
}
