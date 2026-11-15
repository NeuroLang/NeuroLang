// Stand-in for the server of tools/serve_repl, for the login, account, admin
// and activity pages opened from files: `fetch` answers canned JSON.
// Classic script, injected in the head by build.mjs.
(() => {
  const day = (n) => new Date(Date.now() - n * 864e5).toISOString();
  const providers = [{ id: "anthropic", label: "Anthropic" }, { id: "openai", label: "OpenAI" }];
  const series = Array.from({ length: 7 }, (_, i) => ({ start: day(6 - i), inputs: [12, 40, 33, 8, 51, 27, 19][i], failed: i % 3, cpu: 30 + i * 41, tokens: i * 5200, users: 1 + (i % 4) }));
  const ANSWERS = {
    "/login/options": { guests: true, orcid: true },
    "/account/api/me": { name: "alice", groups: ["lab"], admin: true, orcid: "0000-0002-1825-0097", orcid_name: "Alice Martin", orcid_signs_in: true, orcid_only: false,
      assistant: { assistant: "own", provider: "anthropic", own_key: true }, providers },
    "/admin/api/state": { me: "alice", orcid: true, providers, activity: true, dashboard: false, feedback: false, builder: { rss: 2048, cpu: 311 }, limits: { sessions: 8, running: 2, query_seconds: 600 },
      users: [{ name: "alice", groups: ["lab"], assistant: "own", provider: "anthropic", variables: [], own_key: true }, { name: "0000-0003-1234-5678", orcid_name: "Bob Okoye", groups: [], assistant: "none", variables: ["LAB_KEY"], own_key: false }],
      requests: [{ orcid: "0000-0001-9999-0001", name: "Carol Diaz", when: Date.now() / 1000, message: "PhD student in the lab, I would use it for my project on pain." }],
      groups: [{ name: "lab", members: ["alice"], llm: "anthropic", variables: ["LAB_KEY"], own_key: true }],
      sessions: [{ user: "alice", tab: "abcdef0123", engine: "neurosynth", idle: 42, busy: false, running: true, cpu: 12.5, memory: 1530, peak: "1.9 GB" }] },
    "/admin/api/activity/report": { names: ["alice", "bob"], engines: ["neurosynth"], since: day(7), now: day(0), clipped: false, bucket: "day", series,
      figures: { inputs: 190, failed: 14, users: 4, sessions: 11, cpu: 4200, tokens_in: 200000, tokens_out: 21000, tokens_server: 90000, memory: 2300 },
      uptime: { percent: 99.2, crashes: 1, since: day(7), runs: [{ start: day(7), end: day(3), how: "stopped" }, { start: day(3), end: day(2.9), how: "crashed" }, { start: day(2.8), end: day(0), how: "running" }] },
      users: [{ user: "alice", groups: "lab", inputs: 120, failed: 6, seconds: 5400, cpu: 3100, tokens_in: 1e5, tokens_out: 1e4, memory: 2300, last: day(0.1) }, { user: "bob", groups: "", inputs: 70, failed: 8, seconds: 2100, cpu: 1100, tokens_in: 0, tokens_out: 0, memory: 900, last: day(1) }],
      ends: [{ reason: "Logged out", sessions: 6 }, { reason: "Unused for long", sessions: 4 }, { reason: "Room for another", sessions: 1 }] },
    "/admin/api/activity/inputs": [{ id: 1, created: day(0.1), user: "alice", engine: "neurosynth", text: "ans(pmid, term, prob) :- PROB(pmid, term) :- term_in_study_tfidf(pmid, term, tfidf) & (tfidf > 0.01)", failed: false, seconds: 3.2, cpu: 2.9, tokens_in: 0, tokens_out: 0 },
      { id: 2, created: day(0.2), user: "bob", engine: "neurosynth", text: "ans(x) :- p(x", failed: true, seconds: 0.1, cpu: 0.1 }],
  };
  const origFetch = window.fetch.bind(window);
  window.fetch = (url, options) => {
    const path = String(url).split("?")[0];
    if (path in ANSWERS) return Promise.resolve({ ok: true, status: 200, json: async () => ANSWERS[path] });
    if (String(url).startsWith("file:") || String(url).startsWith("/")) return Promise.resolve({ ok: false, status: 404, statusText: "not mocked", json: async () => ({ error: "not mocked" }) });
    return origFetch(url, options);
  };
})();
