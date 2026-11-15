# Rendering the REPL page's live states without the engine

`mock.js` stands in for the Web Worker (`doc/_static/repl/worker.js`) and for
`fetch`: it replays canned messages of the worker's protocol (ready, replies
with tables, the steps of the assistant, errors), so that the page of the REPL
can be looked at as it is with a result, a menu open, the assistant at work,
or a fatal error, without Pyodide or the data (`bundle.json`). `server_mock.js`
does the same for the pages of `tools/serve_repl` (login, account, admin,
activity). The states are the `STATES` of `mock.js`: one line each, with what
the worker sends, what it answers a statement, and what the visitor does.

```sh
node doc/tools/repl_page_states/build.mjs            # writes /tmp/repl_states/*.html
doc/tools/repl_page_states/shoot.sh /tmp/shots table assistant_done server_login
```

`shoot.sh OUTDIR [state...]` takes, per state, a screenshot at 1440 px in the
dark and light themes, and at 390 px with a coarse pointer (the page in an
iframe of `wrapper.html`, because headless Chrome has a minimum window width).
`wrapper.html?state=NAME&measure=1` writes into the page the horizontal
overflow and the controls under 44 px; `&eval=JS` runs JS in the frame.
Loopback HTTP is not needed: the pages are opened from files, with
`--allow-file-access-from-files`. The headless Chrome's
`--blink-settings=preferredColorScheme=0` is the dark theme, `=1` the light one.
Screenshots are not committed.
