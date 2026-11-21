// Writes, in the directory given (default /tmp/repl_states), one page per
// state: the REPL page with mock.js in its head and a <base> that points at
// the real files, and wrapper.html, which shows a state in a 390 px frame and
// writes measures of it. See README.md.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repl = resolve(here, "../../_static/repl");
const out = resolve(process.argv[2] ?? "/tmp/repl_states");
mkdirSync(out, { recursive: true });
const html = readFileSync(join(repl, "index.html"), "utf8").replace(
  "<head>",
  `<head>\n<base href="${pathToFileURL(repl)}/">\n<script src="${pathToFileURL(join(here, "mock.js"))}"></script>`,
);
writeFileSync(join(out, "index.html"), html);
writeFileSync(join(out, "wrapper.html"), `<!doctype html><meta charset=utf-8><body style="margin:0;background:#888">
<iframe id=f style="width:390px;height:1700px;border:0;display:block"></iframe><pre id=m style="display:none"></pre>
<script>
const q = new URLSearchParams(location.search);
const f = document.getElementById("f");
f.src = q.has("page") ? q.get("page") + ".html" + (q.has("v") ? "?v=" + q.get("v") : "") : "index.html?state=" + q.get("state");
// After the page is ready: horizontal overflow of the page, and the small controls.
f.onload = () => setTimeout(() => {
  const d = f.contentDocument, w = f.contentWindow, out = [];
  out.push("scrollWidth " + d.documentElement.scrollWidth + " of " + w.innerWidth);
  const closed = (e) => { const d = e.parentElement?.closest("details:not([open])"); return d && !(e.tagName === "SUMMARY" && e.parentElement === d) || (d && e.tagName === "SUMMARY" && e.parentElement !== d); };
  const wide = [...d.querySelectorAll("body *")].filter((e) => { const r = e.getBoundingClientRect(); return r.right > w.innerWidth + 1 && r.width > 0 && !closed(e) && !e.closest(".rt-scroll,.agent-log,#banner,#out,.guide-list,.agent-query,pre,.asked-list"); });
  out.push("overflowing: " + wide.slice(0, 12).map((e) => e.tagName + "#" + e.id + "." + e.className).join(" | "));
  // Controls under 44 px, but those in a closed <details> (only its summary shows) and the rows of the guide lists.
  const small = [...d.querySelectorAll("button,summary,select,input:not([type=hidden]),a")].filter((e) => { const r = e.getBoundingClientRect(); return r.width > 1 && r.height > 1 && (r.height < 44 || r.width < 44) && !e.closest("[hidden]") && !closed(e) && getComputedStyle(e).visibility !== "hidden"; });
  out.push("under 44px: " + small.map((e) => { const r = e.getBoundingClientRect(); return e.tagName + "#" + e.id + "." + String(e.className).slice(0, 20) + " '" + e.textContent.trim().slice(0, 14) + "' " + Math.round(r.width) + "x" + Math.round(r.height); }).join(" | "));
  if (q.has("eval")) out.push(String(new w.Function(q.get("eval"))()));
  document.getElementById("m").textContent = out.join("\\n");
  document.getElementById("m").style.display = q.has("measure") || q.has("eval") ? "block" : "none";
}, +q.get("delay") || 2500);
</script>`);
// The pages of tools/serve_repl: server_login.html, server_account.html...
const serve = resolve(here, "../../../tools/serve_repl");
const vendor = pathToFileURL(join(serve, "vendor/chartjs/chart.umd.min.js"));
for (const page of ["login", "account", "admin", "activity"]) {
  writeFileSync(join(out, `server_${page}.html`), readFileSync(join(serve, `${page}.html`), "utf8")
    .replace("<head>", `<head>\n<script src="${pathToFileURL(join(here, "server_mock.js"))}"></script>`)
    .replace("/admin/vendor/chart.umd.min.js", String(vendor)));
}
console.log(`written to ${out}`);
