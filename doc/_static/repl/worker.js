// The Web Worker running NeuroLang's interactive query REPL in Pyodide:
//
//     neurolang-query -i --engine neurosynth --resolution 2
//
// It loads Pyodide, gets NeuroLang, installs the data of the engine the page
// asks for (neurosynth, neurosynth_iit, neurostore or neurostore_iit, see
// engines.js) that the bundle ships,
// builds the engine, and answers the page's submissions with the HTML
// that the REPL's output renders to (neurolang_apps.utils.tui.web.WebRepl).
//
// NeuroLang is, by default, the wheels of its packages built with the
// documentation (see doc/make_repl_bundle.py). When the page is asked for a
// `ref` -- a branch, a tag or a commit, of `repo` -- the worker runs the
// NeuroLang of that ref instead: it reads the commit's files from GitHub and
// puts them on Python's path, as it is pure Python.

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";
import { bundleEngines, dataParts, defaultEngine, joinParts } from "./engines.js";

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
const RESOLUTION = 2;

// Packages of Pyodide's distribution, and those installed from PyPI. The
// wheel is installed without its dependencies, which are these: pysdd has
// no WebAssembly build and is optional, and nilearn's own dependencies are
// the ones of Pyodide's distribution.
const PYODIDE_PACKAGES = [
  "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "lxml",
  "requests", "joblib", "pyyaml", "packaging", "micropip",
  // The tables of a NeuroStore release are Parquet.
  "pyarrow",
];
const PYPI_PACKAGES = [
  "lark", "typing_inspect", "rdflib", "nibabel", "rich",
  "prompt_toolkit", "httpx", "omegaconf",
];

const status = (message) => postMessage({ type: "status", message });

async function download(url, what) {
  status(`Downloading ${what}…`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not download ${what}: ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

const GITHUB_API = "https://api.github.com/repos";
const GITHUB_RAW = "https://raw.githubusercontent.com";
const SOURCE_DIR = "/src";
const CONCURRENCY = 16;

// The packages of NeuroLang are in packages/NAME/NAME; before they were
// split, NeuroLang was the package neurolang at the root.
const PACKAGE_SOURCE = /^packages\/([^/]+)\/(?=\1\/)/;

// Where `path`, a file of the repository, goes on Python's path.
const sourcePath = (path) => path.replace(PACKAGE_SOURCE, "");

// The files of the NeuroLang packages that a ref needs to run: not their
// tests nor the legacy web application.
const isSource = (path) =>
  (PACKAGE_SOURCE.test(path) || path.startsWith("neurolang/")) &&
  !/(^|\/)tests\//.test(path) &&
  !sourcePath(path).startsWith("neurolang/utils/server/");

// The module of the REPL in the browser, under one layout or the other.
const WEB_REPL = [
  "neurolang/utils/tui/web.py", "neurolang_apps/utils/tui/web.py",
];

async function github(url, what) {
  const response = await fetch(url);
  // GitHub answers 422 for a ref it does not know.
  if (response.status === 404 || response.status === 422) {
    throw new Error(`${what} was not found on GitHub.`);
  }
  if (response.status === 403 || response.status === 429) {
    throw new Error("GitHub's rate limit for this address is reached: try again later.");
  }
  if (!response.ok) throw new Error(`Could not read ${what}: ${response.status}`);
  return response;
}

async function mapConcurrently(items, task) {
  let next = 0;
  const workers = Array.from({ length: CONCURRENCY }, async () => {
    while (next < items.length) await task(items[next++]);
  });
  await Promise.all(workers);
}

// Put the NeuroLang of `repo` at `ref` in the file system; return its commit.
async function installSource(pyodide, repo, ref) {
  status(`Looking up ${repo} at ${ref}…`);
  const commit = await (await github(
    `${GITHUB_API}/${repo}/commits/${encodeURI(ref)}`, `${repo} at ${ref}`,
  )).json();
  const tree = await (await github(
    `${GITHUB_API}/${repo}/git/trees/${commit.sha}?recursive=1`, `the files of ${commit.sha}`,
  )).json();
  const files = tree.tree.filter((entry) => entry.type === "blob" && isSource(entry.path));
  if (!files.some((entry) => WEB_REPL.includes(sourcePath(entry.path)))) {
    throw new Error(
      `${repo} at ${ref} has no neurolang/utils/tui/web.py: it predates the REPL ` +
      "in the browser, which needs a ref that has it.",
    );
  }
  let done = 0;
  await mapConcurrently(files, async ({ path }) => {
    const response = await github(`${GITHUB_RAW}/${repo}/${commit.sha}/${path}`, path);
    const target = `${SOURCE_DIR}/${sourcePath(path)}`;
    pyodide.FS.mkdirTree(target.slice(0, target.lastIndexOf("/")));
    pyodide.FS.writeFile(target, new Uint8Array(await response.arrayBuffer()));
    status(`Downloading NeuroLang from ${repo}@${ref}: ${++done}/${files.length} files…`);
  });
  pyodide.runPython(`import sys; sys.path.insert(0, "${SOURCE_DIR}")`);
  return commit.sha;
}

async function start({ repo, ref, full, engine }) {
  status("Loading Python…");
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  status("Loading numpy, scipy, pandas, scikit-learn and pyarrow…");
  await pyodide.loadPackage(PYODIDE_PACKAGES);

  const bundle = await (await fetch("bundle.json", { cache: "no-store" })).json();
  const engines = bundleEngines(bundle);
  const name = engines[engine] ? engine : defaultEngine(bundle);
  const entry = engines[name];
  const micropip = pyodide.pyimport("micropip");
  status("Installing the packages NeuroLang needs…");
  for (const name of PYPI_PACKAGES) await micropip.install(name);
  await micropip.install.callKwargs("nilearn", { deps: false });

  let source;
  if (ref) {
    source = `${repo}@${ref} (${(await installSource(pyodide, repo, ref)).slice(0, 8)})`;
  } else {
    for (const wheel of bundle.wheels) {
      pyodide.FS.writeFile(`/tmp/${wheel}`, await download(wheel, wheel.split("-")[0]));
      await micropip.install.callKwargs(`emfs:/tmp/${wheel}`, { deps: false });
    }
    source = `the wheels of this page (${bundle.wheels.join(", ")})`;
  }

  const parts = dataParts(entry);
  const chunks = [];
  for (const [number, url] of parts.entries()) {
    const of = parts.length > 1 ? ` (${number + 1}/${parts.length})` : "";
    chunks.push(await download(url, `the ${entry.label} data${of}`));
  }
  const data = joinParts(chunks);
  pyodide.FS.mkdirTree("/data");
  pyodide.unpackArchive(data, "zip", { extractDir: "/data" });

  status("Importing NeuroLang…");
  await pyodide.runPythonAsync(`
import inspect
import json
import os
from pathlib import Path

import js

def say(message):
    # To the page while this worker is busy: postMessage does not wait for it.
    js.postMessage(js.JSON.parse(json.dumps({"type": "status", "message": message + "…"})))

try:
    from neurolang_apps.utils import engine_registry
    from neurolang_apps.utils.tui import web
except ImportError:  # a ref from before the split into packages
    from neurolang.utils import engine_registry
    from neurolang.utils.tui import web
WebRepl = web.WebRepl

# The precomputed analyses (lift, ratios...) fill gigabytes, more than
# a tab is given: they are left out unless the page asks for them.
yaml_path = None
if not ${full ? "True" : "False"}:
    try:
        yaml_path = web.lightweight_engines_yaml("/tmp")
    except AttributeError:  # a version that predates it
        pass
# The steps of the build are told as they start, by the versions that can.
tells = {}
if "progress" in inspect.signature(engine_registry.build_engine).parameters:
    tells = {"progress": say}
# The files an engine reads by a relative path (the atlas of tractography)
# are in its data directory.
os.chdir("/data")
nl = engine_registry.build_engine(
    "${name}", Path("/data"), ${RESOLUTION}.0, yaml_path=yaml_path, **tells
)
say("Starting the REPL")
# The metadata of the studies (their DOIs, titles...) let the page link the
# studies of a result; a version that predates them does not take them. They
# are those of the studies of Neurosynth, whose ids are PubMed's.
options = {}
if "${name}".startswith("neurosynth") and "study_metadata" in inspect.signature(WebRepl).parameters:
    try:
        from neurolang.frontend.neurosynth_utils import fetch_study_metadata
        options["study_metadata"] = fetch_study_metadata(Path("/data"), verbose=0)
    except Exception:
        pass
repl = WebRepl(nl, "${name}", ${RESOLUTION}, **options)

# The steps of the AI assistant go to the page as they happen, while this
# worker is busy with the question: postMessage does not wait for it.
if hasattr(repl, "set_event_sink"):
    repl.set_event_sink(lambda text: js.postMessage(js.JSON.parse(text)))

# The files of the text embeddings (the SPECTER2 query model and the
# embeddings of the studies) are fetched when a query first needs them.
FILES = json.loads(${JSON.stringify(JSON.stringify(entry.files ?? {}))})

def fetch_missing(path):
    import js
    from pyodide.ffi import to_js
    try:
        parts = FILES[str(path.relative_to("/data"))]
    except (KeyError, ValueError):
        return
    what = (
        "the SPECTER2 query model" if "/models/" in str(path)
        else "the embeddings of the studies"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    with open(partial, "wb") as file:
        for number, part in enumerate(parts, 1):
            of = f" ({number}/{len(parts)})" if len(parts) > 1 else ""
            js.postMessage(to_js(
                {"type": "status", "message": f"Downloading {what}{of}…"},
                dict_converter=js.Object.fromEntries,
            ))
            # Synchronously, as the query waits for it: a worker may.
            request = js.XMLHttpRequest.new()
            request.open("GET", part, False)
            request.responseType = "arraybuffer"
            request.send()
            if request.status != 200:
                return
            file.write(request.response.to_py())
    partial.rename(path)

try:
    from neurolang.utils import text_embeddings
    text_embeddings.fetch_missing = fetch_missing
except ImportError:  # a version without text embeddings
    pass

def submit(text):
    reply = repl.submit(text)
    return json.dumps(
        {"html": reply.html, "viewers": reply.viewers,
         "failed": reply.failed, "tables": reply.tables,
         "prompt": reply.prompt, "mode": getattr(reply, "mode", ""),
         "predicates": getattr(reply, "predicates", []),
         "ratable": getattr(reply, "ratable", False),
         "graph": getattr(reply, "graph", "")}
    )

# The providers of the AI assistant, for the page's form: none when this
# version of NeuroLang predates it.
try:
    try:
        from neurolang_apps.utils.llm.browser import providers_for_form
    except ImportError:
        from neurolang.utils.llm.browser import providers_for_form
except ImportError:
    providers_for_form = list

def download(block, kind, value):
    file = repl.download(block, kind, value)
    return file.name, file.mime, file.data

def llm_providers():
    return json.dumps(providers_for_form())

def rate(verdict, comment):
    return json.dumps(repl.rate_answer(verdict, comment))
`);
  const banner = pyodide.runPython("repl.banner()");
  const providers = JSON.parse(pyodide.runPython("llm_providers()"));
  const examples = JSON.parse(pyodide.runPython("json.dumps(repl.examples() if hasattr(repl, 'examples') else {})"));
  const questions = JSON.parse(pyodide.runPython("json.dumps(repl.questions() if hasattr(repl, 'questions') else [])"));
  const predicates = JSON.parse(pyodide.runPython("json.dumps(repl.predicates() if hasattr(repl, 'predicates') else [])"));
  const commands = JSON.parse(pyodide.runPython("json.dumps(repl.commands() if hasattr(repl, 'commands') else [])"));
  const assistant = pyodide.runPython("repl.assistant_ready() if hasattr(repl, 'assistant_ready') else False");
  // Where a rated answer goes (feedback.json next to the page); without a
  // document, or in a version without ratings, answers cannot be rated.
  let canRate = false;
  try {
    const answer = await fetch("feedback.json", { cache: "no-store" });
    const { url = "", doc = "" } = answer.ok ? await answer.json() : {};
    pyodide.globals.set("feedback_target", pyodide.toPy([url, doc]));
    canRate = pyodide.runPython("(repl.set_feedback(*feedback_target), repl.can_rate())[1] if hasattr(repl, 'set_feedback') else False");
  } catch {
    // the page works without ratings
  }
  postMessage({ type: "ready", banner, source, providers, examples, questions, predicates, commands, canRate, assistant });
  repl = pyodide.globals.get("repl");
  makeFile = pyodide.globals.get("download");
  rateAnswer = pyodide.globals.get("rate");
  return pyodide.globals.get("submit");
}

// The page starts the worker with the version to run, then submits.
let submit = null;
let repl = null;
let makeFile = null; // the file of a result, for a download
let rateAnswer = null; // shares the last answer of the assistant with its rating

// The visitor's key is given to the REPL, in this worker's memory: from
// here it goes to the provider, and to no one else.
function useKey({ provider, key, model, baseUrl }) {
  if (!key) {
    repl.clear_llm();
  } else {
    repl.set_llm(provider, key, model ?? "", baseUrl ?? "");
  }
}

onmessage = async ({ data }) => {
  if (data.type === "start") {
    try {
      submit = await start(data);
    } catch (error) {
      postMessage({ type: "error", message: String(error.message ?? error) });
    }
    return;
  }
  if (!submit) return;
  if (data.type === "llm") {
    try {
      useKey(data);
      postMessage({ type: "llm", ok: true });
    } catch (error) {
      // A ValueError of Python says what is wrong with the choices.
      const message = String(error.message ?? error).trim().split("\n").at(-1);
      postMessage({ type: "llm", ok: false, message: message.replace(/^\w+Error: /, "") });
    }
    return;
  }
  if (data.type === "rate") {
    try {
      postMessage({ type: "rated", ...JSON.parse(rateAnswer(data.verdict, data.comment ?? "")) });
    } catch (error) {
      const message = String(error.message ?? error).trim().split("\n").at(-1);
      postMessage({ type: "rated", ok: false, message });
    }
    return;
  }
  if (data.type === "download") {
    try {
      const [name, mime, bytes] = makeFile(data.block, data.kind, data.value ?? "").toJs();
      postMessage({ type: "file", name, mime, bytes }, [bytes.buffer]);
    } catch (error) {
      const message = String(error.message ?? error).trim().split("\n").at(-1);
      postMessage({ type: "file", error: message.replace(/^[\w.]+Error: /, "") });
    }
    return;
  }
  try {
    const reply = JSON.parse(submit(data.text));
    postMessage({ type: "reply", id: data.id, ...reply });
  } catch (error) {
    const text = String(error).replace(/&/g, "&amp;").replace(/</g, "&lt;");
    postMessage({ type: "reply", id: data.id, failed: true, viewers: [], tables: [], html: `<pre>${text}</pre>` });
  }
};
