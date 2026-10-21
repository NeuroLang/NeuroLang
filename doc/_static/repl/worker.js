// The Web Worker running NeuroLang's interactive query REPL in Pyodide:
//
//     neurolang-query -i --engine neurosynth --resolution 2
//
// It loads Pyodide, gets NeuroLang, installs the Neurosynth data it ships,
// builds the engine, and answers the page's submissions with the HTML
// that the REPL's output renders to (neurolang.utils.tui.web.WebRepl).
//
// NeuroLang is, by default, the wheels of its packages built with the
// documentation (see doc/make_repl_bundle.py). When the page is asked for a
// `ref` -- a branch, a tag or a commit, of `repo` -- the worker runs the
// NeuroLang of that ref instead: it reads the commit's files from GitHub and
// puts them on Python's path, as it is pure Python.

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
const ENGINE = "neurosynth";
const RESOLUTION = 2;

// Packages of Pyodide's distribution, and those installed from PyPI. The
// wheel is installed without its dependencies, which are these: pysdd has
// no WebAssembly build and is optional, and nilearn's own dependencies are
// the ones of Pyodide's distribution.
const PYODIDE_PACKAGES = [
  "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "lxml",
  "requests", "joblib", "pyyaml", "packaging", "micropip",
];
const PYPI_PACKAGES = [
  "lark", "typing_inspect", "rdflib", "nibabel", "rich",
  "prompt_toolkit", "httpx",
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

async function start({ repo, ref }) {
  status("Loading Python…");
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  status("Loading numpy, scipy, pandas and scikit-learn…");
  await pyodide.loadPackage(PYODIDE_PACKAGES);

  const bundle = await (await fetch("bundle.json")).json();
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

  const data = await download(bundle.data, "the Neurosynth data");
  pyodide.FS.mkdirTree("/data");
  pyodide.unpackArchive(data, "zip", { extractDir: "/data" });

  status(`Building the ${ENGINE} engine at ${RESOLUTION} mm…`);
  await pyodide.runPythonAsync(`
import json
from pathlib import Path
from neurolang.utils import engine_registry
from neurolang.utils.tui.web import WebRepl

nl = engine_registry.build_engine("${ENGINE}", Path("/data"), ${RESOLUTION}.0)
repl = WebRepl(nl, "${ENGINE}", ${RESOLUTION})

def submit(text):
    reply = repl.submit(text)
    return json.dumps(
        {"html": reply.html, "viewers": reply.viewers,
         "failed": reply.failed, "tables": reply.tables,
         "prompt": reply.prompt}
    )

# The providers of the AI assistant, for the page's form: none when this
# version of NeuroLang predates it.
try:
    from neurolang.utils.llm.browser import providers_for_form
except ImportError:
    providers_for_form = list

def llm_providers():
    return json.dumps(providers_for_form())
`);
  const banner = pyodide.runPython("repl.banner()");
  const providers = JSON.parse(pyodide.runPython("llm_providers()"));
  postMessage({ type: "ready", banner, source, providers });
  repl = pyodide.globals.get("repl");
  return pyodide.globals.get("submit");
}

// The page starts the worker with the version to run, then submits.
let submit = null;
let repl = null;

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
  try {
    const reply = JSON.parse(submit(data.text));
    postMessage({ type: "reply", id: data.id, ...reply });
  } catch (error) {
    const text = String(error).replace(/&/g, "&amp;").replace(/</g, "&lt;");
    postMessage({ type: "reply", id: data.id, failed: true, viewers: [], tables: [], html: `<pre>${text}</pre>` });
  }
};
