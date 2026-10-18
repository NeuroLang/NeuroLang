// The Web Worker running NeuroLang's interactive query REPL in Pyodide:
//
//     neurolang-query -i --engine neurosynth --resolution 2
//
// It loads Pyodide, installs the NeuroLang wheel (built with the
// documentation, see doc/make_repl_bundle.py) and the Neurosynth data it ships,
// builds the engine, and answers the page's submissions with the HTML
// that the REPL's output renders to (neurolang.utils.tui.web.WebRepl).

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
  "lark", "typing_inspect", "rdflib", "nibabel", "seaborn", "rich",
  "prompt_toolkit",
];

const status = (message) => postMessage({ type: "status", message });

async function download(url, what) {
  status(`Downloading ${what}…`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not download ${what}: ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function start() {
  status("Loading Python…");
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  status("Loading numpy, scipy, pandas and scikit-learn…");
  await pyodide.loadPackage(PYODIDE_PACKAGES);

  const bundle = await (await fetch("bundle.json")).json();
  const micropip = pyodide.pyimport("micropip");
  status("Installing the packages NeuroLang needs…");
  for (const name of PYPI_PACKAGES) await micropip.install(name);
  await micropip.install.callKwargs("nilearn", { deps: false });

  pyodide.FS.writeFile(`/tmp/${bundle.wheel}`, await download(bundle.wheel, "NeuroLang"));
  await micropip.install.callKwargs(`emfs:/tmp/${bundle.wheel}`, { deps: false });

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
        {"html": reply.html, "viewers": reply.viewers, "failed": reply.failed}
    )
`);
  const banner = pyodide.runPython("repl.banner()");
  const submit = pyodide.globals.get("submit");
  postMessage({ type: "ready", banner });
  return submit;
}

const starting = start().catch((error) =>
  postMessage({ type: "error", message: String(error) }),
);

onmessage = async ({ data }) => {
  const submit = await starting;
  if (!submit) return;
  try {
    const reply = JSON.parse(submit(data.text));
    postMessage({ type: "reply", id: data.id, ...reply });
  } catch (error) {
    postMessage({ type: "reply", id: data.id, failed: true, viewers: [],
                  html: `<pre>${String(error).replace(/&/g, "&amp;").replace(/</g, "&lt;")}</pre>` });
  }
};
