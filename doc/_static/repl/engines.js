// The engines the page offers, from the bundle (bundle.json), and the one the
// page's address asks for: ?engine=neurostore. `?engine=full` asks for the
// default engine with its precomputed analyses (see worker.js).

// The engines of the bundle, by name, with their label; a bundle from before
// the choice has the one engine, neurosynth.
export function bundleEngines(bundle) {
  if (bundle.engines && Object.keys(bundle.engines).length) {
    return Object.fromEntries(
      Object.entries(bundle.engines).map(([name, entry]) => [name, { ...entry, label: entry.label || name }]),
    );
  }
  return { neurosynth: { label: "Neurosynth", data: bundle.data, files: bundle.files ?? {} } };
}

export function defaultEngine(bundle) {
  const engines = bundleEngines(bundle);
  return bundle.default in engines ? bundle.default : Object.keys(engines)[0];
}

// The engine to start and whether it is with its precomputed analyses, from
// the page's address *params* (a URLSearchParams) and the *bundle*.
export function engineChoice(params, bundle) {
  const asked = params.get("engine");
  const engines = bundleEngines(bundle);
  const fallback = defaultEngine(bundle);
  if (asked === "full") return { name: fallback, full: true };
  return { name: asked in engines ? asked : fallback, full: false };
}

// *params* with the engine *name* in the address, unless it is the default.
export function withEngine(params, name, bundle) {
  const next = new URLSearchParams(params);
  if (name === defaultEngine(bundle)) next.delete("engine");
  else next.set("engine", name);
  return next;
}

// The urls of the data of an engine: one zip, or its parts, which the page
// joins (GitHub takes no file of more than 100 MB).
export function dataParts(entry) {
  return [].concat(entry.data);
}

// The bytes of *chunks* (Uint8Arrays) as one.
export function joinParts(chunks) {
  if (chunks.length === 1) return chunks[0];
  const joined = new Uint8Array(chunks.reduce((size, chunk) => size + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.length;
  }
  return joined;
}
