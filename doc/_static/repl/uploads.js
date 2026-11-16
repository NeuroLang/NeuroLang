// The files the page uploaded in the session (Upload data), as each reply
// of the REPL carries them (web.Reply.uploads): what the page lists of them.

// A size in bytes, as a text.
export function bytes(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1 << 20) return `${(size / 1024).toFixed(1)} KB`;
  const megabytes = size / (1 << 20);
  return `${Number.isInteger(megabytes) ? megabytes : megabytes.toFixed(1)} MB`;
}

// The line over the list: how many files, and the room they take; empty
// without any, unless the engine's tractogram (listed with them, never
// removed) is not the one indexed, to be indexed again.
export function uploadsSummary(uploads) {
  const files = uploads?.files ?? [];
  const uploaded = files.filter((file) => !file.engine);
  if (!uploaded.length && !files.some((file) => file.engine && file.index)) return "";
  return `Uploaded files (${uploaded.length}): ${bytes(uploads.used)} of ${bytes(uploads.most)} used`;
}

// A row of the list for each file: its name, kind, size, and the relations
// loaded from it; the engine's tractogram has no size, it is the engine's.
export function uploadRows(uploads) {
  return (uploads?.files ?? []).map((file) => [
    file.name,
    file.engine || /\.(trk|tck)$/i.test(file.name) ? "tractogram"
      : /\.nii(\.gz)?$/i.test(file.name) ? "image" : "table",
    file.engine ? "the engine's" : bytes(file.size),
    file.loaded.length ? file.loaded.join(", ") : "not loaded",
  ]);
}

// The files of *names*, those the page keeps to upload again, that the
// session no longer has (.uploads remove): none without the uploads of a
// reply, as an older version gives.
export function removedUploads(names, uploads) {
  if (!uploads?.files) return [];
  const kept = new Set(uploads.files.map((file) => file.name));
  return names.filter((name) => !kept.has(name));
}
