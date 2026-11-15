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
// without any.
export function uploadsSummary(uploads) {
  const files = uploads?.files ?? [];
  if (!files.length) return "";
  return `Uploaded files: ${files.length}, ${bytes(uploads.used)} of ${bytes(uploads.most)}`;
}

// A row of the list for each file: its name, kind, size, and the relations
// loaded from it.
export function uploadRows(uploads) {
  return (uploads?.files ?? []).map((file) => [
    file.name,
    /\.nii(\.gz)?$/i.test(file.name) ? "image" : "table",
    bytes(file.size),
    file.loaded.length ? file.loaded.join(", ") : "not loaded",
  ]);
}
