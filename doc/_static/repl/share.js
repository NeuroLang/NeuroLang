// A query shared as a link: its text and mode are in the address fragment,
// which a browser keeps to itself: no server of the page's host sees the
// query.

const bytesToBase64Url = (bytes) => {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
};

const base64UrlToBytes = (text) => {
  const binary = atob(text.replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
};

// The fragment (without its #) carrying *code* in *mode*.
export function encodeShare(code, mode = "datalog") {
  const params = new URLSearchParams({ q: bytesToBase64Url(new TextEncoder().encode(code)) });
  if (mode !== "datalog") params.set("mode", mode);
  return params.toString();
}

// The {code, mode} a fragment (with or without its #) carries, or null.
export function decodeShare(fragment) {
  const params = new URLSearchParams(String(fragment ?? "").replace(/^#/, ""));
  const encoded = params.get("q");
  if (!encoded) return null;
  try {
    const code = new TextDecoder("utf-8", { fatal: true }).decode(base64UrlToBytes(encoded));
    return { code, mode: params.get("mode") === "squall" ? "squall" : "datalog" };
  } catch {
    return null; // not a query of ours
  }
}
