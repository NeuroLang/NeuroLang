// The pages shown in frames of the results (the brain viewer, the graph of a
// query) are sandboxed: they cannot read the theme of this page, nor can this
// page read how tall they are. They are told the theme, in their html tag when
// they are made and in a message when it changes, and say their height in a
// message (see static/frame_hooks.js), which the frame is made as tall as.

const THEMES = ["light", "dark"];
const MOST = 20000; // the tallest a page may ask its frame to be, in pixels

const theme = (name) => (THEMES.includes(name) ? name : "dark");

// The HTML *page* with the theme *name* on its html tag.
export function themed(page, name) {
  return page.replace(/<html(\s[^>]*)?>/i, (tag, attributes = "") => {
    const others = attributes.replace(/\sdata-theme="[^"]*"/, "");
    return `<html${others} data-theme="${theme(name)}">`;
  });
}

// What a frame is sent when the theme changes.
export const themeMessage = (name) => ({ type: "neurolang-theme", theme: theme(name) });

// The frame of *frames* a message *event* of its page asks to be made as tall
// as it says, and that height; null for any other message, and for a page
// with no height, that of a page in a tab that is not shown.
export function frameHeight(event, frames) {
  const data = event.data;
  if (data?.type !== "neurolang-height") return null;
  const frame = frames.find((candidate) => candidate.contentWindow === event.source);
  const height = typeof data.height === "number" ? data.height : NaN;
  if (!frame || !Number.isFinite(height) || height <= 0 || height > MOST) return null;
  return { frame, height: Math.ceil(height) };
}
