// The code box with syntax colouring: the textarea is transparent, over a
// <pre> showing the same text coloured (highlight.js). Both have the same
// font, padding and wrapping; the box grows with its text, so that they never
// scroll apart.
import { highlightNodes } from "./highlight.js";

const MAX_HEIGHT = 24 * 16; // px: beyond it the box scrolls, and the colours follow

export class Editor {
  constructor(textarea, mirror, isSquall = () => false) {
    this.textarea = textarea;
    this.mirror = mirror;
    this.isSquall = isSquall;
    textarea.addEventListener("input", () => this.refresh());
    textarea.addEventListener("scroll", () => this.follow());
    new ResizeObserver(() => this.fit()).observe(textarea);
  }

  get value() {
    return this.textarea.value;
  }

  // Put *text* in the box, as the visitor's typing would.
  set value(text) {
    this.textarea.value = text;
    this.refresh();
  }

  // Colour the text again, e.g. after a change of language, and fit the box to it.
  refresh() {
    const text = this.textarea.value;
    // A last empty line has no height in a <pre>: a space gives it one.
    this.mirror.replaceChildren(...highlightNodes(text, this.isSquall()), text.endsWith("\n") ? " " : "");
    this.fit();
  }

  fit() {
    const { textarea } = this;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight + 2, MAX_HEIGHT)}px`;
    this.follow();
  }

  follow() {
    this.mirror.scrollTop = this.textarea.scrollTop;
  }
}
