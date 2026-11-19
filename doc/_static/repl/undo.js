// Undo and redo of a text box, kept by the page. The browser's own history of
// a box loses what the page writes in it (an example, the History menu, the
// assistant's query, Format) and, in Chrome, undoes a character at a time
// when the page changes as one types, as the colours of the code box do.
// This one keeps the states of the box, its text and selection: typing of one
// kind is one step until a pause, and what the page wrote is a step.
import { SETTINGS } from "./settings.js";

const { pause_ms: PAUSE_MS, steps: STEPS } = SETTINGS.undo;

// The kind of an edit whose steps join while it goes on, or null for an edit
// that is a step of its own (a paste, a cut, a drop, a write of the page).
export function editKind(inputType = "") {
  if (inputType === "insertText" || inputType === "insertLineBreak") return "type";
  if (inputType.startsWith("delete") && inputType !== "deleteByCut") return "delete";
  return null;
}

// The states of a box, of which one is shown.
export class TextHistory {
  constructor(state, steps = STEPS) {
    this.states = [state];
    this.index = 0;
    this.steps = steps;
    this.last = null; // the kind and time of the last edit, while its step may grow
  }

  get current() {
    return this.states[this.index];
  }

  // Keep *state*, the box after an edit of *kind* at *time* (ms): it joins
  // the step of the edit before it when both are of a kind, with a pause
  // shorter than PAUSE_MS. The steps undone are forgotten.
  record(state, kind = null, time = 0) {
    if (state.text === this.current.text) return;
    const joins = kind !== null && this.last?.kind === kind && time - this.last.time < PAUSE_MS;
    this.states.length = this.index + 1;
    if (joins) {
      this.states[this.index] = state;
    } else {
      this.states.push(state);
      if (this.states.length > this.steps) this.states.shift();
      this.index = this.states.length - 1;
    }
    this.last = kind === null ? null : { kind, time };
  }

  // The state before the current one, or after it; null when there is none.
  undo() {
    this.last = null;
    return this.index > 0 ? this.states[--this.index] : null;
  }

  redo() {
    this.last = null;
    return this.index < this.states.length - 1 ? this.states[++this.index] : null;
  }
}

// Keep the history of the textarea *box*: Ctrl or Cmd Z undoes, with Shift
// redoes, as does Ctrl Y; so do the Edit menu's Undo and Redo.
export function keepHistory(box) {
  const state = () => ({ text: box.value, start: box.selectionStart, end: box.selectionEnd });
  const history = new TextHistory(state());
  // What the page wrote since the last edit is seen before the next one.
  const catchUp = () => history.record(state());
  const move = (back) => {
    catchUp();
    const to = back ? history.undo() : history.redo();
    if (!to) return;
    box.value = to.text;
    box.setSelectionRange(to.start, to.end);
    box.dispatchEvent(new Event("input")); // the colours of the code box follow
  };
  box.addEventListener("keydown", (event) => {
    if (!(event.ctrlKey || event.metaKey) || event.altKey || event.isComposing) return;
    const key = event.key.toLowerCase();
    const redo = (key === "z" && event.shiftKey) || (key === "y" && !event.shiftKey);
    if (key !== "z" && !redo) return;
    event.preventDefault();
    move(!redo);
  });
  box.addEventListener("beforeinput", (event) => {
    if (event.inputType === "historyUndo" || event.inputType === "historyRedo") {
      event.preventDefault();
      move(event.inputType === "historyUndo");
    } else {
      catchUp();
    }
  });
  box.addEventListener("input", (event) => {
    // The page's own input events (above, the editor's) have no inputType.
    if (event.inputType && !event.isComposing) history.record(state(), editKind(event.inputType), event.timeStamp);
  });
  box.addEventListener("compositionend", catchUp);
  return history;
}
