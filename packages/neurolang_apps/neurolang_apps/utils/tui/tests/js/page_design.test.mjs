import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// The design of the REPL's page that a later edit could undo without anyone
// seeing it: the contrast of its colours, and what a screen reader finds.
const dir = new URL("../../../../../../../doc/_static/repl/", import.meta.url);
const tokens = readFileSync(new URL("tokens.css", dir), "utf8");
const page = readFileSync(new URL("index.html", dir), "utf8");

// The custom properties of a theme: the dark one is :root, the light one :root[data-theme=light].
const theme = (selector) => {
  const block = tokens.match(new RegExp(`${selector.replace(/[[\]]/g, "\\$&")} \\{([^}]*)\\}`))[1];
  return Object.fromEntries([...block.matchAll(/--([\w-]+):\s*(#[0-9a-f]{6})/gi)].map(([, name, hex]) => [name, hex]));
};
const themes = { dark: theme(":root"), light: theme(":root[data-theme=light]") };

const luminance = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const contrast = (a, b) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};

for (const [name, colours] of Object.entries(themes)) {
  test(`${name}: the text is read at 4.5:1 on the grounds it sits on`, () => {
    for (const ink of ["text", "muted", "faint", "accent"]) {
      for (const ground of ["panel", "sunken"]) {
        assert.ok(contrast(colours[ink], colours[ground]) >= 4.5, `--${ink} on --${ground}`);
      }
    }
  });

  test(`${name}: the faint text, row numbers and times, is read at 4.5:1 on the hovered and striped grounds too`, () => {
    for (const ground of ["ground", "hover", "row-hover", "zebra"]) {
      assert.ok(contrast(colours.faint, colours[ground]) >= 4.5, `--faint on --${ground}`);
    }
  });

  test(`${name}: the border of a field is seen at 3:1, on the card and on its own fill`, () => {
    for (const ground of ["panel", "sunken"]) {
      assert.ok(contrast(colours.control, colours[ground]) >= 3, `--control on --${ground}`);
    }
  });
}

test("the fields draw their border with --control, and the focus keeps an outline for forced colours", () => {
  assert.match(page, /input, select, textarea \{[^}]*border: 1px solid var\(--control\)/);
  assert.match(page, /\.editor pre, \.editor textarea \{[^}]*border: 1px solid var\(--control\)/);
  assert.match(page, /input:focus, select:focus, textarea:focus \{ outline: 2px solid transparent;/);
});

test("the page has one h1, and the examples a heading", () => {
  assert.equal([...page.matchAll(/<h1[ >]/g)].length, 1);
  assert.match(page, /<summary><h2>Examples and relations<\/h2><\/summary>/);
});

test("the controls are 44px high on a touch screen", () => {
  const coarse = page.match(/@media \(pointer: coarse\) \{([\s\S]*?)\n  \}/)[1];
  for (const control of ["button:not(", "select", ".nav-menu > summary", ".rt-bar .rt-menu > summary", ".agent-obs > summary", "input:not([type=checkbox])", ".topbar nav a", "#key", "#run", "#theme button"]) {
    assert.ok(coarse.includes(control), control);
  }
  assert.match(coarse, /min-height: 44px/);
});

test("the shortcuts that change with the system are marked for the script", () => {
  assert.equal([...page.matchAll(/<kbd class="mod">Ctrl<\/kbd>/g)].length, 2);
});

test("the third theme is the computer's", () => {
  assert.match(page, /data-theme="system" title="Computer" aria-label="Computer"/);
});

test("a menu, open in a narrow screen, is in the flow: an absolute one from the left of the screen leaves it", () => {
  const narrow = page.match(/@media \(max-width: 40rem\) \{([\s\S]*?)\n  \}/)[1];
  assert.match(narrow, /\.nav-menu-items, \.menu-items, \.rt-menu-items \{ position: static; \}/);
});

test("the log of the assistant is one column the width of its box, so that its lines wrap in it", () => {
  assert.match(page, /\.agent-log \{[^}]*grid-template-columns: minmax\(0, 1fr\)/);
});

// The pages of tools/serve_repl: the login page has the tokens of the REPL's own, and
// the account and admin pages the colours of their status and notes read at 4.5:1.
const serve = new URL("../../../../../../../tools/serve_repl/", import.meta.url);
const login = readFileSync(new URL("login.html", serve), "utf8");
const loginTheme = (selector) => {
  const block = login.match(new RegExp(`${selector.replace(/[[\]]/g, "\\$&")} \\{([^}]*)\\}`))[1];
  return Object.fromEntries([...block.matchAll(/--([\w-]+):\s*(#[0-9a-f]{6})/gi)].map(([, name, hex]) => [name, hex]));
};

for (const [name, selector] of [["dark", ":root"], ["light", ":root[data-theme=light]"]]) {
  test(`the login page's ${name} colours are those of the REPL's tokens`, () => {
    const own = loginTheme(selector);
    for (const [token, value] of Object.entries(own)) {
      if (token in themes[name]) assert.equal(value, themes[name][token], `--${token}`);
    }
    for (const token of ["control", "faint"]) assert.ok(token in own, `--${token}`);
  });
}

test("the fields of the login page draw their border with --control", () => {
  assert.match(login, /input, textarea \{[^}]*border: 1px solid var\(--control\)/);
});

for (const file of ["account.html", "admin.html"]) {
  test(`${file}: its notes and states are read at 4.5:1 in both colour schemes`, () => {
    const html = readFileSync(new URL(file, serve), "utf8");
    for (const token of ["dim", "bad", "ok"]) {
      const [, light, dark] = html.match(new RegExp(`--${token}: light-dark\\((#[0-9a-f]{6}), (#[0-9a-f]{6})\\)`, "i"));
      assert.ok(contrast(light, "#ffffff") >= 4.5, `--${token} on white`);
      assert.ok(contrast(dark, "#121212") >= 4.5, `--${token} on the dark canvas`);
    }
    assert.match(html, /@media \(pointer: coarse\) \{[^}]*min-height: 44px/);
  });
}
