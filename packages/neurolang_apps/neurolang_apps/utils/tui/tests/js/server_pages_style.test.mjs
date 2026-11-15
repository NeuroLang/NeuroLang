import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// The pages of tools/serve_repl look alike: login, account and admin carry the
// REPL's tokens (doc/_static/repl/tokens.css), activity its own palette. What
// a later edit of one of them could undo unseen: its tokens, its contrast,
// its focus, its touch targets, its headings and table heads.
const repl = new URL("../../../../../../../doc/_static/repl/", import.meta.url);
const serve = new URL("../../../../../../../tools/serve_repl/", import.meta.url);
const block = (css, selector) =>
  css.match(new RegExp(`${selector.replace(/[[\]]/g, "\\$&")} \\{([^}]*)\\}`))[1];
const tokensOf = (css, selector) =>
  Object.fromEntries([...block(css, selector).matchAll(/--([\w-]+):\s*(#[0-9a-f]{6})/gi)].map(([, name, hex]) => [name, hex]));
const tokenCss = readFileSync(new URL("tokens.css", repl), "utf8");
const themes = { dark: ":root", light: ":root[data-theme=light]" };

const luminance = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const contrast = (a, b) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};

const pages = Object.fromEntries(["login", "account", "admin", "activity"]
  .map((name) => [name, readFileSync(new URL(`${name}.html`, serve), "utf8")]));

for (const [name, html] of Object.entries(pages)) {
  test(`${name}.html has one h1`, () => {
    assert.equal(html.match(/<h1[\s>]/g).length, 1);
  });
  test(`${name}.html never takes the focus outline away without a replacement`, () => {
    assert.doesNotMatch(html, /outline:\s*none/);
  });
}

for (const name of ["login", "account", "admin"]) {
  const html = pages[name];
  for (const [scheme, selector] of Object.entries(themes)) {
    const own = tokensOf(html, selector);
    const ref = tokensOf(tokenCss, selector);

    test(`${name}.html: its ${scheme} tokens equal those of tokens.css`, () => {
      for (const [token, value] of Object.entries(own)) {
        if (token in ref) assert.equal(value, ref[token], `--${token}`);
      }
      for (const token of ["control", "faint", "ground", "panel", "sunken", "text", "accent"]) {
        assert.ok(token in own, `--${token}`);
      }
    });

    test(`${name}.html: ${scheme}: the text is read at 4.5:1 on its grounds`, () => {
      for (const ink of ["text", "muted", "accent", "bad", "ok"]) {
        for (const ground of ["ground", "panel", "sunken"]) {
          assert.ok(contrast(own[ink], own[ground]) >= 4.5, `--${ink} on --${ground}`);
        }
      }
      for (const [ink, ground] of [["on-accent", "accent"], ["text", "accent-soft"], ["text", "bad-soft"]]) {
        assert.ok(contrast(own[ink], own[ground]) >= 4.5, `--${ink} on --${ground}`);
      }
    });

    test(`${name}.html: ${scheme}: the border of a field is read at 3:1 on the ground and the card`, () => {
      for (const ground of ["ground", "panel", "sunken"]) {
        assert.ok(contrast(own.control, own[ground]) >= 3, `--control on --${ground}`);
      }
    });
  }
}

test("the fields of the login, account and admin pages draw their border with --control", () => {
  assert.match(pages.login, /input, textarea \{[^}]*border: 1px solid var\(--control\)/);
  for (const name of ["account", "admin"]) {
    assert.match(pages[name], /input:not\(\[type=checkbox\]\), select \{[^}]*border: 1px solid var\(--control\)/, name);
  }
});

test("the fields of the activity page draw their border at 3:1", () => {
  const html = pages.activity;
  assert.match(html, /select, input\[type="search"\] \{[^}]*border: 1px solid var\(--line-ctl\)/);
  for (const selector of [":root", ":root\\[data-theme=\"dark\"\\]"]) {
    const own = Object.fromEntries([...html.match(new RegExp(`${selector} \\{([^}]*)\\}`))[1]
      .matchAll(/--([\w-]+):\s*(#[0-9a-f]{6})/gi)].map(([, token, hex]) => [token, hex]));
    for (const ground of ["bg", "surface"]) assert.ok(contrast(own["line-ctl"], own[ground]) >= 3, `${selector} --line-ctl on --${ground}`);
    for (const ink of ["ink", "muted", "accent"]) {
      for (const ground of ["bg", "surface", "tint"]) assert.ok(contrast(own[ink], own[ground]) >= 4.5, `${selector} --${ink} on --${ground}`);
    }
  }
});

for (const name of ["login", "account", "admin", "activity"]) {
  test(`${name}.html: its controls are 44px under a coarse pointer`, () => {
    if (name === "login") return assert.match(pages.login, /min-height: 2\.75rem/);
    if (name === "activity") return assert.match(pages.activity, /\(pointer: coarse\) \{[^}]*min-height: 44px/);
    assert.match(pages[name], /@media \(pointer: coarse\) \{[^}]*min-height: 44px/);
  });
}

test("the account and admin pages take the theme of the REPL before the first paint", () => {
  for (const name of ["account", "admin"]) {
    assert.match(pages[name], /localStorage\.getItem\("neurolang-repl-theme"\)/, name);
    assert.match(pages[name], /dataset\.theme = light \? "light" : "dark"/, name);
  }
});

test("the activity page keeps its theme under the REPL's key, with the REPL's values", () => {
  const html = pages.activity;
  assert.match(html, /localStorage\.getItem\("neurolang-repl-theme"\)/);
  assert.match(html, /localStorage\.setItem\("neurolang-repl-theme", choice \|\| "system"\)/);
  assert.match(html, /event\.key === "neurolang-repl-theme"/);
  assert.doesNotMatch(html, /(get|set|remove)Item\("theme"/);
});

test("the focus of the account and admin pages is a ring in the accent colour, visible in forced colours", () => {
  for (const name of ["account", "admin"]) {
    assert.match(pages[name], /:focus-visible \{\s*outline: 2px solid transparent; box-shadow: [^;]*var\(--accent\)/, name);
  }
});

test("the status lines of account and admin say it in a mark as well as a colour", () => {
  for (const name of ["account", "admin"]) {
    assert.match(pages[name], /\[role=status\]\.error::before \{ content: "\\2715/, name);
    assert.match(pages[name], /\[role=status\]\.ok::before \{ content: "\\2713/, name);
  }
});

test("the tables of the admin page have column heads with a scope, a label for each empty one, and a scroll wrapper", () => {
  const html = pages.admin;
  const heads = [...html.matchAll(/<th\b([^>]*)>(.*?)<\/th>/g)];
  assert.ok(heads.length > 20);
  for (const [, attributes, text] of heads) {
    assert.match(attributes, /scope="col"/);
    assert.ok(text.replace(/<[^>]*>/g, "").trim() || /sr-only/.test(text), "an empty head without a label");
  }
  assert.equal(html.match(/<table>/g).length, html.match(/<div class="scroll"><table>/g).length);
  assert.match(html, /\.scroll \{ overflow-x: auto; position: relative; \}/);
});

test("the nav of account and admin keeps the links the server pages and the tests find", () => {
  assert.match(pages.account, /<a href="\/logout">Log out<\/a>/);
  assert.match(pages.admin, /<a href="\/admin\/activity">Activity<\/a>/);
});
