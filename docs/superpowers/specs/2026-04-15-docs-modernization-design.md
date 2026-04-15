# NeuroLang Documentation Modernisation — Design Spec

**Date:** 2026-04-15  
**Status:** Approved  
**Approach:** Option A — Incremental Sphinx upgrade

---

## 1. Goals

Modernise the NeuroLang documentation to match the quality and visual style of
projects like [Scallop-lang](https://www.scallop-lang.org), while ensuring
it renders correctly on GitHub Pages. Specifically:

1. Replace the unmaintained `sphinx_bootstrap_theme` (Bootstrap 3) with
   **PyData Sphinx Theme** — the standard for scientific Python projects
   (NumPy, pandas, scikit-learn, nilearn).
2. Rebuild the landing page as a rich hero + feature cards + quickstart,
   inspired by Scallop-lang's structure.
3. Fix all broken content (encoding errors, outdated URLs, missing sections).
4. Add new content pages: Concepts, Changelog, Contributing.
5. Add a GitHub Actions workflow that automatically deploys docs to GitHub Pages
   on every push to `master`.
6. Replace all Anaconda references with **uv** (modern, fast Python package
   manager).

---

## 2. Architecture

### Files modified

| File | Change |
|------|--------|
| `doc/conf.py` | Swap theme, add sphinx-design extension, brand colour, navbar |
| `doc/index.rst` | Rewrite as hero landing page with grid cards and quickstart |
| `doc/install.rst` | Modernise with uv, keep OS tabs, fix broken URLs |
| `doc/tutorial.rst` | Clean up, improve flow |
| `doc/tutorial_logic_programming.rst` | Fix prose, keep probabilistic stub as-is |
| `doc/api.rst` | Minor cleanup |
| `setup.cfg` | Update `[options.extras_require] doc` with new theme deps |

### Files created

| File | Purpose |
|------|---------|
| `doc/_static/neurolang.css` | Brand colour overrides via PyData CSS variables |
| `doc/concepts.rst` | Concepts overview: Datalog, probabilistic reasoning, neuroimaging integration, architecture |
| `doc/changelog.rst` | What's New / release history |
| `doc/contributing.rst` | Dev setup, running tests, building docs, PR process |
| `.github/workflows/docs.yml` | Build + deploy docs to GitHub Pages on push to master |

---

## 3. Theme & Visual Identity

### Theme: PyData Sphinx Theme

```python
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NeuroLang/NeuroLang",
            "icon": "fa-brands fa-github",
        }
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "header_links_before_dropdown": 4,
}
html_sidebars = {
    "index": [],          # no sidebar on landing page
    "**": ["sidebar-nav-bs"],
}
```

### Extensions to add

- `sphinx_design` — provides `.. grid::`, `.. card::`, `.. button-ref::` directives
  for the landing page cards and quickstart.

### Brand colour: Indigo `#5C6BC0`

`doc/_static/neurolang.css`:
```css
:root {
    --pst-color-primary: #5C6BC0;
    --pst-color-primary-highlight: #3949AB;
    --pst-color-secondary: #26A69A;  /* teal accent */
}
```

---

## 4. Landing Page (`doc/index.rst`)

Structure (using `sphinx-design` directives):

1. **Hero section** — Logo, one-line tagline, two CTA buttons  
   - "Get Started →" → `install.rst`
   - "View Examples →" → gallery

2. **Three feature cards** in a `.. grid:: 1 1 3 3` layout:
   - **Language** — Datalog-based declarative language for relational queries
   - **Probabilistic Solver** — Discrete, probabilistic, and open-world reasoning
   - **Python Integration** — Embed logic programs in Python with `NeurolangDL`

3. **Quickstart block** — `pip install neurolang` + 4-line usage snippet

4. **Standard toctree** (hidden) — Install, Tutorial, Concepts, Examples, API,
   Changelog, Contributing

---

## 5. Install Page (`doc/install.rst`)

### Structure

- Tabs for: **Windows** / **macOS** / **Linux** / **From source**  
  (using `sphinx-design` `.. tab-set::` instead of the old JS-based tab HTML)
- Primary install method on all platforms:

```text
pip install neurolang
```

- If uv is available, prefer:

```text
uv pip install neurolang
```

- **From source** tab:
  ```text
  git clone https://github.com/NeuroLang/NeuroLang.git
  cd NeuroLang
  uv pip install -e ".[dev,doc]"
  ```

- **All Anaconda references removed.** uv is the recommended tool.
- Verification step: `python -c "import neurolang; print('OK')`

---

## 6. New Content Pages

### `doc/concepts.rst`

Four sections:
1. **Logic Programming & Datalog** — what rules and facts are, with a short
   readable example from the brain atlas domain
2. **Probabilistic Reasoning** — independent probabilistic facts, possible
   worlds semantics, briefly
3. **Neuroimaging Integration** — how nibabel/nilearn images and ontologies
   are loaded as relations
4. **Architecture** — text diagram of Frontend → IR → Solver pipeline

### `doc/changelog.rst`

Standard `What's New` format:
- v0.0.1 (current): initial alpha release entry (placeholder, contributors fill in)
- Instructions for adding entries

### `doc/contributing.rst`

Sections:
1. Prerequisites (Python ≥ 3.6, git, uv)
2. Dev install: `git clone` + `uv pip install -e ".[dev,doc]"`
3. Running tests: `pytest neurolang/`
4. Building docs: `make -C doc html` then open `doc/_build/html/index.html`
5. PR checklist (tests pass, docstrings updated, changelog entry added)

---

## 7. GitHub Actions — `docs.yml`

```yaml
name: Deploy Documentation

on:
  push:
    branches: [master]
  pull_request:           # build-only check on PRs, no deploy
    branches: [master]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv pip install --system -e ".[doc]"

      - name: Build docs
        run: make -C doc html

      - name: Deploy to GitHub Pages
        if: github.event_name == 'push' && github.ref == 'refs/heads/master'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./doc/_build/html
          cname: neurolang.github.io   # update if custom domain differs
```

`setup.cfg` `[options.extras_require] doc` updated to include:
```
pydata-sphinx-theme>=0.15
sphinx-design>=0.5
sphinx>=7.0
sphinx-gallery
numpydoc
```

---

## 8. What is NOT in scope

- Converting `.rst` files to MyST Markdown (future work)
- Adding executable / Binder notebooks to the gallery
- Redesigning the examples gallery (sphinx-gallery handles this)
- Filling in the probabilistic programming section of the tutorial (left as stub
  intentionally — content not yet ready)

---

## 9. Out-of-scope risks to note

- `sphinx-gallery` requires the actual Python environment with all neuroimaging
  packages to build examples. The GitHub Actions workflow will need the
  `pysdd` and `neurosynth` dependencies to succeed fully; we may need a
  `SKIP_GALLERY=1` flag for fast CI builds initially.
- The existing `logo.png` is low-resolution. A higher-res SVG logo would
  improve the PyData theme header. Out of scope but recommended follow-up.
