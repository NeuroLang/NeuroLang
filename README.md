<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NeuroLang/NeuroLang/master/doc/_static/logo/svg/neurolang-lockup-dark.svg">
    <img alt="NeuroLang" src="https://raw.githubusercontent.com/NeuroLang/NeuroLang/master/doc/_static/logo/svg/neurolang-lockup.svg" width="420">
  </picture>
</h1>

[![Tests](https://github.com/NeuroLang/NeuroLang/actions/workflows/tests.yml/badge.svg)](https://github.com/NeuroLang/NeuroLang/actions/workflows/tests.yml)
[![Coverage](https://raw.githubusercontent.com/NeuroLang/NeuroLang/badges/coverage.svg)](https://github.com/NeuroLang/NeuroLang/actions/workflows/tests.yml)
[![Documentation](https://github.com/NeuroLang/NeuroLang/actions/workflows/docs.yml/badge.svg)](https://neurolang.github.io)
[![Spelling](https://github.com/NeuroLang/NeuroLang/actions/workflows/codespell.yml/badge.svg)](https://github.com/NeuroLang/NeuroLang/actions/workflows/codespell.yml)
[![PyPI](https://img.shields.io/pypi/v/neurolang.svg?color=brightgreen)](https://pypi.org/project/neurolang/)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FNeuroLang%2FNeuroLang%2Fmaster%2Fpyproject.toml&color=brightgreen)](https://github.com/NeuroLang/NeuroLang/blob/master/pyproject.toml)


NeuroLang enables the analysis of neuroimaging data through probabilistic logic programming. It seamlessly combines images, databases, and ontologies within a single framework.


Important links
===============
- Official source code repo: https://github.com/Neurolang/Neurolang/
- Try the interactive query REPL, `neurolang-query -i --engine neurosynth --resolution 2`, in your browser, with nothing to install (it downloads about 100 MB): https://neurolang.github.io/_static/repl/index.html. Add `?ref=BRANCH` (a branch, tag or commit; `&repo=OWNER/NAME` for a fork) to run the NeuroLang of that ref instead of the page's own. To run it from a checkout on your machine: `uv run --extra apps neurolang-repl-serve`
- In-development interactive gallery of examples (best viewed with Chrome) http://neurolang-u18.saclay.inria.fr/hub/gallery
- HTML documentation (stable release): http://neurolang.github.io/
- HTML documentation (dev release): http://neurolang.github.io/dev

Citing
======
G. E. Zanitti, Y. Soto, V. Iovene, M. V. Martinez, R. O. Rodriguez, G. I. Simari, D. Wassermann. *Scalable Query Answering Under Uncertainty to Neuroscientific Ontological Knowledge: The NeuroLang Approach*. Neuroinformatics, 21(2), 407-425, 2023.

