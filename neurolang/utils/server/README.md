# NeuroLang Web Interfaces

This directory contains the legacy **neurolang-web** frontend (HTML + JavaScript).

The Tornado server and Sparklis React/TypeScript frontend have been extracted into
the standalone `neurolang-sparklis` package. See:

<https://github.com/NeuroLang/NeuroLang/tree/master/neurolang-sparklis>

## neurolang-web (legacy)

The frontend application can be built for production by running the command

```bash
$ cd neurolang-web
$ npm install
$ npm run build
```

The backend server is now provided by the `neurolang-sparklis` package:

```bash
$ uv run neuro-sparklis --port 8888
```

## Building

For convenience, if npm is available in your path, the frontend is built
automatically via the hatchling build hook (`hatch_build.py`) when running
`pip install -e .`.
