# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Python Environment
- Python 3.12 via `uv` (venv at `.venv/`)
- Install: `uv sync --extra dev --extra server`
- NeuroLang is installed in development mode from this repo

## Node.js Environment
- Node.js v25.2.1, npm 11.12.0 (via Homebrew on macOS ARM)
- React app at `neurolang/utils/server/neurolang-sparklis/`

## NeuroLang Data
- Datasets are downloaded to `~/neurolang_data/` on first engine start
- Neurosynth: peak coordinates, study IDs, term-study TF-IDF associations
- Destrieux: atlas regions with spatial relationships
- First engine initialization can take 30-60 seconds

## Key Dependencies (Backend)
- `tornado>=6.5`: web framework for backend
- `pyyaml`: for queries.yaml parsing
- `nibabel`: NIfTI image handling
- `numpy`, `pandas`: data processing
- `matplotlib`, `seaborn`: plotting (for agg_kde function)

## Key Dependencies (Frontend - to be installed)
- `react`, `react-dom`: UI framework
- `@niivue/niivue`: brain image visualization
- `@codemirror/view`, `@codemirror/state`, `@codemirror/lang-*`: code editor
- `vite`: build tool
- `vitest`, `@testing-library/react`: testing
- `typescript`: type checking
