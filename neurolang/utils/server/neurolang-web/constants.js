// ENV variables. VITE will update the values of the `import.meta.env`
// variables based on the mode (dev, prod, stage)
// See https://vitejs.dev/guide/env-and-mode.html#env-variables
export const VITE_MODE = import.meta.env.MODE
export const BASE_URL = import.meta.env.BASE_URL
export const DEV = import.meta.env.DEV
export const PROD = import.meta.env.PROD

// Base URL for the backend API. Depends on the mode.
export let BASE_API_URL
export let BASE_API_HOST
switch (VITE_MODE) {
  case 'stage':
    BASE_API_URL = 'http://neurolang-interne.saclay.inria.fr'
    BASE_API_HOST = 'neurolang-interne.saclay.inria.fr'
    break
  case 'production':
    BASE_API_URL = 'http://neurolang-u18.saclay.inria.fr'
    BASE_API_HOST = 'neurolang-u18.saclay.inria.fr'
    break
  default:
    BASE_API_URL = 'http://localhost:8888'
    BASE_API_HOST = 'localhost:8888'
}

// The API routes
export const API_ROUTE = {
  statement: BASE_API_URL + '/v1/statement',
  statementsocket: 'ws://' + BASE_API_HOST + '/v1/statementsocket',
  autocompletion: BASE_API_URL + '/v1/autocompletion',
  status: BASE_API_URL + '/v1/status',
  atlas: BASE_API_URL + '/v1/atlas',
  engines: BASE_API_URL + '/v1/engines',
  symbols: BASE_API_URL + '/v1/symbol',
  downloads: BASE_API_URL + '/v1/download',
  figure: BASE_API_URL + '/v1/figure'
}

// The custom data types which might need special rendering
export const DATA_TYPES = {
  studyID: "<class 'neurolang.frontend.neurosynth_utils.StudyID'>",
  VBR: "<class 'neurolang.regions.ExplicitVBR'>",
  VBROverlay: "<class 'neurolang.regions.ExplicitVBROverlay'>",
  MpltFigure: "<class 'matplotlib.figure.Figure'>"
}

export const PUBMED_BASE_URL = 'https://www.ncbi.nlm.nih.gov/pubmed/?term='
