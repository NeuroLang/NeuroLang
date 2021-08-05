// ENV variables. VITE will update the values of the `import.meta.env`
// variables based on the mode (dev, prod, stage)
// See https://vitejs.dev/guide/env-and-mode.html#env-variables
export const VITE_MODE = import.meta.env.VITE_APP_TITLE
export const BASE_URL = import.meta.env.BASE_URL
export const DEV = import.meta.env.DEV
export const PROD = import.meta.env.PROD

// Base URL for the backend API. Depends on the mode.
export const BASE_API_URL = DEV ? 'http://localhost:8888' : 'http://localhost:8888'
export const BASE_API_HOST = DEV ? 'localhost:8888' : 'localhost:8888'

// The API routes
export const API_ROUTE = {
  statement: BASE_API_URL + '/v1/statement',
  statementsocket: 'ws://' + BASE_API_HOST + '/v1/statementsocket',
  status: BASE_API_URL + '/v1/status',
  atlas: BASE_API_URL + '/v1/atlas',
  engines: BASE_API_URL + '/v1/engines'
}

// The custom data types which might need special rendering
export const DATA_TYPES = {
  studyID: "<class 'neurolang.frontend.neurosynth_utils.StudyID'>",
  VBR: "<class 'neurolang.regions.ExplicitVBR'>",
  VBROverlay: "<class 'neurolang.regions.ExplicitVBROverlay'>"
}

export const PUBMED_BASE_URL = 'https://www.ncbi.nlm.nih.gov/pubmed/?term='
