// The links of the studies of a result: its PubMed record, its DOI (which the
// engine's metadata has), and its PubMed Central record, which Europe PMC
// tells for the studies of a page, from their PubMed ids (it answers pages
// from any site; the NCBI converter does not).

export const pubmedUrl = (id) => `https://pubmed.ncbi.nlm.nih.gov/${id}/`;
export const doiUrl = (doi) => `https://doi.org/${encodeURI(doi)}`;
export const pmcUrl = (pmcid) => `https://pmc.ncbi.nlm.nih.gov/articles/${pmcid}/`;

const EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search";
const BATCH = 50;

// The columns of *payload* that hold studies, as the engine tells them.
export const studyColumns = (payload) => payload.study_links?.columns ?? [];

// What a record says of its study, for a tooltip: "Title. Authors (year) Journal".
export function citation(record = {}) {
  const who = [record.authors, record.year ? `(${record.year})` : ""].filter(Boolean).join(" ");
  const title = (record.title ?? "").replace(/[.\s]+$/, "");
  return [title, [who, record.journal].filter(Boolean).join(" ")].filter(Boolean).join(". ");
}

// The request that finds the PubMed Central ids and DOIs of the studies *ids*.
export function europePmcUrl(ids) {
  const query = `SRC:MED AND (${ids.map((id) => `EXT_ID:${id}`).join(" OR ")})`;
  return `${EUROPE_PMC}?query=${encodeURIComponent(query)}&resultType=lite&pageSize=${ids.length}&format=json`;
}

// The PubMed Central ids and DOIs of Europe PMC's *answer*, by PubMed id.
export function parseEuropePmc(answer) {
  const found = new Map();
  for (const record of answer?.resultList?.result ?? []) {
    if (record.pmid && (record.pmcid || record.doi)) {
      found.set(String(record.pmid), { pmcid: record.pmcid, doi: record.doi });
    }
  }
  return found;
}

// Looks the studies up as the page shows them, a request for each 50, and
// remembers what it found. A request that fails is not asked again.
export class PmcLookup {
  constructor(fetchFn = (...args) => fetch(...args)) {
    this.fetch = fetchFn;
    this.known = new Map();
    this.asked = new Set();
  }

  // The ids not asked yet are, and the *known* are given: id -> { pmcid, doi }.
  async lookup(ids) {
    const wanted = [...new Set(ids.map(String))].filter((id) => /^\d{1,12}$/.test(id) && !this.asked.has(id));
    for (let start = 0; start < wanted.length; start += BATCH) {
      const batch = wanted.slice(start, start + BATCH);
      batch.forEach((id) => this.asked.add(id));
      try {
        const response = await this.fetch(europePmcUrl(batch));
        if (response.ok) {
          for (const [id, info] of parseEuropePmc(await response.json())) this.known.set(id, info);
        }
      } catch {
        // offline, or refused: the PubMed and DOI links are there
      }
    }
    return this.known;
  }
}
