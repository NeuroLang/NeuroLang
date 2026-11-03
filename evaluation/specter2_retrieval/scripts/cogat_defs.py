import json
from neurolang.frontend import NeurolangPDL
nl = NeurolangPDL()
nl.load_ontology("/tmp/cogat.v2.owl", individuals=True, names="prefixed")
lab = dict((c, l) for c, l in nl.symbols["rdfs:label"].value)
dfn = dict((c, d) for c, d in nl.symbols["skos:definition"].value)
by = {}
for c, l in lab.items():
    d = (dfn.get(c) or "").strip()
    if d: by.setdefault(l.strip().lower(), d)
meta = json.load(open("/tmp/stage4/meta.json"))
match = {c: by[c.strip().lower()] for c in meta["concept"] if c.strip().lower() in by and len(by[c.strip().lower()]) > 30}
print("cogat labels with a definition:", len(by), "| NeuroStore concepts matching:", len(match), list(match)[:10])
json.dump(match, open("/tmp/stage4/concept_defs.json", "w"))
json.dump({"def_q": {"adapter": "query", "texts": list(match.values())}, "def_d": {"adapter": "document", "texts": list(match.values())}}, open("/tmp/stage4/jobs_defs.json", "w"))
