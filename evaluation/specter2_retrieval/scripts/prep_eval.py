import json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
out = Path("/tmp/stage4"); out.mkdir(exist_ok=True)
CHECK = {"Humans","Male","Female","Adult","Adolescent","Aged","Middle Aged","Young Adult","Child","Animals","Aged, 80 and over","Infant","Child, Preschool","Rats","Mice"}
# Neurosynth
ids = json.load(open("/tmp/stage3/ids.json")); texts = {k: tuple(v) for k, v in json.load(open("/tmp/stage3/texts.json")).items()}
mesh = json.load(open("/tmp/stage3/mesh.json"))
M = np.load("/tmp/nsdata2/ns_syn/embeddings/study_abstracts.npz", allow_pickle=True)["matrix"]
has = np.linalg.norm(M, axis=1) > 0
docs = [i for i, k in enumerate(ids) if has[i] and texts[k][0].strip()]
print("neurosynth docs with embedding and title:", len(docs), "of", len(ids))
pos = defaultdict(list)
for row in docs:
    for name, ui, major in mesh.get(ids[row], []):
        if name not in CHECK: pos[name].append(row)
mq = {n: r for n, r in pos.items() if 20 <= len(r) <= 1000}
print("MeSH descriptors with 20-1000 studies:", len(mq), "| of all descriptors", len(pos), "| positives/q median", int(np.median([len(v) for v in mq.values()])))
# NeuroStore
ids2 = json.load(open("/tmp/stage2/ids.json")); t2 = {k: tuple(v) for k, v in json.load(open("/tmp/stage2/texts.json")).items()}
M2 = np.load("/tmp/nsdata/ns_local/embeddings/study_abstracts.npz", allow_pickle=True)["matrix"]
has2 = np.linalg.norm(M2, axis=1) > 0
docs2 = [i for i, k in enumerate(ids2) if has2[i] and t2.get(k, ("", ""))[0].strip()]
print("neurostore docs with embedding and title:", len(docs2), "of", len(ids2))
import sys; sys.path.insert(0, "/Users/bot/NeuroLang-neurostore")
from neurolang.utils.engines.neurostore.init import _extract_labels
ann = pd.read_parquet("/tmp/nimare_probe/ex/neurostore-studyset-2026-10/annotations.parquet")
row_of = {k: i for i, k in enumerate(ids2)}
cq, dq = defaultdict(list), defaultdict(list)
for suffix, store in (("Concepts", cq), ("Domain", dq)):
    lab = _extract_labels(ann, suffix)
    for s, c in zip(lab.study_id, lab[suffix]):
        if s in row_of and has2[row_of[s]]: store[c].append(row_of[s])
cq = {c: sorted(set(r)) for c, r in cq.items() if 20 <= len(set(r)) <= 3000}
dq = {c: sorted(set(r)) for c, r in dq.items() if len(set(r)) >= 20}
print("NeuroStore concepts with 20-3000 studies:", len(cq), "| domains:", len(dq), sorted(dq)[:12])
json.dump({"ns_docs": docs, "nst_docs": docs2, "mesh": {k: v for k, v in mq.items()}, "concept": cq, "domain": dq}, open(out / "meta.json", "w"))
light = {
  "ns_title_q": {"adapter": "query", "texts": [texts[ids[i]][0] for i in docs]},
  "ns_title_d": {"adapter": "document", "texts": [texts[ids[i]][0] for i in docs]},
  "nst_title_q": {"adapter": "query", "texts": [t2[ids2[i]][0] for i in docs2]},
  "mesh_q": {"adapter": "query", "texts": list(mq)}, "mesh_d": {"adapter": "document", "texts": list(mq)},
  "concept_q": {"adapter": "query", "texts": list(cq)}, "concept_d": {"adapter": "document", "texts": list(cq)},
  "domain_q": {"adapter": "query", "texts": list(dq)}, "domain_d": {"adapter": "document", "texts": list(dq)},
}
heavy = {"ns_abstract_only_d": {"adapter": "document", "texts": [texts[ids[i]][1] for i in docs]}}
json.dump(light, open(out / "jobs_light.json", "w")); json.dump(heavy, open(out / "jobs_heavy.json", "w"))
print({k: len(v["texts"]) for k, v in {**light, **heavy}.items()})
