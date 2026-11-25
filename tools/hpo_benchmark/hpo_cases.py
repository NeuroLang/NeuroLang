"""Leakage-filtered real phenopackets (spec 2026-11-23-hpo-benchmark-design)."""
import collections, glob, json, random, re
from pathlib import Path
from hpo_lib import HPOA

PPS = Path(__file__).resolve().parents[2] / "neurolang_data/probe/sci/pps"


def documented_pmids():
    refs = collections.defaultdict(set)
    for line in open(HPOA, encoding="utf-8"):
        if line.startswith("#") or line.startswith("database_id"):
            continue
        f = line.rstrip("\n").split("\t")
        for r in f[4].split(";"):
            if r.startswith("PMID:"):
                refs[f[0]].add(r)
    return refs


def load_cases(replaced=None, min_terms=3, cap=5, seed=0, with_pmid=False, pps=PPS):
    """[(disease, observed_terms)] without leaked PMIDs, capped per disease;
    with_pmid adds the source PMID as a third element (same order)."""
    refs, replaced = documented_pmids(), replaced or {}
    out, dropped_leak = [], 0
    for fn in sorted(glob.glob(str(Path(pps) / "**/*.json"), recursive=True)):
        m = re.search(r"PMID_(\d+)", fn)
        if m is None:           # not a phenopacket of a publication
            continue
        d = json.load(open(fn))
        if not d.get("diseases"):
            continue
        dis = d["diseases"][0]["term"]["id"]
        pmid = "PMID:" + m.group(1)
        if pmid in refs[dis]:
            dropped_leak += 1
            continue
        feats = d.get("phenotypicFeatures") or []
        obs = sorted({replaced.get(p["type"]["id"], p["type"]["id"])
                      for p in feats if not p.get("excluded")})
        if len(obs) >= min_terms:
            out.append((dis, obs, pmid) if with_pmid else (dis, obs))
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for c in out:
        by[c[0]].append(c)
    kept = []
    for dis in sorted(by):
        rng.shuffle(by[dis])
        kept += by[dis][:cap]
    return kept
