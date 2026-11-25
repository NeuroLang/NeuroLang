"""Disease names (HPOA) and HPO term labels (hp.json)."""
import json, re
from hpo_lib import HPOA
from hpo_onto import HPJSON


def disease_names():
    out = {}
    for line in open(HPOA, encoding="utf-8"):
        if line.startswith("#") or line.startswith("database_id"):
            continue
        f = line.rstrip("\n").split("\t")
        out[f[0]] = f[1]
    return out


def term_labels():
    g = json.load(open(HPJSON))["graphs"][0]
    return {"HP:" + n["id"].rsplit("HP_", 1)[1]: n.get("lbl", "") for n in g["nodes"] if "HP_" in n["id"]}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
