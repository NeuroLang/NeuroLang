"""Does Sonnet's edge on real HPO patients come from memory of the case
reports? (PROTOCOL_MEMO.md)

    python memo_probe.py plan      # synthetic patients, source papers, post-cutoff cases
    python memo_probe.py run       # Sonnet calls (HPO_LLM_ROUTE=openrouter + OPENROUTER_API_KEY)
    python memo_probe.py analyse   # results/memo_analysis.txt

Three parts:
A. synthetic patients drawn from the HPO annotation itself (in no paper):
   Sonnet direct (H0) against the posterior, compared with the same gap on
   the real patients of the union run;
B. a recognition probe on the union run's real patients: can Sonnet name the
   case report a patient comes from?
C. the few real cases published after Sonnet 5.5's training cutoff (June
   2026), from phenopacket-store's main branch: descriptive only.
"""

import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import step14_hpo_hybrid as h
import union_hybrid as u
from hpo_bench import posterior, rank_of
from hpo_cases import load_cases
from hpo_names import norm

SEED = 21
N_SYNTH = 150
MIN_ANNOTATED = 6
BOOT = 10000
CUTOFF_MONTHS = ("Jul", "Aug", "Sep", "Oct", "Nov", "Dec")   # of 2026
PPS_MAIN = (Path(__file__).resolve().parents[2]
            / "neurolang_data/probe/sci/pps_main")


def pubdate(entry):
    return entry.get("epubdate") or entry.get("pubdate") or ""


def esummary(pmids):
    """PubMed summaries (date, authors) of these PMIDs (numbers only)."""
    out = {}
    pmids = sorted(set(pmids))
    for i in range(0, len(pmids), 200):
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
               "?db=pubmed&retmode=json&id=" + ",".join(pmids[i:i + 200]))
        d = json.load(urllib.request.urlopen(url, timeout=60))["result"]
        for p in d.get("uids", []):
            out[p] = dict(
                pubdate=d[p].get("pubdate", ""),
                epubdate=d[p].get("epubdate", ""),
                first_author=(d[p].get("sortfirstauthor") or "").split(" ")[0],
                title=d[p].get("title", ""),
            )
        time.sleep(0.4)
    return out


def plan():
    W = h.W
    # A. synthetic: one patient per test disease with enough annotations,
    # with as many terms as the median real patient of the union run
    union = json.load(open("results/union_results.json"))
    k = int(np.median([r["n_obs"] for r in union]))
    test_dis = sorted({d for d, _ in h.real
                       if d in W.di and len(W.by_dis[d]) >= MIN_ANNOTATED})
    rng = random.Random(SEED)
    rng.shuffle(test_dis)
    synth = []
    for d in test_dis[:N_SYNTH]:
        # as hpo_bench.simulate, for this disease: each annotated term present
        # with its frequency, redrawn (at most 1,000 times) until k are
        present = []
        for _ in range(1000):
            present = [t for t, p in W.by_dis[d].items() if rng.random() < p]
            if len(present) >= k:
                break
        if len(present) >= 3:
            synth.append(dict(disease=d, terms=sorted(rng.sample(
                present, min(k, len(present))))))
    # B. the source paper of every union patient
    with_pmid = load_cases(h.W.replaced, with_pmid=True)
    test = set(json.load(open("results/hpo_split.json"))["test"])
    real = [c for c in with_pmid if c[0] in test]
    assert [c[:2] for c in real] == [tuple(x) for x in h.real]
    recog = [dict(pid=r["pid"], disease=r["disease"],
                  pmid=real[r["pid"]][2][5:]) for r in union]
    # C. real cases on main whose paper is after the cutoff
    main = load_cases(h.W.replaced, with_pmid=True, pps=PPS_MAIN, cap=10**6)
    dates = esummary([c[2][5:] for c in main])
    late = [dict(disease=d, terms=obs, pmid=p[5:])
            for d, obs, p in main
            if d in W.di and re.match(r"2026 (%s)" % "|".join(CUTOFF_MONTHS),
                                      pubdate(dates.get(p[5:], {})))]
    papers = esummary([r["pmid"] for r in recog])
    papers.update({p: dates[p] for p in {c["pmid"] for c in late}})
    json.dump(dict(k=k, synthetic=synth, recognition=recog, late=late,
                   papers=papers),
              open("results/memo_plan.json", "w"), indent=1)
    print(f"synthetic {len(synth)} (k={k}); recognition {len(recog)}; "
          f"post-cutoff {len(late)} cases from "
          f"{len({c['pmid'] for c in late})} papers")


def prompt_source(obs):
    return (
        "A patient with these Human Phenotype Ontology terms was described "
        "in a published case report or case series:\n" + h.terms_block(obs)
        + "\n\nDo you know which publication describes this patient? Reply "
        'with a JSON object {"pmid": "<PubMed id or empty>", "first_author": '
        '"<surname or empty>", "year": "<year or empty>", "title": "<title '
        'or empty>"}. Leave the fields empty if you do not know. No other '
        "text."
    )


def parse_obj(text):
    try:
        return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except Exception:
        return None


def recognised(claim, pmid, paper):
    """The claim names the source: its PMID, or its first author with the
    year within one."""
    if not isinstance(claim, dict):
        return False
    if re.sub(r"\D", "", str(claim.get("pmid", ""))) == pmid:
        return True
    year = re.match(r"(\d{4})", pubdate(paper))
    try:
        close = year and abs(int(str(claim.get("year"))[:4])
                             - int(year.group(1))) <= 1
    except ValueError:
        close = False
    fa = norm(str(claim.get("first_author", "")))
    return bool(close and fa and fa == norm(paper.get("first_author", "")))


def claims_something(claim):
    return isinstance(claim, dict) and any(
        str(claim.get(f, "")).strip() for f in ("pmid", "first_author"))


def direct(case):
    post = posterior(h.W, case["terms"], h.ch)
    items = h.parse_json(h.ask(h.prompt_h0(case["terms"]))["text"])
    return dict(
        disease=case["disease"], pmid=case.get("pmid"),
        posterior=rank_of(post, h.W.di[case["disease"]]),
        H0=h.rank_direct(items, case["disease"]),
        H0_lenient=u.rank_lenient(items, case["disease"]),
        parsed=items is not None,
    )


def run():
    p = json.load(open("results/memo_plan.json"))

    def synth(case):
        r = direct(case)
        claim = parse_obj(h.ask(prompt_source(case["terms"]))["text"])
        r["claims_source"] = claims_something(claim)
        return r

    def recog(rec):
        cell = next(x["cell"] for x in json.load(open(
            "results/union_plan.json")) if x["pid"] == rec["pid"])
        claim = parse_obj(h.ask(prompt_source(u.observed(rec["pid"],
                                                         *cell)))["text"])
        return dict(rec, claim=claim, claims_source=claims_something(claim),
                    recognised=recognised(claim, rec["pmid"],
                                          p["papers"].get(rec["pmid"], {})))

    with ThreadPoolExecutor(6) as pool:
        out = dict(
            synthetic=list(pool.map(synth, p["synthetic"])),
            recognition=list(pool.map(recog, p["recognition"])),
            late=list(pool.map(direct, p["late"])),
        )
    json.dump(out, open("results/memo_results.json", "w"), indent=1)
    analyse()


def rr(x):
    return 0.0 if x is None else 1.0 / x


def boot_mean(values, groups, seed=0):
    by = {}
    for j, g in enumerate(groups):
        by.setdefault(g, []).append(j)
    keys = sorted(by)
    rng = random.Random(seed)
    v = np.asarray(values, float)
    bs = np.sort([v[[j for g in (rng.choice(keys) for _ in keys)
                     for j in by[g]]].mean() for _ in range(BOOT)])
    return v.mean(), bs[int(.025 * BOOT)], bs[int(.975 * BOOT) - 1]


def analyse():
    res = json.load(open("results/memo_results.json"))
    union = json.load(open("results/union_results.json"))
    lines = []
    # A
    s = res["synthetic"]
    gap_s = [rr(r["H0"]) - rr(r["posterior"]) for r in s]
    gap_r = [rr(r["H0"]) - rr(r["posterior"]) for r in union]
    ms, ls, hs = boot_mean(gap_s, [r["disease"] for r in s])
    mr, lr, hr = boot_mean(gap_r, [r["disease"] for r in union])
    rng = np.random.default_rng(0)
    diff = np.sort([rng.choice(gap_r, len(gap_r)).mean()
                    - rng.choice(gap_s, len(gap_s)).mean()
                    for _ in range(BOOT)])
    lines += [
        f"A. synthetic patients: {len(s)} (unparsed H0 "
        f"{sum(not r['parsed'] for r in s)})",
        f"  MRR posterior {np.mean([rr(r['posterior']) for r in s]):.3f}, "
        f"Sonnet H0 {np.mean([rr(r['H0']) for r in s]):.3f} (lenient "
        f"{np.mean([rr(r['H0_lenient']) for r in s]):.3f})",
        f"  Sonnet - posterior, synthetic {ms:+.3f} [{ls:+.3f}, {hs:+.3f}]; "
        f"real (union run) {mr:+.3f} [{lr:+.3f}, {hr:+.3f}]",
        f"  real gap - synthetic gap {np.mean(gap_r) - np.mean(gap_s):+.3f} "
        f"[{diff[int(.025 * BOOT)]:+.3f}, {diff[int(.975 * BOOT) - 1]:+.3f}]",
        f"  Sonnet claims a source paper for {sum(r['claims_source'] for r in s)}"
        f" of {len(s)} synthetic patients (no such paper exists)",
    ]
    # B
    rc = res["recognition"]
    h0 = {r["pid"]: r for r in union}
    yes = [r for r in rc if r["recognised"]]
    no = [r for r in rc if not r["recognised"]]
    lines += [
        f"B. recognition on {len(rc)} real patients: claims a source "
        f"{sum(r['claims_source'] for r in rc)}, names the right one "
        f"{len(yes)} ({100 * len(yes) / len(rc):.1f}%) from "
        f"{len({r['pmid'] for r in yes})} papers",
        "  Sonnet H0 MRR, recognised "
        + (f"{np.mean([rr(h0[r['pid']]['H0']) for r in yes]):.3f}"
           if yes else "-")
        + f" (n={len(yes)}), not recognised "
        f"{np.mean([rr(h0[r['pid']]['H0']) for r in no]):.3f} (n={len(no)});"
        " posterior: "
        + (f"{np.mean([rr(h0[r['pid']]['posterior']) for r in yes]):.3f}"
           if yes else "-")
        + f" / {np.mean([rr(h0[r['pid']]['posterior']) for r in no]):.3f}",
    ]
    if no:
        m, lo, hi = boot_mean([rr(h0[r["pid"]]["H0"])
                               - rr(h0[r["pid"]]["posterior"]) for r in no],
                              [r["disease"] for r in no])
        lines.append(f"  Sonnet - posterior on the not-recognised patients "
                     f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # C
    lt = res["late"]
    if lt:
        m, lo, hi = boot_mean([rr(r["H0"]) - rr(r["posterior"]) for r in lt],
                              [r["pmid"] for r in lt])
        lines += [
            f"C. post-cutoff cases: {len(lt)} from "
            f"{len({r['pmid'] for r in lt})} papers (descriptive)",
            f"  MRR posterior {np.mean([rr(r['posterior']) for r in lt]):.3f}"
            f", Sonnet H0 {np.mean([rr(r['H0']) for r in lt]):.3f}; Sonnet -"
            f" posterior {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (resampling papers)",
        ]
    text = "\n".join(lines)
    print(text)
    open("results/memo_analysis.txt", "w").write(text + "\n")


if __name__ == "__main__":
    dict(plan=plan, run=run, analyse=analyse)[sys.argv[1]]()
