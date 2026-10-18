import json, time, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import nibabel as nib, nilearn.datasets as nd, nilearn.image as ni
from neurolang.utils import engine_registry as er
P = json.load(open("/tmp/exp/paraphrases.json")); Q = np.load("/tmp/exp/query_emb.npz", allow_pickle=True)
vec = {t: v for t, v in zip(Q["texts"], Q["emb"])}
nl = er.build_engine("ns_syn", Path("/tmp/nsdata2"), resolution=2, yaml_path=Path("/tmp/custom_syn.yaml"))
idx = nl.text_embedding_indexes["study_abstracts"]
def enc(text):
    assert text in vec, text
    return np.asarray(vec[text], dtype=np.float64)
idx.encoder.encode_query = enc
mni = nd.load_mni152_template(resolution=2); n_all = len(nl.symbols["study"].value)
titles = {str(k): v[0] for k, v in json.load(open("/tmp/stage3/texts.json")).items()}
def run(prog): return nl.execute_datalog_program(prog)
def image(name, extra=""):
    r = run(f"ans(i, j, k, u) :- {name}_uniformity({extra}i, j, k, u).").as_pandas_dataframe().values
    d = np.zeros(mni.shape); d[tuple(r[:, :3].astype(int).T)] = r[:, 3]; return d
def corr(a, b):
    m = (a > 0) | (b > 0); u, v = a[m], b[m]; ta, tb = u > np.quantile(u, .95), v > np.quantile(v, .95)
    return np.corrcoef(u, v)[0, 1], 2 * (ta & tb).sum() / (ta.sum() + tb.sum())
run("""
  activation(i, j, k, s) :- brain_mask(i, j, k) & peak_reported(i1, j1, k1, s) & (EUCLIDEAN(i, j, k, i1, j1, k1) <= 3).
  n_voxels(count(i, j, k)) :- brain_mask(i, j, k).
  selected_voxel(i, j, k) ^ (1 / m) :- brain_mask(i, j, k) & n_voxels(m).
  term_study(t, s) :- term_in_study_tfidf(s, t, w) & (w > 0.001).
""")
def uniformity_rules(sel):
    run(f"""
      {sel}_given(i, j, k, PROB) :- (activation(i, j, k, s) & selected_study(s)) // ({sel}(s) & selected_study(s)).
      {sel}_any(PROB) :- (activation(i, j, k, s) & selected_voxel(i, j, k) & selected_study(s)) // ({sel}(s) & selected_study(s)).
      {sel}_uniformity(i, j, k, u) :- {sel}_given(i, j, k, p) & {sel}_any(pbar) & (u == p / pbar).
    """)
out = {}
# ---- D1: a free-text query about pain, end to end in Datalog
q1 = P["pain"][0]; n_pain = int(run("ans(s) :- term_study('pain', s).").as_pandas_dataframe().shape[0])
pct = 100 * (1 - n_pain / n_all)
run(f"""
  sim1(s, v) :- study(s) & (v == query_similarity("{q1}", s)) & (v > 0).
  cut1(percentile(v, {pct:.4f})) :- sim1(s, v).
  txt1(s) :- sim1(s, v) & cut1(c) & (v >= c).
""")
uniformity_rules("txt1"); t = time.perf_counter(); u1 = image("txt1"); print("D1 text query: %r" % q1)
sel1 = run("ans(s) :- txt1(s).").as_pandas_dataframe().iloc[:, 0]
run("pain_sel(s) :- term_study('pain', s)."); uniformity_rules("pain_sel"); ref1 = image("pain_sel")
r, dc = corr(u1, ref1); print("   selected %d studies (the term 'pain' has %d); map vs the 'pain' map: r=%.3f Dice=%.3f (%.0fs)" % (len(sel1), n_pain, r, dc, time.perf_counter() - t))
print("   closest studies:", "; ".join(titles[str(s)][:55] for s in list(sel1)[:3]))
out["D1_text_pain"] = u1; out["D1_term_pain"] = ref1
# ---- D2: set difference. studies about holding information in mind that do NOT mention working memory
q2 = P["working memory"][0]; n_wm = len(run("ans(s) :- term_study('working memory', s).").as_pandas_dataframe())
pct2 = 100 * (1 - 2 * n_wm / n_all)  # twice as many candidates, so that enough remain after the difference
run(f"""
  sim2(s, v) :- study(s) & (v == query_similarity("{q2}", s)) & (v > 0).
  cut2(percentile(v, {pct2:.4f})) :- sim2(s, v).
  txt2(s) :- sim2(s, v) & cut2(c) & (v >= c).
  tagged(s) :- term_study('working memory', s).
  untagged2(s) :- txt2(s) & ~tagged(s).
""")
sel2 = run("ans(s) :- untagged2(s).").as_pandas_dataframe().iloc[:, 0]; tx2 = run("ans(s) :- txt2(s).").as_pandas_dataframe().iloc[:, 0]
uniformity_rules("untagged2"); u2 = image("untagged2"); run("wm_sel(s) :- term_study('working memory', s)."); uniformity_rules("wm_sel"); ref2 = image("wm_sel")
r, dc = corr(u2, ref2); print("\nD2 text query: %r" % q2)
print("   %d studies close to the text, %d of them do not mention 'working memory' (the term tags %d studies); map of the untagged ones vs the 'working memory' map: r=%.3f Dice=%.3f" % (len(tx2), len(sel2), n_wm, r, dc))
print("   some untagged studies:", "; ".join(titles[str(s)][:60] for s in list(sel2)[:4]))
out["D2_untagged"] = u2; out["D2_term_wm"] = ref2
# ---- D3: region constraint and a conditional probability. Fear text, studies reporting a peak in the amygdala
ho = nd.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", data_dir="/tmp/hodata", verbose=0)
lab = np.asarray(ni.resample_to_img(nib.load(ho.maps) if isinstance(ho.maps, str) else ho.maps, nib.Nifti1Image(np.zeros(mni.shape), mni.affine), interpolation="nearest").dataobj).astype(int)
amy = np.argwhere(np.isin(lab, [i for i, n in enumerate(ho.labels) if "Amygdala" in n])); nl.add_tuple_set(amy, name="amygdala")
q3 = P["fear"][2]; n_fear = len(run("ans(s) :- term_study('fear', s).").as_pandas_dataframe()); pct3 = 100 * (1 - n_fear / n_all)
run(f"""
  sim3(s, v) :- study(s) & (v == query_similarity("{q3}", s)) & (v > 0).
  cut3(percentile(v, {pct3:.4f})) :- sim3(s, v).
  txt3(s) :- sim3(s, v) & cut3(c) & (v >= c).
  amy_peak(s) :- peak_reported(i, j, k, s) & amygdala(i, j, k).
  txt3_amy(s) :- txt3(s) & amy_peak(s).
  n_txt3(count(s)) :- txt3(s).
  n_txt3_amy(count(s)) :- txt3_amy(s).
  n_fear(count(s)) :- term_study('fear', s).
  fear_amy(s) :- term_study('fear', s) & amy_peak(s).
  n_fear_amy(count(s)) :- fear_amy(s).
  n_amy(count(s)) :- amy_peak(s).
  n_study(count(s)) :- study(s).
""")
run("""
  amy_lift(a, nt, nta, nf, nfa, ns, lt, lf) :- n_amy(a) & n_txt3(nt) & n_txt3_amy(nta) & n_fear(nf) & n_fear_amy(nfa) & n_study(ns) & (lt == (nta / nt) / (a / ns)) & (lf == (nfa / nf) / (a / ns)).
""")
a_, nt, nta, nf, nfa, ns_, lt, lf = run("ans(a, nt, nta, nf, nfa, ns, lt, lf) :- amy_lift(a, nt, nta, nf, nfa, ns, lt, lf).").as_pandas_dataframe().iloc[0]
g, f, b = nta / nt, nfa / nf, a_ / ns_
print("\nD3 text query: %r" % q3)
print("   P(a peak in the amygdala): all studies %.3f | studies close to the text %.3f (lift %.2f) | studies tagged 'fear' %.3f (lift %.2f)" % (b, g, lt, f, lf))
n3 = len(run("ans(s) :- txt3_amy(s).").as_pandas_dataframe()); uniformity_rules("txt3_amy"); u3 = image("txt3_amy")
uniformity_rules("txt3"); u3b = image("txt3")
print("   %d of the studies close to the text report an amygdala peak; their uniformity map vs the map of all text-selected studies: r=%.3f, max ratio %.1f vs %.1f" % (n3, corr(u3, u3b)[0], u3.max(), u3b.max()))
out["D3_text_fear"] = u3b; out["D3_text_fear_amygdala"] = u3
np.savez("/tmp/exp/demo_maps.npz", affine=mni.affine, **out); print("saved")
