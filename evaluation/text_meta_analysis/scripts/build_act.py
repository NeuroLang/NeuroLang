import numpy as np, pandas as pd, scipy.sparse as sp, json
pk = pd.read_parquet("/tmp/exp/peaks.parquet"); bm = np.load("/tmp/exp/brain.npy"); ids = json.load(open("/tmp/stage3/ids.json"))
shape = (99, 117, 95)
lut = -np.ones(shape, np.int32); lut[tuple(bm.T)] = np.arange(len(bm))
r = 3; g = np.mgrid[-r:r+1, -r:r+1, -r:r+1].reshape(3, -1).T; off = g[(g ** 2).sum(1) <= r * r]
row_of = {int(k): i for i, k in enumerate(ids)}
rows = pk.s.map(row_of).values; ijk = pk[["i", "j", "k"]].values
ok = np.all((ijk >= 0) & (ijk < shape), axis=1); rows, ijk = rows[ok], ijk[ok]
R, C = [], []
for o in off:
    v = ijk + o; inside = np.all((v >= 0) & (v < shape), axis=1); c = lut[tuple(v[inside].T)]; keep = c >= 0
    R.append(rows[inside][keep]); C.append(c[keep])
R = np.concatenate(R); C = np.concatenate(C)
A = sp.coo_matrix((np.ones(len(R), np.float32), (R, C)), shape=(len(ids), len(bm))).tocsr(); A.sum_duplicates(); A.data[:] = 1
sp.save_npz("/tmp/exp/act.npz", A); print("activation matrix", A.shape, "nnz", A.nnz, "mean voxels/study", A.nnz / A.shape[0])
# validate against the Datalog map of 'working memory' TF-IDF > 0.001 (1,091 studies)
tf = pd.read_parquet("/tmp/exp/tfidf.parquet"); sel = tf[(tf.term == "working memory") & (tf.tfidf > 0.001)].s.map(row_of).values
Pv = np.asarray(A[sel].sum(0)).ravel() / len(sel); U = Pv / Pv.mean()
vol = np.zeros(shape); vol[tuple(bm.T)] = U
ref = np.load("/tmp/dump_syn_tfidf.npz", allow_pickle=True)["tfidf_001"]
m = (ref > 0) | (vol > 0); print("validation vs Datalog (Neurosynth working memory): r = %.6f, max |diff| = %.2e" % (np.corrcoef(vol[m], ref[m])[0, 1], np.abs(vol - ref).max()))
