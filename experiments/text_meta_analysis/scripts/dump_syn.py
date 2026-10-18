import json, numpy as np, pandas as pd
from pathlib import Path
from neurolang.utils import engine_registry as er
nl = er.build_engine("ns_syn", Path("/tmp/nsdata2"), resolution=2, yaml_path=Path("/tmp/custom_syn.yaml"))
pk = pd.DataFrame(list(nl.symbols["peak_reported"].value), columns=["i", "j", "k", "s"])
tf = pd.DataFrame(list(nl.symbols["term_in_study_tfidf"].value), columns=["s", "term", "tfidf"])
bm = np.array(list(nl.symbols["brain_mask"].value))
pk.to_parquet("/tmp/exp/peaks.parquet"); tf.to_parquet("/tmp/exp/tfidf.parquet"); np.save("/tmp/exp/brain.npy", bm)
print("peaks", len(pk), "tfidf rows", len(tf), "terms", tf.term.nunique(), "brain voxels", len(bm), "studies", pk.s.nunique(), flush=True)
