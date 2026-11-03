import json, sys, numpy as np
sys.path.insert(0, "/tmp/stage2")
import text_embeddings as te
P = json.load(open("/tmp/exp/paraphrases.json"))
texts = [q for t in P for q in P[t]] + list(P)
enc = te.Specter2Encoder(model_dir="/tmp/stage2/qmodels", device="cpu", batch_size=16)
E = enc._encode  # noqa
enc._load()
out = enc._encode(texts, "query").astype(np.float32)
np.savez("/tmp/exp/query_emb.npz", emb=out, texts=np.array(texts, dtype=object))
print(out.shape)
