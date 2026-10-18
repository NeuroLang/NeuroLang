import json, sys, logging
from pathlib import Path
here = Path(__file__).parent
sys.path.insert(0, str(here / "pylib")); sys.path.insert(0, str(here))
import numpy as np, torch, text_embeddings as te
logging.basicConfig(level=logging.INFO)
jobs = json.load(open(here / sys.argv[1])); out = {}
enc = te.Specter2Encoder(model_dir=here / "models", batch_size=64); enc._load()
print("torch", torch.__version__, "threads", torch.get_num_threads(), flush=True)
for name, job in jobs.items():
    print(name, len(job["texts"]), job["adapter"], flush=True)
    out[name] = enc._encode(job["texts"], job["adapter"]).astype(np.float32)
    np.savez(here / sys.argv[2], **out)
print("done", flush=True)
