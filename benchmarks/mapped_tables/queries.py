"""Write the queries bench.py runs for ENGINE as JSON: the example queries of
the engine (its ``assistant_examples``) and the benchmark queries of
``queries/*.dl``, each a program of its own.

    python queries.py ENGINE OUT.json
"""
import json
import sys
from pathlib import Path

from neurolang_apps.utils import engine_registry

engine, out = sys.argv[1], sys.argv[2]
cfg = engine_registry.get_engine_config(engine)
queries = {
    f"ex{i:03d}": example["datalog"]
    for i, example in enumerate(cfg.get("assistant_examples") or [])
    if example.get("datalog")
}
for path in sorted((Path(__file__).parent / "queries").glob("*.dl")):
    queries[path.stem] = path.read_text()
Path(out).write_text(json.dumps(queries, indent=1))
print(len(queries), "queries")
