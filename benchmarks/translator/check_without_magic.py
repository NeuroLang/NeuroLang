"""Answer queries with and without magic sets, and compare the answers.

    python benchmarks/translator/check_without_magic.py ENGINE OUT
        [--file q.dl ...] [--grep REGEX]

Builds the engine once; each query runs in a forked child per mode: as
the frontend runs it, rewritten with magic sets when it can be, and
without them (the rewrite failing, so that the frontend solves the query
itself). A rewrite of magic sets, applied in every mode of the
probabilistic solver, is checked against the program it rewrites this
way. The answers are compared as ``compare_modes.py`` compares them.
"""
import argparse
import os
import pickle
import re
import time
from pathlib import Path

from compare_modes import frames, same


def run_child(nl, name, text, mode, out):
    from nl_core.datalog.exceptions import InvalidMagicSetError

    if mode == "plain":
        def no_magic(query):
            raise InvalidMagicSetError("magic sets turned off")
        nl._magic_rewrite_query = no_magic
    t = time.perf_counter()
    try:
        with nl.scope:
            answer = frames(nl.execute_datalog_program(text)), None
    except Exception as e:  # recorded, compared as such
        answer = None, f"{type(e).__name__}: {e}"[:300]
    with open(out / f"{name}-{mode}.pkl", "wb") as f:
        pickle.dump(answer, f)
    print(f"  {name} {mode}: {time.perf_counter() - t:.1f} s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("out")
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--grep")
    args = parser.parse_args()
    from neurolang_apps.utils import engine_registry

    queries = {Path(f).stem: Path(f).read_text() for f in args.file}
    if args.grep:
        config = engine_registry.get_engine_config(args.engine)
        for key in ("examples", "assistant_examples"):
            for i, ex in enumerate(config.get(key) or []):
                text = ex.get("datalog") if isinstance(ex, dict) else None
                if text and re.search(args.grep, f"{key[:3]}{i:03d}"):
                    queries[f"{key[:3]}{i:03d}"] = text
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nl = engine_registry.build_engine(
        args.engine, Path.home() / "neurolang_data", resolution=2
    )
    for name, text in queries.items():
        for mode in ("magic", "plain"):
            pid = os.fork()
            if pid == 0:
                try:
                    run_child(nl, name, text, mode, out)
                finally:
                    os._exit(0)
            os.waitpid(pid, 0)
        ok, note = same(*(
            pickle.load(open(out / f"{name}-{m}.pkl", "rb"))
            for m in ("magic", "plain")
        ))
        print(f"{name}: {'SAME' if ok else 'DIFFERENT'} ({note})", flush=True)


if __name__ == "__main__":
    main()
