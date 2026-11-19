"""
Run example programs of an engine's example bank on a built engine, time
them and pickle the answers, to compare a branch with upstream master.

    python run_examples.py ENGINE OUT.pkl LIMIT_SECONDS NAME=YAML:KIND:SELECTOR ...

KIND is ``examples`` (SELECTOR: index in examples.datalog) or
``assistant`` (SELECTOR: prefix of the question).  Measurement only.
"""
import pickle
import signal
import sys
import time
from pathlib import Path

import yaml

from neurolang_apps.utils.engine_registry import build_engine


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def program(spec):
    name, rest = spec.split("=", 1)
    path, kind, selector = rest.split(":", 2)
    data = yaml.safe_load(open(path))
    if kind == "examples":
        return name, data["examples"]["datalog"][int(selector)]
    for entry in data["assistant_examples"]:
        if entry["question"].startswith(selector):
            return name, entry["datalog"]
    raise KeyError(selector)


def main():
    engine, out, limit = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    progs = [program(s) for s in sys.argv[4:]]
    nl = build_engine(engine, Path.home() / "neurolang_data", resolution=2)
    signal.signal(signal.SIGALRM, _alarm)
    records = {}
    for name, text in progs:
        error, frames = None, None
        start = time.time()
        signal.alarm(limit)
        try:
            with nl.scope:
                result = nl.execute_datalog_program(text)
            results = result if isinstance(result, dict) else {"ans": result}
            frames = {k: v.as_pandas_dataframe() for k, v in results.items()}
        except Timeout:
            error = f"timeout {limit} s"
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            signal.alarm(0)
        seconds = round(time.time() - start, 1)
        rows = {k: len(v) for k, v in (frames or {}).items()}
        print(f"{name}: {seconds} s, rows={rows}, error={error}", flush=True)
        records[name] = {"seconds": seconds, "error": error, "frames": frames}
        pickle.dump(records, open(out, "wb"))


if __name__ == "__main__":
    main()
