"""
Tally which probabilistic solver of the chain plans each PROB rule of the
example banks of an engine (examples and assistant_examples of its
configuration), with show_ra (plans only, the rules are not run).

    python tally.py ENGINE OUTFILE.json

Measurement only.
"""
import contextlib
import io
import json
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from neurolang_apps.utils.engine_registry import build_engine, get_engine_config

from probtrace import SOLVER_LOG, instrument


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def programs(config):
    out = []
    for i, text in enumerate((config.get("examples") or {}).get("datalog", [])):
        out.append((f"examples[{i}]", text))
    for i, entry in enumerate(config.get("assistant_examples") or []):
        out.append((f"assistant[{i}] {entry.get('question', '')[:70]}",
                    entry.get("datalog", "")))
    return [(name, text) for name, text in out
            if "PROB" in text or "//" in text]


def main():
    engine, outfile = sys.argv[1], Path(sys.argv[2])
    config = get_engine_config(engine)
    progs = programs(config)
    print(f"{len(progs)} programs with PROB", flush=True)
    nl = build_engine(engine, Path.home() / "neurolang_data", resolution=2)
    instrument(nl)
    signal.signal(signal.SIGALRM, _alarm)
    records = []
    for name, text in progs:
        SOLVER_LOG.clear()
        buf = io.StringIO()
        error = None
        t = time.time()
        signal.alarm(120)
        try:
            with contextlib.redirect_stdout(buf):
                with nl.scope:
                    nl.execute_datalog_program(text, show_ra=True)
        except Timeout:
            error = "timeout 120 s"
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            signal.alarm(0)
        # The first solver not raising, per rule.
        per_rule = {}
        for entry in SOLVER_LOG:
            rule = entry["rule"]
            per_rule.setdefault(rule, [])
            per_rule[rule].append(
                (entry["solver"], entry["raised"])
            )
        accepted = {
            rule: next((s for s, r in tries if r is None), None)
            for rule, tries in per_rule.items()
        }
        records.append({
            "name": name, "error": error, "seconds": round(time.time() - t, 2),
            "rules": [
                {"rule": rule, "accepted": accepted[rule], "tries": tries}
                for rule, tries in per_rule.items()
            ],
            "plan": buf.getvalue()[-6000:],
        })
        print(name, error, Counter(accepted.values()), flush=True)
    tally = Counter(
        rule["accepted"] for rec in records for rule in rec["rules"]
    )
    errors = Counter(rec["error"] is not None for rec in records)
    summary = {"programs": len(records), "rules_by_solver": dict(tally),
               "programs_with_error": errors[True]}
    print(json.dumps(summary, indent=1))
    with open(outfile, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1,
                  default=str)


if __name__ == "__main__":
    main()
