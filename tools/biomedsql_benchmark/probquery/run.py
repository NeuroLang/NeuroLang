"""Run an arm on the questions, one run per question, resumably.

    uv run --no-sync --with duckdb python run.py ARM [--dev] [--ids A,B] [--max-usd X]

ARM is S0 (Sonnet alone), S1 (Sonnet + SQL) or S2 (Sonnet + probabilistic
Datalog). The scored questions go to ``results/<ARM>.jsonl``, the dev ones
(``--dev``) to ``results/dev/<ARM>.jsonl``; a question already there is not run
again. ``--ids`` runs only those questions; the run stops once the arm's file
has cost ``--max-usd`` (default 40, as OpenRouter bills it). S1 and S2 first
wait for their turn of this Mac's shared heavy-job lock (heavy.py), unless
``--no-lock`` is given (elsewhere, e.g. on Margaret, give it: the lock lives
in /private/tmp; without that directory the lock is skipped anyway). Needs
OPENROUTER_API_KEY.
"""

import argparse
import json
from pathlib import Path

from arms import RESULTS_TAG
HERE = Path(__file__).resolve().parent


def records(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["S0", "S1", "S2"])
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--ids", default="")
    ap.add_argument("--max-usd", type=float, default=40.0)
    ap.add_argument("--no-lock", action="store_true",
                    help="do not wait for the shared heavy-job lock")
    args = ap.parse_args()
    from heavy import heavy_turn

    # S1 (DuckDB scans, 4 threads) and S2 (the 9 GB engine and its query
    # processes) take a turn of the shared heavy-job lock; S0 is light.
    with heavy_turn(f"{args.arm} {'dev' if args.dev else 'scored'} run",
                    enabled=args.arm != "S0" and not args.no_lock
                    and Path("/private/tmp").is_dir()):
        run_arm(args)


def run_arm(args):

    split = "dev" if args.dev else "scored"
    questions = [q for q in json.load(open(HERE / "questions.json"))
                 if q["split"] == split]
    if args.ids:
        wanted = args.ids.split(",")
        questions = [q for q in questions if q["id"] in wanted]
    gold = json.load(open(HERE / "gold.json"))
    out_path = HERE / "results" / RESULTS_TAG / ("dev" if args.dev else "") / f"{args.arm}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = records(out_path)
    seen = {r["id"] for r in done}
    spent = sum(r["usage"]["cost"] for r in done)
    todo = [q for q in questions if q["id"] not in seen]
    if not todo:
        print("nothing to run")
        return

    if args.arm == "S0":
        from sonnet_alone import run
    elif args.arm == "S1":
        from sql_agent import Database, run as run_sql

        db = Database()
        run = lambda question: run_sql(question, db)
    else:
        from datalog_agent import DatalogArm

        run = DatalogArm().run

    with open(out_path, "a") as out:
        for q in todo:
            if spent >= args.max_usd:
                print(f"spend cap {args.max_usd} reached")
                break
            rec = {"id": q["id"], "arm": args.arm, "level": q["level"],
                   "template": q["template"], "split": split}
            rec.update(run(q["question"]))
            spent += rec["usage"]["cost"]
            out.write(json.dumps(rec, default=str) + "\n")
            out.flush()
            g = gold[q["id"]]["gold"]
            print(f"{args.arm} {q['id']}: answer {rec['answer']} gold {g:.6f} "
                  f"steps {rec['steps']} ${rec['usage']['cost']:.3f} "
                  f"{rec['seconds']}s {rec['error'][:80]} (total ${spent:.2f})",
                  flush=True)


if __name__ == "__main__":
    main()
