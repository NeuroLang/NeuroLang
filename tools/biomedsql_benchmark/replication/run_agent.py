"""System P of PROTOCOL.md: a ReAct agent over the QTL tokens without the held-out dataset.

    python run_agent.py DATA_DIR OUT.jsonl [SHARD NSHARDS]

For each held-out dataset h, an engine holds qtl_dataset, qtl_tested, qtl_significant and
selected_qtl for the other 27 datasets only, so that no query can read h. The agent is
given the engine's guidance (engines/biomedsql/guidance.yaml).
"""

import json
import signal
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ENGINES = Path(__file__).resolve().parents[3] / (
    "packages/neurolang_apps/neurolang_apps/utils/engines"
)
PRICE = {"input": 2.5, "cached": 0.10, "output": 10.0}
THRESHOLD = 2.95e-6
COLUMNS = {
    "qtl_dataset": ["qtl_dataset", "tissue", "kind", "organ"],
    "qtl_tested": ["qtl_dataset", "gene", "disease"],
    "qtl_significant": ["qtl_dataset", "gene", "disease"],
    "selected_qtl": ["qtl_dataset"],
}
DESCRIPTION = (
    "Results of summary-data-based Mendelian randomisation (SMR): which QTL "
    "datasets tested a gene as a mediator of a disease, and which found it "
    "significant. The QTL datasets are equally likely units of evidence."
)


def build(smr, h):
    from neurolang.probabilistic import (
        dalvi_suciu_lift,
        small_dichotomy_theorem_based_solver,
    )
    from neurolang.frontend import NeurolangPDL
    from neurolang_apps.utils.engines.biomedsql.init import organ_of

    nl = NeurolangPDL(
        probabilistic_solvers=(
            small_dichotomy_theorem_based_solver.solve_succ_query,
            dalvi_suciu_lift.solve_succ_query,
        ),
        probabilistic_marg_solvers=(
            small_dichotomy_theorem_based_solver.solve_marg_query,
            dalvi_suciu_lift.solve_marg_query,
        ),
    )
    tests = smr[smr.Omic != h]
    info = tests.drop_duplicates("Omic")
    nl.add_tuple_set(
        [
            (r.Omic, r.Omic_tissue, r.Omic_type, organ_of(r.Omic_tissue))
            for r in info.itertuples()
        ],
        name="qtl_dataset",
    )
    nl.add_tuple_set(tests[["Omic", "Gene", "Disease"]], name="qtl_tested")
    nl.add_tuple_set(
        tests.loc[tests.p_SMR_multi < THRESHOLD, ["Omic", "Gene", "Disease"]],
        name="qtl_significant",
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [(q,) for q in sorted(info.Omic)], name="selected_qtl"
    )
    return nl


class Alarm(Exception):
    pass


def main(data_dir, out_path, shard="0", nshards="1"):
    from neurolang_apps.utils.llm.agent import ReactAgent
    from neurolang_apps.utils.llm.config import load_config

    items = json.load(open(HERE / "items.json"))
    items = [i for k, i in enumerate(items) if k % int(nshards) == int(shard)]
    smr = pd.read_parquet(
        Path(data_dir)
        / "biomedsql/db/NeurodegenerativeDiseases_SMR_Genes_Full.parquet",
        columns=[
            "Omic",
            "Gene",
            "Disease",
            "p_SMR_multi",
            "Omic_tissue",
            "Omic_type",
        ],
    )
    smr = (
        smr.groupby(["Omic", "Gene", "Disease"], observed=True)
        .agg(
            p_SMR_multi=("p_SMR_multi", "min"),
            Omic_tissue=("Omic_tissue", "first"),
            Omic_type=("Omic_type", "first"),
        )
        .reset_index()
    )
    config = load_config(str(HERE.parent / "llm.yaml"))
    a = config.assistant
    a.strategy, a.max_steps, a.verify_rounds, a.comment_rules = (
        "react",
        12,
        0,
        False,
    )
    a.query_timeout, a.query_memory_mb = 300, 30000.0
    provider = config.provider("anthropic-inria")
    guidance = yaml.safe_load(open(ENGINES / "biomedsql/guidance.yaml"))[
        "assistant_guidance"
    ]
    guidance = guidance[guidance.index("A question asking how consistently") :]
    done = (
        {json.loads(l)["id"] for l in open(out_path)}
        if Path(out_path).exists()
        else set()
    )
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Alarm()))
    engines = {}
    with open(out_path, "a") as out:
        for item in sorted(items, key=lambda i: i["heldout"]):
            if item["id"] in done:
                continue
            h = item["heldout"]
            if h not in engines:
                engines = {h: build(smr, h)}
            nl = engines[h]
            q = (
                f"Gene {item['gene']} was tested as a mediator of {item['disease_name']} "
                f"({item['disease']}) in several QTL datasets. What is the probability that a "
                f"QTL dataset finds {item['gene']} significant for {item['disease']}, given that it tested it?"
            )
            rec = {"system": "P", "id": item["id"], "probability": None}
            t0 = time.time()
            try:
                agent = ReactAgent(
                    nl,
                    provider,
                    a,
                    "datalog",
                    engine_name="",
                    loaded=COLUMNS,
                    description=DESCRIPTION,
                    guidance=guidance,
                    examples=[],
                )
                res = agent.run(q)
                u = dict(res.usage)
                rec.update(
                    query=res.query,
                    valid=res.valid,
                    calls=res.calls,
                    usage=u,
                    usd=(
                        (u.get("input", 0) - u.get("cached", 0))
                        * PRICE["input"]
                        + u.get("cached", 0) * PRICE["cached"]
                        + u.get("output", 0) * PRICE["output"]
                    )
                    / 1e6,
                )
                if res.valid:
                    signal.alarm(300)
                    try:
                        with nl.scope:
                            rows = [
                                tuple(r)
                                for r in nl.execute_datalog_program(res.query)
                            ]
                    finally:
                        signal.alarm(0)
                    nums = [r[-1] for r in rows if isinstance(r[-1], float)]
                    rec["n_rows"] = len(rows)
                    rec["probability"] = nums[0] if len(nums) == 1 else None
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["seconds"] = round(time.time() - t0, 1)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(
                item["id"],
                rec["probability"],
                round(item["support"], 4),
                rec.get("error", ""),
                flush=True,
            )


if __name__ == "__main__":
    main(*sys.argv[1:])
