"""What the three arms share: the model route, the schema text, the
uncertainty paragraph and the parsing of a final number.

    uv run --no-sync python arms.py      # (re)writes schema.txt

Every arm calls Sonnet 5.5 through OpenRouter (``anthropic/claude-sonnet-5.5``,
Anthropic as the preferred host), temperature 0, at most 8000 tokens per reply,
with the client of the NeuroLang assistant (``OpenRouterClient``, which also
records each call's host and cost as OpenRouter reports them).

Another model (PROTOCOL.md, amendment 3) is chosen by the environment:
``PQ_MODEL`` (model id), ``PQ_BASE_URL`` (an OpenAI-compatible endpoint, e.g.
vLLM), ``PQ_API_KEY_ENV`` (the variable holding its key) and
``PQ_RESULTS_TAG`` (results go to ``results/<tag>/``). Unset, every arm uses
Sonnet through OpenRouter as above.
"""

import os
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from build_tables import DATA, DB, DETERMINISTIC, ENGINES, UNCERTAIN

HERE = Path(__file__).resolve().parent
MODEL = os.environ.get("PQ_MODEL", "anthropic/claude-sonnet-5.5")
BASE_URL = os.environ.get("PQ_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY_ENV = os.environ.get("PQ_API_KEY_ENV", "OPENROUTER_API_KEY")
RESULTS_TAG = os.environ.get("PQ_RESULTS_TAG", "")
OPENROUTER = "openrouter.ai" in BASE_URL
MAX_TOKENS = 8000
MAX_STEPS = 25
TEMPERATURE = 0.0
# The list price of PROTOCOL.md's amendment, USD per million tokens.
PRICE = {"input": 2.0, "output": 10.0}
QUERY_TIMEOUT = 300           # seconds, a query of S1 or S2
REQUEST_TIMEOUT = 600.0       # seconds, an LLM request
SCHEMA_FILE = HERE / "schema.txt"
SCHEMA_DATALOG_FILE = HERE / "schema_datalog.txt"

DESCRIPTIONS = {
    "test_evidence": "uncertain: the QTL dataset's SMR test is evidence that "
                     "the gene mediates the disease",
    "dataset_reliable": "uncertain: the QTL dataset is reliable",
    "indication_success": "uncertain: the drug succeeds for the indication "
                          "(a MeSH heading of drug_indication)",
    "snp_causal": "uncertain: the GWAS variant (pd_gwas for PD, ad_gwas for "
                  "AD) is a causal SNP of its nearest gene for the disease",
    "drug_target": "the drug (drugName of drug_indication) targets the gene "
                   "(approvedSymbol)",
    "drug_approved": "the drug (drugName of drug) is approved",
    "dataset_organ": "the organ of the QTL dataset's tissue: brain, blood, "
                     "spinal cord, cerebrospinal fluid, liver, muscle or "
                     "nerve",
}

UNCERTAINTY = """\
Four tables are uncertain: each of their rows is an independent random event \
that holds with the probability given in its column prob, and a row that is \
not in the table never holds.
- test_evidence(dataset, gene, disease, prob): the QTL dataset's SMR test is \
evidence that the gene mediates the disease ("the gene has evidence for the \
disease in the dataset").
- dataset_reliable(dataset, prob): the QTL dataset is reliable.
- indication_success(drug, mesh_heading, prob): the drug succeeds for the \
indication.
- snp_causal(snp, gene, disease, prob): the GWAS variant is a causal SNP of \
the gene for the disease (AD or PD).
The other tables are certain: drug_target(drug, gene), the drug targets the \
gene; drug_approved(drug), the drug is approved; dataset_organ(dataset, \
organ), the organ of the QTL dataset (a brain QTL dataset has organ 'brain', \
a blood QTL dataset 'blood'). Drugs are named as in drug_target, genes by \
their symbol, diseases by their code (AD, PD, ALS, FTD, LBD, PSP)."""

# For S2: the same paragraph, the uncertain tables as probabilistic relations.
UNCERTAINTY_DATALOG = (
    UNCERTAINTY
    .replace("Four tables are uncertain: each of their rows",
             "Four relations are probabilistic facts: each of their rows")
    .replace("with the probability given in its column prob",
             "with its own probability (not a column of the relation)")
    .replace(", prob)", ")")
    .replace("The other tables are certain", "The other relations are certain")
)

ANSWER_SQL = (
    "Answer with a single number: end your final reply with a line "
    "'ANSWER: <number>' (a decimal number, no percent sign)."
)
ANSWER_DATALOG = (
    "Answer with a single number: the final query's answer is one row whose "
    "last column is that number."
)


def biomedsql_tables():
    from neurolang_apps.utils.engines.biomedsql.init import TABLES

    return {name: DB / f"{src}.parquet" for name, src in TABLES.items()}


def table_files():
    files = dict(biomedsql_tables())
    for name in {**UNCERTAIN, **DETERMINISTIC}:
        files[name] = DATA / f"{name}.parquet"
    return files


def _value(v):
    """A value as both arms store it: booleans as text, missing text ''."""
    if isinstance(v, bool):
        return str(v)
    return "" if v is None else v


def _sample(path, n=2, width=40, drop=()):
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=n))
    cut = lambda v: (s if len(s := repr(_value(v))) <= width
                     else s[:width] + "...")
    return [
        "(" + ", ".join(cut(v) for k, v in r.items() if k not in drop) + ")"
        for r in batch.to_pylist()
    ]


def write_schema():
    """The schema text of every arm: tables, columns, description, 2 rows;
    for S2 (schema_datalog.txt) the uncertain tables without their column
    prob, which is the probability of the fact."""
    for path, datalog in ((SCHEMA_FILE, False), (SCHEMA_DATALOG_FILE, True)):
        path.write_text(_schema_text(datalog))


def _schema_text(datalog):
    cfg = yaml.safe_load(open(ENGINES / "engines.yaml"))
    preds = cfg["engines"]["biomedsql"]["predicates"]
    lines = ["The database has twelve tables (columns in order):"]
    for name, path in table_files().items():
        cols = pq.read_schema(path).names
        drop = ("prob",) if datalog and name in UNCERTAIN else ()
        cols = [c for c in cols if c not in drop]
        desc = DESCRIPTIONS.get(name) or " ".join(preds[name]["description"].split())
        if drop:
            desc = desc.replace("uncertain:", "probabilistic facts:")
        lines.append(f"- {name}({', '.join(cols)}): {desc}.")
        lines += [f"    e.g. {s}" for s in _sample(path, drop=drop)]
    lines.append(
        "In the five BiomedSQL tables (pd_gwas, ad_gwas, smr, drug, "
        "drug_indication) missing text is the empty string and the booleans "
        "are the text 'True' or 'False'."
    )
    return "\n".join(lines) + "\n"


def schema(datalog=False):
    return (SCHEMA_DATALOG_FILE if datalog else SCHEMA_FILE).read_text()


_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


_AGGREGATES = re.compile(
    r"\b(count|sum|avg|mean|min|max|product|string_agg|list|array_agg|"
    r"bool_or|bool_and|any_value|arg_max|arg_min|median|first|last)\s*\(",
    re.I,
)


def query_size(query, sql=False):
    """The length of a final query: characters and lines; for SQL also the
    JOINs, the sub-SELECTs (SELECTs after the first) and the aggregate
    function calls."""
    if not query:
        return None
    size = {"chars": len(query.strip()),
            "lines": len(query.strip().splitlines())}
    if sql:
        size["joins"] = len(re.findall(r"\bjoin\b", query, re.I))
        size["subselects"] = max(
            0, len(re.findall(r"\bselect\b", query, re.I)) - 1
        )
        size["aggregates"] = len(_AGGREGATES.findall(query))
    return size


def parse_answer(text):
    """The number of the last 'ANSWER: <number>' line, or None."""
    found = re.findall(r"ANSWER\W{0,3}\s*(" + _NUMBER + ")", text or "")
    return float(found[-1]) if found else None


def openrouter_client():
    """The NeuroLang assistant's OpenAI-API client, routed to Anthropic on
    OpenRouter, recording the host and cost of each call in ``calls``."""
    from neurolang_apps.utils.llm.config import ProviderConfig
    from neurolang_apps.utils.llm.providers import OpenAIChatClient

    class OpenRouterClient(OpenAIChatClient):
        calls = None

        def _payload(self, messages, tools, temperature):
            payload = super()._payload(messages, tools, temperature)
            if OPENROUTER:
                payload["provider"] = {"order": ["Anthropic"]}
                payload["usage"] = {"include": True}
            return payload

        def complete(self, messages, tools=None, temperature=0.0):
            t0 = time.perf_counter()
            try:
                return super().complete(messages, tools, temperature)
            finally:
                # The model time of the call, failed or not.
                self.model_seconds += time.perf_counter() - t0

        def _reply(self, data):
            reply = super()._reply(data)
            usage = data.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            n_in = reply.usage.get("input", 0)
            n_out = reply.usage.get("output", 0)
            self.calls.append({
                "host": data.get("provider"),
                "input": n_in,
                "cached": reply.usage.get("cached", 0),
                "output": n_out,
                "reasoning": details.get("reasoning_tokens") or 0,
                "cost": usage.get("cost"),
                "list_cost": (n_in * PRICE["input"] + n_out * PRICE["output"])
                / 1e6,
                "finish": data["choices"][0].get("finish_reason"),
            })
            return reply

    provider = ProviderConfig(
        name="openrouter" if OPENROUTER else "openai-compatible", api="openai",
        model=MODEL, base_url=BASE_URL,
        api_key_env=API_KEY_ENV, tool_calling="native",
        max_tokens=MAX_TOKENS, headers={"X-Title": "NeuroLang"},
    )
    client = OpenRouterClient(provider, REQUEST_TIMEOUT)
    client.calls = []
    client.model_seconds = 0.0
    return provider, client


def usage_of(client):
    """The tokens (input, of which cached, output, of which reasoning), the
    cost (USD, as OpenRouter billed it, ``cost``, and at the protocol's list
    price, ``list_cost``), the model time and the hosts of *client*'s
    calls."""
    calls = client.calls
    out = {k: sum(c[k] or 0 for c in calls)
           for k in ("input", "cached", "output", "reasoning", "cost",
                     "list_cost")}
    out["model_seconds"] = round(client.model_seconds, 2)
    out["calls"] = len(calls)
    out["hosts"] = sorted({str(c["host"]) for c in calls})
    out["cost_missing"] = sum(c["cost"] is None for c in calls)
    return out


if __name__ == "__main__":
    write_schema()
    print(schema(datalog=True))
    print(UNCERTAINTY_DATALOG)
